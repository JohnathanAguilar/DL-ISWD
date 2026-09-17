"""Global per-block map of the mean-over-cycles variance of high-pass SSHA,
exported as a Cloud-Optimized GeoTIFF (COG) instead of a PNG.

SWOT is a 21-day repeat orbit, so a given *pass number* covers the same ground
track (identical L3 swath grid) in every cycle. Each pass tiles cleanly into
20x20 native-pixel blocks (9800/20 = 490 along-track, 60/20 = 3 cross-track =
1470 blocks). For every block:

  * for each cycle, take the pixels whose ``internal_wave_bbox_confidence`` clears
    a |latitude|-banded cutoff (same rule as ``plot_detection_frequency.py``:
    one ``--threshold`` per band defined by ``--band-edges``), and compute the
    variance of ``gaussian_filter_highpass`` (high-pass SSHA, m) over them;
  * average those per-cycle variances over the ~48 cycles that observed the block.

This is the same per-block quantity ``plot_highpass_variance.py`` scatters at
each block's geographic centre -- but block centres are NOT on a shared grid
(each pass has its own along-track/cross-track tiling), so a raster needs the
scattered blocks binned onto a regular lat/lon grid first (``--resolution``,
degrees; same ``np.histogram2d`` approach as ``plot_global_confidence_map.py``).
Band 1 of the output is the SUM of the per-block variances landing in each
cell; band 2 is the number of blocks binned into that cell (coverage, for QA).
Empty cells are NoData (NaN) in both bands.

Reads the per-pass NetCDFs under ``results/cycle_*/`` directly (the high-pass
field is not in the zarr).

Requires ``rasterio``. On this machine only the ``gis-stuff`` conda env has
it: ``/Users/aguilarj/miniforge3/envs/gis-stuff/bin/python``.

Example
-------
    cd cnn-transformer
    /Users/aguilarj/miniforge3/envs/gis-stuff/bin/python \\
      src/inference/plot_highpass_variance_geotiff.py \\
      --threshold 0.95 0.98 0.998 --band-edges 30 60 90 --workers 8 \\
      --out results/all_cycles/highpass_variance_perblock_banded_0.95-0.98-0.998.tif
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # parallelism is across processes

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/

_FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path, default=_MODEL_DIR / "results",
                   help="Directory containing the per-cycle result folders.")
    p.add_argument("--cycle-glob", default="cycle_*",
                   help="Glob under --results-root selecting the cycle folders.")
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence")
    p.add_argument("--hp-var", default="gaussian_filter_highpass",
                   help="High-pass SSHA variable to take the variance of.")
    p.add_argument("--threshold", type=float, nargs="+", default=[0.95, 0.98, 0.998],
                   metavar="CONF",
                   help="Confidence cutoff: one value, or one per |latitude| band "
                        "(see --band-edges), equator outward.")
    p.add_argument("--band-edges", type=float, nargs="+", default=[30.0, 60.0, 90.0],
                   metavar="DEG",
                   help="Strictly increasing, positive upper edges (deg) of the "
                        "|latitude| bands the per-band thresholds apply to.")
    p.add_argument("--block", type=int, default=20,
                   help="Block edge in native pixels (9800 and 60 must divide it).")
    p.add_argument("--min-pixels", type=int, default=2,
                   help="A block needs this many qualifying pixels for a variance.")
    p.add_argument("--min-cycles", type=int, default=1,
                   help="A block needs a valid variance in this many cycles to be "
                        "kept.")
    p.add_argument("--resolution", type=float, default=0.25,
                   help="Output raster cell size in degrees.")
    p.add_argument("--out", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/highpass_variance_perblock.tif")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only the first N pass numbers (smoke run).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


def wrap_longitude(lon: np.ndarray) -> np.ndarray:
    """0..360 (the SWOT L3 convention) -> [-180, 180)."""
    return ((lon + 180.0) % 360.0) - 180.0


def band_labels(band_edges):
    labels, lo = [], 0.0
    for hi in band_edges:
        labels.append(f"{lo:g}-{hi:g}")
        lo = hi
    return labels


def describe_thresholds(thresholds, band_edges):
    if len(set(thresholds)) == 1:
        return f"conf \N{GREATER-THAN OR EQUAL TO} {thresholds[0]:g}"
    parts = ", ".join(
        f"{lbl}\N{DEGREE SIGN}: {t:g}"
        for lbl, t in zip(band_labels(band_edges), thresholds)
    )
    return f"conf \N{GREATER-THAN OR EQUAL TO} [{parts}] by |lat|"


# ---------------------------------------------------------------------------


def process_pass(recs, conf_var, hp_var, thresholds, band_edges, block,
                 min_pixels, min_cycles):
    """One pass number: mean-over-cycles per-block variance of the high-pass SSHA.

    ``recs`` is a list of (cycle, path). Returns 1-D arrays (lon, lat, mean_var,
    n_cycles) for the blocks kept.
    """
    thr_arr = np.asarray(thresholds, dtype=np.float32)
    inner = np.asarray(band_edges[:-1], dtype=np.float32)
    B = block

    mv_sum = mv_n = blat = blon = None
    for _cyc, path in sorted(recs):
        try:
            with xr.open_dataset(path) as ds:
                hp = np.asarray(ds[hp_var].values, dtype=np.float32)
                conf = np.asarray(ds[conf_var].values, dtype=np.float32)
                lat = np.asarray(ds["latitude"].values, dtype=np.float32)
                lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            print(f"    {Path(path).name}: FAILED: {exc}", flush=True)
            continue

        nl, npx = hp.shape
        nb, mb = nl // B, npx // B
        sl = (slice(0, nb * B), slice(0, mb * B))
        r4 = lambda a: a[sl].reshape(nb, B, mb, B)  # noqa: E731

        thr_px = thr_arr[np.digitize(np.abs(lat), inner)]
        keep = (np.isfinite(hp) & np.isfinite(lat) & np.isfinite(lon)
                & (conf >= thr_px))

        cnt = r4(keep).sum(axis=(1, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
            bvar = np.nanvar(r4(np.where(keep, hp, np.nan)), axis=(1, 3))
        good = (cnt >= min_pixels) & np.isfinite(bvar)

        if mv_sum is None:
            mv_sum = np.zeros((nb, mb), dtype=np.float64)
            mv_n = np.zeros((nb, mb), dtype=np.int32)
        mv_sum[good] += bvar[good]
        mv_n[good] += 1

        if blat is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blat = np.nanmean(r4(lat), axis=(1, 3))
                lonr = np.radians(r4(lon))
                blon = np.degrees(np.arctan2(
                    np.nanmean(np.sin(lonr), axis=(1, 3)),
                    np.nanmean(np.cos(lonr), axis=(1, 3)),
                ))

    if mv_sum is None:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty.astype(np.int32)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_var = np.where(mv_n > 0, mv_sum / np.maximum(mv_n, 1), np.nan)
    sel = (mv_n >= min_cycles) & np.isfinite(mean_var) & np.isfinite(blat) & np.isfinite(blon)
    return (blon[sel].astype(np.float32), blat[sel].astype(np.float32),
            mean_var[sel].astype(np.float32), mv_n[sel].astype(np.int32))


def main(argv=None):
    args = parse_args(argv)

    cycle_dirs = sorted(p for p in args.results_root.glob(args.cycle_glob) if p.is_dir())
    if not cycle_dirs:
        raise FileNotFoundError(
            f"No '{args.cycle_glob}' folders under {args.results_root.resolve()}"
        )

    band_edges = [float(e) for e in args.band_edges]
    if band_edges[0] <= 0 or any(b <= a for a, b in zip(band_edges, band_edges[1:])):
        raise SystemExit("--band-edges must be strictly increasing and positive")
    thresholds = [float(t) for t in args.threshold]
    if len(thresholds) == 1:
        thresholds = thresholds * len(band_edges)
    elif len(thresholds) != len(band_edges):
        raise SystemExit(
            f"got {len(thresholds)} --threshold value(s) for {len(band_edges)} "
            f"|latitude| band(s); pass 1 or {len(band_edges)}"
        )

    by_pass = defaultdict(list)
    n_files = 0
    for cdir in cycle_dirs:
        for nc in cdir.glob("*.nc"):
            m = _FNAME_RE.search(nc.name)
            if not m:
                continue
            by_pass[int(m.group(2))].append((int(m.group(1)), str(nc)))
            n_files += 1
    passes = sorted(by_pass)
    if args.limit:
        passes = passes[: args.limit]
    if not passes:
        raise SystemExit("no passes discovered")

    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 2))
    thr_desc = describe_thresholds(thresholds, band_edges)
    band_desc = " ".join(f"{lbl}\N{DEGREE SIGN}:{t:g}"
                         for lbl, t in zip(band_labels(band_edges), thresholds))
    print(f"results_root={args.results_root.resolve()}")
    print(f"cycles={len(cycle_dirs)} passes={len(passes)} files={n_files} "
          f"block={args.block}px workers={workers}")
    print(f"conf_var={args.conf_var} hp_var={args.hp_var} thresholds[{band_desc}] "
          f"min_pixels={args.min_pixels} min_cycles={args.min_cycles}")

    lons, lats, vals, ns = [], [], [], []
    args_common = (args.conf_var, args.hp_var, thresholds, band_edges, args.block,
                   args.min_pixels, args.min_cycles)
    done = 0
    if workers == 1:
        for p in passes:
            lo, la, v, n = process_pass(by_pass[p], *args_common)
            lons.append(lo); lats.append(la); vals.append(v); ns.append(n)
            done += 1
            if done % 50 == 0 or done == len(passes):
                print(f"  {done}/{len(passes)} passes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pass, by_pass[p], *args_common): p for p in passes}
            for f in as_completed(futs):
                lo, la, v, n = f.result()
                lons.append(lo); lats.append(la); vals.append(v); ns.append(n)
                done += 1
                if done % 50 == 0 or done == len(passes):
                    print(f"  {done}/{len(passes)} passes", flush=True)

    lon = np.concatenate(lons) if lons else np.empty(0, np.float32)
    lat = np.concatenate(lats)
    val = np.concatenate(vals)
    n = np.concatenate(ns)
    if val.size == 0:
        raise SystemExit("no blocks passed the confidence/coverage cuts")

    print(f"kept blocks={val.size:,} | cycles/block: min={n.min()} "
          f"median={int(np.median(n))} max={n.max()}")
    print(f"per-block variance m^2: min={val.min():.4g} median={np.median(val):.4g} "
          f"max={val.max():.4g}")

    res = float(args.resolution)
    if res <= 0:
        raise SystemExit("--resolution must be > 0")
    lat_edges = np.arange(-90.0, 90.0 + res / 2, res)
    lon_edges = np.arange(-180.0, 180.0 + res / 2, res)
    print(f"grid resolution={res}\N{DEGREE SIGN} shape={lat_edges.size - 1}x{lon_edges.size - 1}")

    val_sum, _, _ = np.histogram2d(
        lat.astype(np.float64), lon.astype(np.float64),
        bins=[lat_edges, lon_edges], weights=val.astype(np.float64),
    )
    n_blocks, _, _ = np.histogram2d(
        lat.astype(np.float64), lon.astype(np.float64), bins=[lat_edges, lon_edges],
    )
    empty = n_blocks == 0
    val_sum[empty] = np.nan
    coverage = n_blocks.astype(np.float32)
    coverage[empty] = np.nan

    # histogram2d row 0 = lowest latitude; raster row 0 must be north (top).
    band1 = np.flipud(val_sum).astype(np.float32)
    band2 = np.flipud(coverage).astype(np.float32)

    n_cells = int(np.isfinite(band1).sum())
    print(f"non-empty cells={n_cells:,} | cell-sum variance m^2: "
          f"min={np.nanmin(band1):.4g} median={np.nanmedian(band1):.4g} "
          f"max={np.nanmax(band1):.4g} | coverage max={int(np.nanmax(band2))}")

    transform = from_origin(lon_edges[0], lat_edges[-1], res, res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        args.out, "w", driver="COG",
        height=band1.shape[0], width=band1.shape[1], count=2,
        dtype="float32", crs="EPSG:4326", transform=transform,
        nodata=np.nan, compress="DEFLATE",
    ) as dst:
        dst.write(band1, 1)
        dst.set_band_description(1, "sum_highpass_variance_m2")
        dst.write(band2, 2)
        dst.set_band_description(2, "n_blocks_binned")
        dst.update_tags(
            conf_var=args.conf_var, hp_var=args.hp_var,
            thresholds=thr_desc, block_px=str(args.block),
            min_pixels=str(args.min_pixels), min_cycles=str(args.min_cycles),
            n_cycles=str(len(cycle_dirs)),
        )

    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
