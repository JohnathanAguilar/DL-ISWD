"""Global per-block map of the mean-over-cycles variance of high-pass SSHA.

SWOT is a 21-day repeat orbit, so a given *pass number* covers the same ground
track (identical L3 swath grid) in every cycle. Each pass tiles cleanly into
20x20 native-pixel blocks (9800/20 = 490 along-track, 60/20 = 3 cross-track =
1470 blocks). For every block:

  * for each cycle, take the pixels whose ``internal_wave_bbox_confidence`` clears
    a |latitude|-banded cutoff (same rule as ``plot_detection_frequency.py``:
    one ``--threshold`` per band defined by ``--band-edges``), and compute the
    variance of ``gaussian_filter_highpass`` (high-pass SSHA, m) over them;
  * average those per-cycle variances over the ~48 cycles that observed the block.

Every retained block is scattered at its geographic centre on a global map
(NO geographic binning / regridding -- the plotted unit is the raw 20x20 block).

Reads the per-pass NetCDFs under ``results/cycle_*/`` directly (the high-pass
field is not in the zarr).

Example
-------
    cd cnn-transformer
    python src/inference/plot_highpass_variance.py \\
      --threshold 0.95 0.98 0.998 --band-edges 30 60 90 --workers 8 \\
      --out results/all_cycles/highpass_variance_perblock_banded_0.95-0.98-0.998.png
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
                        "plotted.")
    p.add_argument("--vmin", type=float, default=None,
                   help="Bottom of the colour scale (raw m^2). Default: 0, or the "
                        "low --clim-quantile if given.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Top of the colour scale (raw m^2). Default: 98th pct of "
                        "block values, or the high --clim-quantile if given.")
    p.add_argument("--clim-quantile", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Set the colour limits from two quantiles of the plotted "
                        "block values, fractions in [0,1] (e.g. 0.2 0.98). "
                        "--vmin / --vmax override the matching end.")
    p.add_argument("--log", action="store_true", help="Log colour scale.")
    p.add_argument("--cmap", default="magma")
    p.add_argument("--scatter-size", type=float, default=1.0)
    p.add_argument("--out", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/highpass_variance_perblock.png")
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

    order = np.argsort(val, kind="stable")  # strongest blocks drawn last
    lon, lat, val, n = lon[order], lat[order], val[order], n[order]

    q_lo, q_hi = args.clim_quantile if args.clim_quantile else (None, None)
    if args.clim_quantile and not (0.0 <= q_lo < q_hi <= 1.0):
        raise SystemExit("--clim-quantile LO HI must satisfy 0 <= LO < HI <= 1")
    vmin = (args.vmin if args.vmin is not None
            else float(np.quantile(val, q_lo)) if q_lo is not None else 0.0)
    vmax = (args.vmax if args.vmax is not None
            else float(np.quantile(val, q_hi)) if q_hi is not None
            else float(np.quantile(val, 0.98)))
    if vmax <= vmin:
        vmax = vmin + (abs(vmin) or 1.0) * 1e-3

    print(f"kept blocks={val.size:,} | cycles/block: min={n.min()} "
          f"median={int(np.median(n))} max={n.max()}")
    print(f"variance m^2: min={val.min():.4g} median={np.median(val):.4g} "
          f"max={val.max():.4g} | vmin={vmin:.4g} vmax={vmax:.4g} | log={args.log}")

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.coastlines(linewidth=0.4)

    if args.log:
        pos = val > 0
        lon, lat, val = lon[pos], lat[pos], val[pos]
        vmin_log = vmin if vmin > 0 else max(float(np.quantile(val, 0.02)), vmax * 1e-4)
        norm = LogNorm(vmin=vmin_log, vmax=max(vmax, vmin_log * 1.001))
        sc = ax.scatter(lon, lat, c=val, s=args.scatter_size, marker="s", linewidths=0,
                        cmap=args.cmap, norm=norm, transform=ccrs.PlateCarree(),
                        rasterized=True)
    else:
        sc = ax.scatter(lon, lat, c=val, s=args.scatter_size, marker="s", linewidths=0,
                        cmap=args.cmap, vmin=vmin, vmax=vmax,
                        transform=ccrs.PlateCarree(), rasterized=True)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                 label="mean-over-cycles variance of high-pass SSHA "
                       f"in {args.block}\N{MULTIPLICATION SIGN}{args.block}-px blocks (m\N{SUPERSCRIPT TWO})")
    ax.set_title(
        f"SWOT high-pass SSHA variance per {args.block}\N{MULTIPLICATION SIGN}{args.block} block  |  "
        f"{len(cycle_dirs)} cycles  |  {thr_desc}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
