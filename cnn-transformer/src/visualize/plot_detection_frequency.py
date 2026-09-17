"""Global per-block map of internal-wave detection *frequency* across SWOT repeat cycles.

SWOT is a 21-day repeat orbit, so a given *pass number* covers the same ground track (identical
L3 swath grid) in every cycle. Each pass tiles cleanly into 20x20 native-pixel blocks
(9800/20 = 490 along-track, 60/20 = 3 cross-track = 1470 blocks). For every block:

  * for each cycle, **average** ``internal_wave_bbox_confidence`` over the block's pixels; the
    block "counts" for that cycle if the block-mean confidence >= a |latitude|-banded cutoff
    (one ``--threshold`` per band defined by ``--band-edges``, e.g. 0.95 / 0.98 / 0.998 for
    |lat| bands 0-30 / 30-60 / 60-90, applied N and S);
  * the map colour is the number of cycles that cleared it.

Every block is scattered at its geographic centre on a global map -- NO geographic binning /
regridding. Reads the per-pass NetCDFs under ``results/cycle_*/`` directly.

Example
-------
    cd cnn-transformer
    python src/inference/plot_detection_frequency.py \\
      --threshold 0.95 0.98 0.998 --band-edges 30 60 90 --workers 8 \\
      --out results/all_cycles/detection_frequency_perblock_banded_0.95-0.98-0.998.png
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
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/

_FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_")

# named internal-wave regions: name -> (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "AR":  (-48.0, -30.0,   0.0, 15.0),
    "BoB": ( 80.0,  98.5,  -5.0, 15.0),
    "IDN": (113.0, 133.0, -10.0, 10.0),
    "SCS": (110.0, 122.0,  14.0, 23.0),
    "PO":  (-160.0, -72.0,   0.0, 12.0),
    "IO":  ( 40.0,  70.0, -18.0,  0.0),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path, default=_MODEL_DIR / "results",
                   help="Directory containing the per-cycle result folders.")
    p.add_argument("--cycle-glob", default="cycle_*",
                   help="Glob under --results-root selecting the cycle folders.")
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence")
    p.add_argument("--threshold", type=float, nargs="+", default=[0.90, 0.95, 1.0],
                   metavar="CONF",
                   help="Block-mean confidence cutoff: one value, or one per |latitude| "
                        "band (see --band-edges), ordered from the equator outward.")
    p.add_argument("--band-edges", type=float, nargs="+", default=[30.0, 60.0, 90.0],
                   metavar="DEG",
                   help="Strictly increasing, positive upper edges (deg) of the "
                        "|latitude| bands the per-band thresholds apply to.")
    p.add_argument("--block", type=int, default=20,
                   help="Block edge in native pixels (9800 and 60 must divide it).")
    p.add_argument("--min-pixels", type=int, default=1,
                   help="A block needs this many finite conf pixels to be an "
                        "observation that cycle.")
    p.add_argument("--min-count", type=int, default=1,
                   help="Blocks cleared in fewer than this many cycles are left unplotted.")
    p.add_argument("--vmax", type=float, default=None,
                   help="Top of the colour scale. Default: max per-block count.")
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--scatter-size", type=float, default=1.0)
    p.add_argument("--region-stats", action=argparse.BooleanOptionalAction, default=True,
                   help="Print each named region's share of the total detection counts "
                        "and draw its box + label on the map (--no-region-stats to skip).")
    p.add_argument("--out", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/detection_frequency_perblock.png")
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


def process_pass(recs, conf_var, thresholds, band_edges, block, min_pixels, min_count):
    """One pass number: cross-cycle count of blocks whose mean confidence clears the
    |lat|-banded cutoff.

    ``recs`` is a list of (cycle, path). Returns 1-D arrays (lon, lat, count, n_obs)
    for the blocks kept (count >= min_count).
    """
    thr_arr = np.asarray(thresholds, dtype=np.float32)
    inner = np.asarray(band_edges[:-1], dtype=np.float32)
    B = block

    count = n_obs = blat = blon = thr_blk = None
    for _cyc, path in sorted(recs):
        try:
            with xr.open_dataset(path) as ds:
                conf = np.asarray(ds[conf_var].values, dtype=np.float32)
                lat = np.asarray(ds["latitude"].values, dtype=np.float32)
                lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            print(f"    {Path(path).name}: FAILED: {exc}", flush=True)
            continue

        nl, npx = conf.shape
        nb, mb = nl // B, npx // B
        sl = (slice(0, nb * B), slice(0, mb * B))
        r4 = lambda a: a[sl].reshape(nb, B, mb, B)  # noqa: E731

        cnt = np.isfinite(r4(conf)).sum(axis=(1, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks
            bmean = np.nanmean(r4(conf), axis=(1, 3))

        if count is None:
            count = np.zeros((nb, mb), dtype=np.int32)
            n_obs = np.zeros((nb, mb), dtype=np.int32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blat = np.nanmean(r4(lat), axis=(1, 3))
                lonr = np.radians(r4(lon))
                blon = np.degrees(np.arctan2(
                    np.nanmean(np.sin(lonr), axis=(1, 3)),
                    np.nanmean(np.cos(lonr), axis=(1, 3)),
                ))
            thr_blk = thr_arr[np.digitize(np.abs(np.nan_to_num(blat)), inner)]

        obs = cnt >= min_pixels
        n_obs += obs
        count += obs & np.isfinite(bmean) & (bmean >= thr_blk)

    if count is None:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty.astype(np.int32), empty.astype(np.int32)

    sel = (count >= min_count) & np.isfinite(blat) & np.isfinite(blon)
    return (blon[sel].astype(np.float32), blat[sel].astype(np.float32),
            count[sel].astype(np.int32), n_obs[sel].astype(np.int32))


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
    print(f"conf_var={args.conf_var} thresholds[{band_desc}] "
          f"min_pixels={args.min_pixels} min_count={args.min_count}")

    lons, lats, cnts, nobs = [], [], [], []
    common = (args.conf_var, thresholds, band_edges, args.block, args.min_pixels,
              args.min_count)
    done = 0
    if workers == 1:
        for p in passes:
            lo, la, c, no = process_pass(by_pass[p], *common)
            lons.append(lo); lats.append(la); cnts.append(c); nobs.append(no)
            done += 1
            if done % 50 == 0 or done == len(passes):
                print(f"  {done}/{len(passes)} passes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pass, by_pass[p], *common): p for p in passes}
            for f in as_completed(futs):
                lo, la, c, no = f.result()
                lons.append(lo); lats.append(la); cnts.append(c); nobs.append(no)
                done += 1
                if done % 50 == 0 or done == len(passes):
                    print(f"  {done}/{len(passes)} passes", flush=True)

    lon = np.concatenate(lons) if lons else np.empty(0, np.float32)
    lat = np.concatenate(lats)
    cnt = np.concatenate(cnts)
    nob = np.concatenate(nobs)
    if cnt.size == 0:
        raise SystemExit("no blocks cleared the cutoff in >= --min-count cycles")

    order = np.argsort(cnt, kind="stable")  # highest count drawn last / on top
    lon, lat, cnt, nob = lon[order], lat[order], cnt[order], nob[order]

    vmax = int(args.vmax) if args.vmax is not None else int(cnt.max())
    print(f"kept blocks={cnt.size:,} | count: median={int(np.median(cnt))} "
          f"max={int(cnt.max())} | observed cycles/block: median={int(np.median(nob))} "
          f"max={int(nob.max())} | vmax={vmax}")

    region_rows = []
    if args.region_stats:
        total = int(cnt.astype(np.int64).sum())
        print(f"region share of total detection counts ({total:,}):")
        for name, (lo, hi, la0, la1) in REGIONS.items():
            m = (lon >= lo) & (lon < hi) & (lat >= la0) & (lat < la1)
            rc = int(cnt[m].astype(np.int64).sum())
            pct = 100.0 * rc / total if total else 0.0
            print(f"  {name:4s} lon[{lo:g},{hi:g}] lat[{la0:g},{la1:g}]  "
                  f"blocks={int(m.sum()):>6,}  counts={rc:>10,}  {pct:6.2f}%")
            region_rows.append((name, lo, hi, la0, la1, pct))

    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="0.85", zorder=0)
    ax.coastlines(linewidth=0.4)

    sc = ax.scatter(lon, lat, c=cnt, s=args.scatter_size, marker="s", linewidths=0,
                    cmap=args.cmap, vmin=args.min_count, vmax=max(vmax, args.min_count),
                    transform=ccrs.PlateCarree(), rasterized=True)

    for name, lo, hi, la0, la1, pct in region_rows:
        ax.add_patch(mpatches.Rectangle(
            (lo, la0), hi - lo, la1 - la0, transform=ccrs.PlateCarree(),
            fill=False, edgecolor="#ed0909", linewidth=1.1, zorder=5))
        ax.text(lo, la1 + 1.0, f"{name} {pct:.1f}%", transform=ccrs.PlateCarree(),
                fontsize=8, color="#ed0909", weight="bold", va="bottom", zorder=6,
                clip_on=True)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.6", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, pad=0.02,
                 label=f"cycles where the {args.block}\N{MULTIPLICATION SIGN}{args.block}-px "
                       f"block-mean confidence clears the cutoff ({thr_desc})")
    ax.set_title(
        f"SWOT internal-wave detection frequency per {args.block}\N{MULTIPLICATION SIGN}{args.block} "
        f"block  |  {len(cycle_dirs)} cycles  |  {thr_desc}"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
