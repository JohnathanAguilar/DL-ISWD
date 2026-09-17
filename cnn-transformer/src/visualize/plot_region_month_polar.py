"""Per-region seasonal polar ("radar") plots of normalized ISW count + amplitude variance.

For each of the six named regions (AR, BoB, IDN, SCS, PO, IO -- boxes imported from
``plot_detection_frequency.REGIONS``) a polar subplot is drawn where:

  * the ANGLE is the calendar month (Jan at top, clockwise);
  * the RADIUS shows two normalized 12-value series:
      - ISW count      -- per month, the number of 20x20 blocks whose block-mean
                          ``internal_wave_bbox_confidence`` clears the |lat|-banded cutoff,
                          summed over that month's passes (the plot_detection_frequency metric);
      - amplitude var. -- per month, the mean of the per-block high-pass-SSHA variance
                          (the plot_highpass_variance metric) over the qualifying blocks.

Both series are normalized by a GLOBAL scale: divided by the global maximum of that same metric
computed over the whole globe by calendar month (busiest month anywhere -> ~1.0).

Reads the per-pass NetCDFs under ``results/cycle_*/`` directly; month comes from the filename
timestamp. Aggregates over all 50 cycles pooled by calendar month.

Example
-------
    cd cnn-transformer
    python src/inference/plot_region_month_polar.py \\
      --threshold 0.90 0.95 1.0 --band-edges 30 60 90 --workers 8 \\
      --out results/all_cycles/region_month_polar_banded_0.90-0.95-1.0.png
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

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import xarray as xr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_detection_frequency import (
    REGIONS, wrap_longitude, band_labels, describe_thresholds,
)

_INFERENCE_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _INFERENCE_DIR.parent.parent  # cnn-transformer/

# cycle, pass, YYYYMMDD
_FNAME_RE = re.compile(r"SWOT_L3_LR_SSH_Expert_(\d{3})_(\d{3})_(\d{8})T")
_MONTHS = "JFMAMJJASOND"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path, default=_MODEL_DIR / "results",
                   help="Directory containing the per-cycle result folders.")
    p.add_argument("--cycle-glob", default="cycle_*",
                   help="Glob under --results-root selecting the cycle folders.")
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence")
    p.add_argument("--hp-var", default="gaussian_filter_highpass")
    p.add_argument("--threshold", type=float, nargs="+", default=[0.90, 0.95, 1.0],
                   metavar="CONF",
                   help="Block-mean / per-pixel confidence cutoff: one value or one per "
                        "|latitude| band (see --band-edges), equator outward.")
    p.add_argument("--band-edges", type=float, nargs="+", default=[30.0, 60.0, 90.0],
                   metavar="DEG")
    p.add_argument("--block", type=int, default=20,
                   help="Block edge in native pixels (9800 and 60 must divide it).")
    p.add_argument("--min-pixels", type=int, default=2,
                   help="A block needs this many per-pixel-gated pixels for a variance.")
    p.add_argument("--out", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/region_month_polar.png")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only the first N pass numbers (smoke run).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------


def process_pass(recs, conf_var, hp_var, thresholds, band_edges, block, min_pixels):
    """One pass number over all its cycle files -> month-indexed accumulators.

    Returns count_rm(NR,12), count_gm(12), var_sum_rm(NR,12), var_n_rm(NR,12),
    var_sum_gm(12), var_n_gm(12).
    """
    NR = len(REGIONS)
    boxes = [b for b in REGIONS.values()]
    thr_arr = np.asarray(thresholds, dtype=np.float32)
    inner = np.asarray(band_edges[:-1], dtype=np.float32)
    B = block

    count_rm = np.zeros((NR, 12), dtype=np.int64)
    count_gm = np.zeros(12, dtype=np.int64)
    var_sum_rm = np.zeros((NR, 12), dtype=np.float64)
    var_n_rm = np.zeros((NR, 12), dtype=np.int64)
    var_sum_gm = np.zeros(12, dtype=np.float64)
    var_n_gm = np.zeros(12, dtype=np.int64)

    ridx = thr_blk = None
    for _cyc, path in sorted(recs):
        m = _FNAME_RE.search(Path(path).name)
        if not m:
            continue
        mi = int(m.group(3)[4:6]) - 1  # 0..11
        try:
            with xr.open_dataset(path) as ds:
                conf = np.asarray(ds[conf_var].values, dtype=np.float32)
                hp = np.asarray(ds[hp_var].values, dtype=np.float32)
                lat = np.asarray(ds["latitude"].values, dtype=np.float32)
                lon = wrap_longitude(np.asarray(ds["longitude"].values, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            print(f"    {Path(path).name}: FAILED: {exc}", flush=True)
            continue

        nl, npx = conf.shape
        nb, mb = nl // B, npx // B
        sl = (slice(0, nb * B), slice(0, mb * B))
        r4 = lambda a: a[sl].reshape(nb, B, mb, B)  # noqa: E731

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bmean = np.nanmean(r4(conf), axis=(1, 3))

        if ridx is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blat = np.nanmean(r4(lat), axis=(1, 3))
                lonr = np.radians(r4(lon))
                blon = np.degrees(np.arctan2(
                    np.nanmean(np.sin(lonr), axis=(1, 3)),
                    np.nanmean(np.cos(lonr), axis=(1, 3)),
                ))
            thr_blk = thr_arr[np.digitize(np.abs(np.nan_to_num(blat)), inner)]
            ridx = np.full(bmean.shape, -1, dtype=np.int8)
            for ri, (lo, hi, la0, la1) in enumerate(boxes):
                inb = (blon >= lo) & (blon < hi) & (blat >= la0) & (blat < la1)
                ridx[inb] = ri

        cleared = bmean >= thr_blk
        count_gm[mi] += int(cleared.sum())
        for ri in range(NR):
            count_rm[ri, mi] += int(cleared[ridx == ri].sum())

        thr_px = thr_arr[np.digitize(np.abs(lat), inner)]
        keep = (np.isfinite(hp) & np.isfinite(lat) & np.isfinite(lon)
                & (conf >= thr_px))
        cnt = r4(keep).sum(axis=(1, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bvar = np.nanvar(r4(np.where(keep, hp, np.nan)), axis=(1, 3))
        valid = (cnt >= min_pixels) & np.isfinite(bvar)

        var_sum_gm[mi] += float(bvar[valid].sum())
        var_n_gm[mi] += int(valid.sum())
        for ri in range(NR):
            vv = valid & (ridx == ri)
            var_sum_rm[ri, mi] += float(bvar[vv].sum())
            var_n_rm[ri, mi] += int(vv.sum())

    return count_rm, count_gm, var_sum_rm, var_n_rm, var_sum_gm, var_n_gm


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
    names = list(REGIONS)
    NR = len(names)
    print(f"results_root={args.results_root.resolve()}")
    print(f"cycles={len(cycle_dirs)} passes={len(passes)} files={n_files} "
          f"block={args.block}px workers={workers}")
    print(f"thresholds[{band_desc}] min_pixels={args.min_pixels} regions={names}")

    count_rm = np.zeros((NR, 12), dtype=np.int64)
    count_gm = np.zeros(12, dtype=np.int64)
    var_sum_rm = np.zeros((NR, 12), dtype=np.float64)
    var_n_rm = np.zeros((NR, 12), dtype=np.int64)
    var_sum_gm = np.zeros(12, dtype=np.float64)
    var_n_gm = np.zeros(12, dtype=np.int64)

    common = (args.conf_var, args.hp_var, thresholds, band_edges, args.block, args.min_pixels)
    done = 0

    def _fold(res):
        nonlocal count_rm, count_gm, var_sum_rm, var_n_rm, var_sum_gm, var_n_gm
        a, b, c, d, e, f = res
        count_rm += a; count_gm += b
        var_sum_rm += c; var_n_rm += d
        var_sum_gm += e; var_n_gm += f

    if workers == 1:
        for p in passes:
            _fold(process_pass(by_pass[p], *common))
            done += 1
            if done % 50 == 0 or done == len(passes):
                print(f"  {done}/{len(passes)} passes", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pass, by_pass[p], *common): p for p in passes}
            for fut in as_completed(futs):
                _fold(fut.result())
                done += 1
                if done % 50 == 0 or done == len(passes):
                    print(f"  {done}/{len(passes)} passes", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        V_rm = np.where(var_n_rm > 0, var_sum_rm / np.maximum(var_n_rm, 1), np.nan)

    # normalize each series WITHIN its region: divide by that region's own
    # 12-month max, so every polar plot shows the seasonal shape peaking at 1.0.
    c_row_max = np.maximum(count_rm.max(axis=1, keepdims=True), 1)
    Cn = count_rm / c_row_max
    with np.errstate(invalid="ignore", divide="ignore"):
        v_row_max = np.nanmax(V_rm, axis=1, keepdims=True)
        Vn = V_rm / np.where(v_row_max > 0, v_row_max, np.nan)

    print(f"global monthly cleared-block count (all blocks): {count_gm.tolist()}")
    print("per region: peak month  |  raw peak count  |  raw peak mean per-block variance")
    for ri, nm in enumerate(names):
        cpk = int(np.argmax(count_rm[ri]))
        vpk = int(np.nanargmax(V_rm[ri])) if np.isfinite(V_rm[ri]).any() else 0
        print(f"  {nm:4s}  count peak {_MONTHS[cpk]} ({int(count_rm[ri, cpk]):,})   "
              f"var peak {_MONTHS[vpk]} ({V_rm[ri, vpk]:.4g} m^2)")

    # ---- figure ----
    # both series are normalized within-region to [0, 1]; fixed 0..1 radial axis.
    theta = np.deg2rad(np.arange(12) * 30.0)
    th_c = np.concatenate([theta, theta[:1]])
    c_col, v_col = "#1f77b4", "#d62728"

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={"projection": "polar"})
    handles = []
    for ri, ax in enumerate(axes.flat):
        if ri >= NR:
            ax.set_visible(False)
            continue
        cr = np.concatenate([Cn[ri], Cn[ri, :1]])
        vr = np.nan_to_num(np.concatenate([Vn[ri], Vn[ri, :1]]))
        l1, = ax.plot(th_c, cr, color=c_col, lw=1.8, marker="o", ms=3)
        ax.fill(th_c, cr, color=c_col, alpha=0.12)
        l2, = ax.plot(th_c, vr, color=v_col, lw=1.8, marker="o", ms=3)
        ax.fill(th_c, vr, color=v_col, alpha=0.12)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(theta)
        ax.set_xticklabels([str(mm) for mm in range(1, 13)], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_rticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["", "0.5", "", "1.0"] if ri == 0 else [], fontsize=7)
        ax.set_title(names[ri], fontsize=13, pad=14)
        ax.grid(alpha=0.4)
        if ri == 0:
            handles = [l1, l2]

    fig.legend(handles,
               ["ISW count (norm. within region)", "amplitude variance (norm. within region)"],
               loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(f"Seasonal ISW activity by region (each series scaled to its own regional max)"
                 f"  |  {thr_desc}  |  {len(cycle_dirs)} cycles", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
