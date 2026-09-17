"""Per-pass, per-pixel highpass-SSHA variance NetCDFs (mean + std across cycles).

For every SWOT pass number, and for every 20x20-pixel block (same tiling as
``plot_highpass_variance.py``): for each cycle, take the pixels whose
``internal_wave_bbox_confidence`` clears a |latitude|-banded threshold (same
threshold/band-edges convention as ``plot_detection_frequency.py`` and
``plot_highpass_variance.py``), and compute the variance of
``gaussian_filter_highpass`` (high-pass SSHA, m) over just those pixels --
one variance value per block per cycle. ``plot_highpass_variance.py`` only
ever averages that per-cycle variance across the ~48-50 cycles that observed
a block, for a plot; this script also computes the STANDARD DEVIATION across
cycles of that same per-cycle variance, and saves both.

Each block's mean/std is broadcast to every pixel in that block's 20x20
footprint, so the output is at NATIVE pixel resolution (``num_lines`` x
``num_pixels``, matching ``detection_frequency_pass*.nc``'s convention),
not the smaller 490x3 block grid.

Writes one NetCDF per pass with ``highpass_variance_mean`` (m^2),
``highpass_variance_std`` (m^2), and ``n_cycles_observed`` (cycles that
contributed a valid variance to that pixel's block).

Example
-------
    cd cnn-transformer/src/visualize
    python make_highpass_variance_netcdf.py --threshold 0.90 0.95 1.0 \\
      --band-edges 30 60 90 --block 20 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import xarray as xr

from plot_detection_frequency import (
    _FNAME_RE, band_labels, describe_thresholds, wrap_longitude,
)

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent.parent  # cnn-transformer/

COMPRESSION_ENCODING = {"zlib": True, "complevel": 4}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results-root", type=Path, default=_MODEL_DIR / "results")
    p.add_argument("--cycle-glob", default="cycle_*")
    p.add_argument("--conf-var", default="internal_wave_bbox_confidence")
    p.add_argument("--hp-var", default="gaussian_filter_highpass",
                   help="High-pass SSHA variable to take the variance of.")
    p.add_argument("--threshold", type=float, nargs="+", default=[0.90, 0.95, 1.0],
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
                        "written (std additionally needs >= 2 cycles, else NaN).")
    p.add_argument("--out-dir", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/highpass_variance_netcdf")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only the first N pass numbers (smoke run).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


def process_pass_variance(recs, conf_var, hp_var, thresholds, band_edges, block,
                          min_pixels, min_cycles):
    """One pass: mean- and std-over-cycles per-block variance of the high-pass
    SSHA, broadcast to native pixel resolution.

    ``recs`` is a list of (cycle, path). Returns (mean_px, std_px, n_cycles_px,
    lat, lon) at native pixel resolution, or None if every file in ``recs``
    failed to open.
    """
    thr_arr = np.asarray(thresholds, dtype=np.float32)
    inner = np.asarray(band_edges[:-1], dtype=np.float32)
    B = block

    mv_sum = mv_sumsq = mv_n = lat_ref = lon_ref = None
    nb = mb = None
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

        if lat_ref is None:
            lat_ref, lon_ref = lat, lon

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
            mv_sumsq = np.zeros((nb, mb), dtype=np.float64)
            mv_n = np.zeros((nb, mb), dtype=np.int32)
        mv_sum[good] += bvar[good]
        mv_sumsq[good] += bvar[good].astype(np.float64) ** 2
        mv_n[good] += 1

    if mv_sum is None:
        return None

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_var = np.where(mv_n > 0, mv_sum / np.maximum(mv_n, 1), np.nan)
        var_of_var = mv_sumsq / np.maximum(mv_n, 1) - mean_var ** 2
        std_var = np.where(mv_n >= 2, np.sqrt(np.clip(var_of_var, 0, None)), np.nan)
    sel = mv_n >= min_cycles
    mean_var = np.where(sel, mean_var, np.nan)
    std_var = np.where(sel, std_var, np.nan)
    n_cycles_blk = np.where(sel, mv_n, 0)

    nl, npx = lat_ref.shape
    mean_px = np.full((nl, npx), np.nan, dtype=np.float32)
    std_px = np.full((nl, npx), np.nan, dtype=np.float32)
    n_cycles_px = np.zeros((nl, npx), dtype=np.int32)
    mean_px[: nb * B, : mb * B] = np.repeat(np.repeat(mean_var, B, axis=0), B, axis=1)
    std_px[: nb * B, : mb * B] = np.repeat(np.repeat(std_var, B, axis=0), B, axis=1)
    n_cycles_px[: nb * B, : mb * B] = np.repeat(np.repeat(n_cycles_blk, B, axis=0), B, axis=1)

    return mean_px, std_px, n_cycles_px, lat_ref, lon_ref


def write_pass_netcdf(pas, mean_px, std_px, n_cycles_px, lat, lon, n_cycles_total,
                      conf_var, hp_var, thr_desc, block, out_dir):
    ds = xr.Dataset(
        {
            "highpass_variance_mean": (
                ("num_lines", "num_pixels"), mean_px,
                {"long_name": f"mean over cycles of the {block}x{block}-px block "
                               f"variance of {hp_var}", "units": "m^2"},
            ),
            "highpass_variance_std": (
                ("num_lines", "num_pixels"), std_px,
                {"long_name": f"std across cycles of the {block}x{block}-px block "
                               f"variance of {hp_var}", "units": "m^2"},
            ),
            "n_cycles_observed": (
                ("num_lines", "num_pixels"), n_cycles_px.astype(np.uint8),
                {"long_name": "cycles that contributed a valid variance to this "
                               "pixel's block"},
            ),
        },
        coords={
            "latitude": (("num_lines", "num_pixels"), lat),
            "longitude": (("num_lines", "num_pixels"), lon),
        },
        attrs={
            "pass": int(pas),
            "n_cycles": int(n_cycles_total),
            "conf_var": conf_var,
            "hp_var": hp_var,
            "thresholds": thr_desc,
            "block_px": int(block),
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"highpass_variance_pass{pas:03d}.nc"
    ds.to_netcdf(out_path, encoding={v: COMPRESSION_ENCODING for v in ds.data_vars})
    return out_path


def _worker(pas, recs, conf_var, hp_var, thresholds, band_edges, block, min_pixels,
           min_cycles, out_dir, thr_desc):
    result = process_pass_variance(recs, conf_var, hp_var, thresholds, band_edges,
                                   block, min_pixels, min_cycles)
    if result is None:
        print(f"  pass {pas:03d}: no readable files, skipped", flush=True)
        return None
    mean_px, std_px, n_cycles_px, lat, lon = result
    out_path = write_pass_netcdf(
        pas, mean_px, std_px, n_cycles_px, lat, lon, len(recs), conf_var, hp_var,
        thr_desc, block, out_dir,
    )
    n_finite = int(np.isfinite(mean_px).sum())
    print(f"  pass {pas:03d}: cycles={len(recs)} blocks_with_var={n_finite // (block * block):,} "
          f"-> {out_path.name}", flush=True)
    return out_path


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
    print(f"out_dir={args.out_dir.resolve()}")

    common = (args.conf_var, args.hp_var, thresholds, band_edges, args.block,
              args.min_pixels, args.min_cycles, args.out_dir, thr_desc)
    written = []
    if workers == 1:
        for p in passes:
            out = _worker(p, by_pass[p], *common)
            if out is not None:
                written.append(out)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, p, by_pass[p], *common): p for p in passes}
            for f in as_completed(futs):
                out = f.result()
                if out is not None:
                    written.append(out)

    total_bytes = sum(p.stat().st_size for p in written)
    print(f"\nwrote {len(written)}/{len(passes)} passes "
          f"({total_bytes / 1e6:.1f} MB) to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
