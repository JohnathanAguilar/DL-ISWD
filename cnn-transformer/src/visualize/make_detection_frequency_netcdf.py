"""Per-pass, per-pixel detection-frequency NetCDFs.

For every SWOT pass number, count -- at NATIVE pixel resolution, no block
averaging -- how many of that pass's cycles had ``internal_wave_bbox_confidence``
clear a |latitude|-banded threshold (same threshold/band-edges convention as
``plot_detection_frequency.py``'s ``process_pass()``, just applied per pixel
instead of per 20x20 block-mean: confidence is already stored at native pixel
resolution in the slim result files, so no reshape is needed).

Writes one NetCDF per pass (dims/coords matching the source slim files:
``num_lines`` x ``num_pixels``, 2-D ``latitude``/``longitude``) with
``detection_count`` (cycles that cleared the threshold at that pixel) and
``n_cycles_observed`` (cycles with a finite confidence value there).

Example
-------
    cd cnn-transformer/src/visualize
    python make_detection_frequency_netcdf.py --threshold 0.95 0.98 0.998 \\
      --band-edges 30 60 90 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
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
    p.add_argument("--threshold", type=float, nargs="+", default=[0.90, 0.95, 1.0],
                   metavar="CONF",
                   help="Confidence cutoff: one value, or one per |latitude| band "
                        "(see --band-edges), equator outward.")
    p.add_argument("--band-edges", type=float, nargs="+", default=[30.0, 60.0, 90.0],
                   metavar="DEG",
                   help="Strictly increasing, positive upper edges (deg) of the "
                        "|latitude| bands the per-band thresholds apply to.")
    p.add_argument("--out-dir", type=Path,
                   default=_MODEL_DIR / "results/all_cycles/detection_frequency_netcdf")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only the first N pass numbers (smoke run).")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


def process_pass_pixels(recs, conf_var, thresholds, band_edges):
    """One pass: per-pixel cross-cycle count of confidence >= its |lat|-banded
    threshold. ``recs`` is a list of (cycle, path). Returns
    (count, n_obs, lat, lon) at native pixel resolution, or None if every
    file in ``recs`` failed to open."""
    thr_arr = np.asarray(thresholds, dtype=np.float32)
    inner = np.asarray(band_edges[:-1], dtype=np.float32)

    count = n_obs = lat_ref = lon_ref = None
    for _cyc, path in sorted(recs):
        try:
            with xr.open_dataset(path) as ds:
                conf = np.asarray(ds[conf_var].values, dtype=np.float32)
                lat = np.asarray(ds["latitude"].values, dtype=np.float32)
                lon = np.asarray(ds["longitude"].values, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"    {Path(path).name}: FAILED: {exc}", flush=True)
            continue

        if count is None:
            count = np.zeros(conf.shape, dtype=np.int32)
            n_obs = np.zeros(conf.shape, dtype=np.int32)
            lat_ref, lon_ref = lat, wrap_longitude(lon)

        thr_px = thr_arr[np.digitize(np.abs(lat), inner)]
        obs = np.isfinite(conf)
        n_obs += obs
        count += obs & (conf >= thr_px)

    if count is None:
        return None
    return count, n_obs, lat_ref, lon_ref


def write_pass_netcdf(pas, count, n_obs, lat, lon, n_cycles, conf_var, thr_desc, out_dir):
    ds = xr.Dataset(
        {
            "detection_count": (
                ("num_lines", "num_pixels"), count.astype(np.uint8),
                {"long_name": "cycles where confidence cleared the |lat|-banded "
                               "threshold at this pixel"},
            ),
            "n_cycles_observed": (
                ("num_lines", "num_pixels"), n_obs.astype(np.uint8),
                {"long_name": "cycles with a finite confidence value at this pixel"},
            ),
        },
        coords={
            "latitude": (("num_lines", "num_pixels"), lat),
            "longitude": (("num_lines", "num_pixels"), lon),
        },
        attrs={
            "pass": int(pas),
            "n_cycles": int(n_cycles),
            "conf_var": conf_var,
            "thresholds": thr_desc,
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"detection_frequency_pass{pas:03d}.nc"
    ds.to_netcdf(out_path, encoding={v: COMPRESSION_ENCODING for v in ds.data_vars})
    return out_path


def _worker(pas, recs, conf_var, thresholds, band_edges, out_dir, thr_desc):
    result = process_pass_pixels(recs, conf_var, thresholds, band_edges)
    if result is None:
        print(f"  pass {pas:03d}: no readable files, skipped", flush=True)
        return None
    count, n_obs, lat, lon = result
    out_path = write_pass_netcdf(
        pas, count, n_obs, lat, lon, len(recs), conf_var, thr_desc, out_dir
    )
    print(f"  pass {pas:03d}: cycles={len(recs)} max_count={int(count.max())} "
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
    print(f"cycles={len(cycle_dirs)} passes={len(passes)} files={n_files} workers={workers}")
    print(f"conf_var={args.conf_var} thresholds[{band_desc}]")
    print(f"out_dir={args.out_dir.resolve()}")

    common = (args.conf_var, thresholds, band_edges, args.out_dir, thr_desc)
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
