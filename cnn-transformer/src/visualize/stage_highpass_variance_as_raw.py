"""Stage highpass_variance_pass*.nc files to look like raw per-pass SWOT NetCDFs.

``build_isw_store.py`` regrids raw per-pass SWOT NetCDFs
(``SWOT_L3_LR_SSH_Expert_{cycle}_{pass}_{date}T{time}_*.nc`` under
``results/cycle_*/``) and only recognizes 3 hardcoded variable names
(``internal_wave_bbox_confidence``, ``gaussian_filter_highpass``,
``quality_flag``). It is deliberately left unmodified.

The highpass-variance NetCDFs produced by ``make_highpass_variance_netcdf.py``
(``results/all_cycles/highpass_variance_netcdf/highpass_variance_pass{NNN}.nc``,
one per SWOT pass number, already aggregated across that pass's repeat
cycles) don't match either the filename pattern or the variable names, so
``build_isw_store.py`` can't read them directly.

This script stages a renamed copy of each highpass-variance file so it looks
like a raw per-pass file to ``build_isw_store.py``, filling all 3 VAR_MAP
slots:

    highpass_variance_mean   -> gaussian_filter_highpass   (REDUCE = mean)
    highpass_variance_std    -> quality_flag                (REDUCE = max)
    n_cycles_observed        -> internal_wave_bbox_confidence (REDUCE = max)

Filenames are synthesized as
``SWOT_L3_LR_SSH_Expert_001_{pass:03d}_{date}T{time}_hpvar.nc`` with a
constant cycle (001, there is no real cycle axis here -- one file already
aggregates over cycles) and a synthetic date/time chosen so that
``build_isw_store.py``'s own timestamp parsing (``discover()``) reproduces
the pass number as the resulting Unix epoch seconds -- i.e. slice ``i``'s
"time" coordinate ends up equal to that file's pass number. All staged files
land in one ``cycle_01/`` folder, matching ``build_isw_store.py``'s default
``--cycle-glob cycle_0[12]``.

Example
-------
    cd cnn-transformer/src/visualize
    python stage_highpass_variance_as_raw.py --workers 8

    python build_isw_store.py \\
      --results-root _staging_hpvar_raw --cycle-glob cycle_01 \\
      --vars internal_wave_bbox_confidence,gaussian_filter_highpass,quality_flag \\
      --out stores/isw-highpass-variance --resolution 0.02 --levels 1
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import xarray as xr

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent.parent  # cnn-transformer/

_FNAME_PREFIX = "highpass_variance_pass"
_FNAME_GLOB = f"{_FNAME_PREFIX}*.nc"

RENAME = {
    "highpass_variance_mean": "gaussian_filter_highpass",
    "highpass_variance_std": "quality_flag",
    "n_cycles_observed": "internal_wave_bbox_confidence",
}
CYCLE = 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hpvar-dir", type=Path,
                   default=_MODEL_DIR / "results" / "all_cycles" / "highpass_variance_netcdf")
    p.add_argument("--staging-dir", type=Path, default=_HERE / "_staging_hpvar_raw")
    p.add_argument("--limit", type=int, default=0, help="If >0, only the first N passes.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel workers (0 = auto = min(8, cpu-2)).")
    return p.parse_args(argv)


def stage_one(src_path: str, out_dir: str):
    src_path = Path(src_path)
    name = src_path.name
    if not name.startswith(_FNAME_PREFIX) or not name.endswith(".nc"):
        return None
    pas = int(name[len(_FNAME_PREFIX):-len(".nc")])

    start = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=pas)
    date, time = start.strftime("%Y%m%d"), start.strftime("%H%M%S")
    out_name = f"SWOT_L3_LR_SSH_Expert_{CYCLE:03d}_{pas:03d}_{date}T{time}_hpvar.nc"
    out_path = Path(out_dir) / out_name

    with xr.open_dataset(src_path) as ds:
        ds = ds.rename(RENAME)
        ds.to_netcdf(out_path)
    return out_path


def _worker(args):
    src_path, out_dir = args
    try:
        return stage_one(src_path, out_dir)
    except Exception as exc:  # noqa: BLE001 - report, keep going
        print(f"    {Path(src_path).name}: FAILED {exc}", flush=True)
        return None


def main(argv=None):
    args = parse_args(argv)

    files = sorted(args.hpvar_dir.glob(_FNAME_GLOB))
    if not files:
        raise SystemExit(f"no '{_FNAME_GLOB}' files under {args.hpvar_dir.resolve()}")
    if args.limit:
        files = files[: args.limit]

    out_dir = args.staging_dir / f"cycle_{CYCLE:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 2))
    print(f"hpvar_dir = {args.hpvar_dir.resolve()}")
    print(f"passes = {len(files)}  workers = {workers}")
    print(f"staging_dir = {out_dir.resolve()}")

    jobs = [(str(f), str(out_dir)) for f in files]
    written = []
    if workers == 1:
        for j in jobs:
            out = _worker(j)
            if out is not None:
                written.append(out)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker, j) for j in jobs]
            for k, f in enumerate(as_completed(futs), 1):
                out = f.result()
                if out is not None:
                    written.append(out)
                if k % 50 == 0 or k == len(jobs):
                    print(f"  staged {k}/{len(jobs)}", flush=True)

    print(f"\nstaged {len(written)}/{len(files)} passes to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
