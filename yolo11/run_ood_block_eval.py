"""Run the YOLO11 block-level evaluation (README Steps 5-6) on the labeled
out-of-distribution datasets in ``/workspace/OOD/`` instead of the in-distribution
BoB test split.

For each requested run and each OOD dataset this driver:

  1. stages the dataset's ``labeled_data/*.nc`` as symlinks under
     ``<stage-dir>/<key>/test/`` (so ``src.inference.infer`` can be pointed at it
     with ``--netcdf-root``/``--split test``);
  2. runs ``python -m src.inference.infer`` to produce the slim per-file
     detection NetCDFs under ``<results-dir>/<key>/<run>/test/netcdf/``;
  3. symlinks ``<results-dir>/<key>/<run>/valid`` -> the run's existing
     in-distribution ``yolo11/results/<run>/valid`` so the confidence operating
     point is still selected on the in-distribution validation set (keeping the
     OOD numbers comparable to the published in-distribution test-set numbers);
  4. calls the existing ``eval_run_all.py`` to score every discovered run with
     ``eval/evaluate_block_metrics.py`` (``--majority-threshold 0.25``);
  5. calls the existing ``aggregate_eval_metrics.py`` to roll the seeds up into
     ``<eval-results-dir>/<key>/summary_by_variant.csv``.

Finally it concatenates the three per-dataset summaries into
``<eval-results-dir>/summary_all_datasets.csv`` with a leading ``dataset`` column.

Nothing under ``src/`` or the other scripts is modified -- this is a thin
orchestrator that shells out to them with overridden paths.

Examples
--------
    # small verification run: 1 model, 1 dataset, 3 files
    python run_ood_block_eval.py --runs yolo11n_gauss_s123 --datasets amazon_mouth --limit 3

    # full nano run: gauss + frgb, 3 seeds, all 3 OOD datasets
    python run_ood_block_eval.py --device 0

    # also include the yolo11s_ runs
    python run_ood_block_eval.py --device 0 --include-small

    # re-score existing inference output without re-running inference
    python run_ood_block_eval.py --skip-inference
"""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

YOLO11_DIR = Path(__file__).resolve().parent
EVAL_RUN_ALL = YOLO11_DIR / "eval_run_all.py"
AGGREGATE = YOLO11_DIR / "aggregate_eval_metrics.py"
OOD_SRC_ROOT = Path("/workspace/OOD")

DATASETS = {
    "amazon_mouth": "SWOT_L3_Amazon_Mouth_Dataset",
    "mascarene_ridge": "SWOT_L3_Mascarene_Ridge_Dataset",
    "south_china_sea": "SWOT_L3_South_china_Sea_Dataset",
}

NANO_RUNS = [f"yolo11n_{v}_s{s}" for v in ("gauss", "frgb") for s in (123, 456, 789)]
SMALL_RUNS = [f"yolo11s_{v}_s{s}" for v in ("gauss", "frgb") for s in (123, 456, 789)]

MAJORITY_THRESHOLD = 0.25  # fixed project convention; eval_run_all.py hard-codes this too


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", nargs="+", default=None,
                   help="Explicit run-name list; overrides the gauss/frgb presets.")
    p.add_argument("--include-small", action="store_true",
                   help="With no --runs, also evaluate the yolo11s_ runs (12 instead of 6).")
    p.add_argument("--runs-dir", type=Path, default=YOLO11_DIR / "runs",
                   help="Where <run>/weights/best.pt lives. Use yolo11/runs_customaug for the "
                        "online-augmentation sweep.")
    p.add_argument("--indist-results-dir", type=Path, default=YOLO11_DIR / "results",
                   help="In-distribution inference outputs whose <run>/valid set sets the "
                        "confidence threshold. Must match --runs-dir "
                        "(e.g. yolo11/results_customaug for runs_customaug).")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                   help="Which OOD datasets to process (default: all three).")
    p.add_argument("--device", default="0",
                   help="Passed to infer.py --device (CUDA index, MIG UUID via env, or 'cpu').")
    p.add_argument("--conf", type=float, default=0.05,
                   help="Passed to infer.py --conf (raw-detection floor).")
    p.add_argument("--stage-dir", type=Path, default=YOLO11_DIR / "ood_staging",
                   help="Root of the symlink farm: <stage-dir>/<key>/test/*.nc.")
    p.add_argument("--results-dir", type=Path, default=YOLO11_DIR / "results_ood",
                   help="Inference output root: <results-dir>/<key>/<run>/test/{netcdf,text}/.")
    p.add_argument("--eval-results-dir", type=Path, default=YOLO11_DIR / "eval_results_ood",
                   help="Block-metrics output root: <eval-results-dir>/<key>/<run>/*.csv.")
    p.add_argument("--skip-inference", action="store_true",
                   help="Skip staging + inference; only wire the valid symlink, then eval + aggregate.")
    p.add_argument("--overwrite", action="store_true",
                   help="Forward --overwrite to infer.py and recreate stale symlinks.")
    p.add_argument("--limit", type=int, default=None,
                   help="Forward --limit N to infer.py (debug; limits inference only).")
    p.add_argument("--no-combined-summary", action="store_true",
                   help="Skip writing <eval-results-dir>/summary_all_datasets.csv.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command / symlink action; execute nothing.")
    p.add_argument("--python", default=sys.executable,
                   help="Interpreter used for the sub-scripts (default: this one).")
    p.add_argument("--majority-threshold", type=float, default=MAJORITY_THRESHOLD,
                   help="Guard only: eval_run_all.py fixes this at 0.25; a different value errors out.")
    return p.parse_args(argv)


def ensure_symlink(link: Path, target: Path, overwrite: bool, dry: bool) -> str:
    """Create/verify ``link`` -> ``target`` (absolute). Never clobbers a real file."""
    target = target.resolve()
    if link.is_symlink():
        if os.path.realpath(link) == str(target):
            return "ok"
        if overwrite:
            if not dry:
                link.unlink()
                os.symlink(target, link)
            return "recreated"
        return f"WARN: mismatched symlink kept (-> {os.path.realpath(link)})"
    if link.exists():
        return "WARN: non-symlink path kept, not touched"
    if not dry:
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)
    return "created"


def stage_dataset(key: str, stage_dir: Path, overwrite: bool, dry: bool) -> int:
    src = OOD_SRC_ROOT / DATASETS[key] / "labeled_data"
    files = sorted(src.glob("*.nc"))
    if not files:
        raise RuntimeError(f"no .nc files in {src}")

    test_dir = stage_dir / key / "test"
    if not dry:
        test_dir.mkdir(parents=True, exist_ok=True)

    wanted = set()
    for f in files:
        wanted.add(f.name)
        ensure_symlink(test_dir / f.name, f, overwrite, dry)

    if overwrite and test_dir.is_dir():
        for p in test_dir.iterdir():
            if p.is_symlink() and p.name not in wanted:
                if not dry:
                    p.unlink()

    return len(files)


def run_cmd(cmd, dry: bool) -> int:
    print("  $ " + " ".join(str(c) for c in cmd))
    if dry:
        return 0
    return subprocess.run(cmd, cwd=YOLO11_DIR).returncode


def preflight(runs, need_infer: bool, runs_dir: Path, indist_results_dir: Path):
    if need_infer and importlib.util.find_spec("ultralytics") is None:
        print("WARNING: `ultralytics` is not importable in this environment; inference will "
              "fail until you run:  pip install ultralytics\n")

    usable = []
    for run in runs:
        weights = runs_dir / run / "weights" / "best.pt"
        valid = indist_results_dir / run / "valid" / "netcdf"
        if not weights.is_file():
            print(f"  drop {run}: no weights at {weights}")
            continue
        if not (valid.is_dir() and any(valid.glob("*.nc"))):
            print(f"  drop {run}: no in-distribution valid/netcdf at {valid} "
                  "(needed to select the confidence threshold)")
            continue
        usable.append(run)
    return usable


def write_combined_summary(eval_results_dir: Path, keys, dry: bool):
    import pandas as pd

    frames = []
    for key in keys:
        p = eval_results_dir / key / "summary_by_variant.csv"
        if p.is_file():
            df = pd.read_csv(p)
            df.insert(0, "dataset", key)
            frames.append(df)
    if not frames:
        print("no per-dataset summary_by_variant.csv found; skipping combined summary")
        return

    out = eval_results_dir / "summary_all_datasets.csv"
    print(f"writing {out}")
    if not dry:
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    if abs(args.majority_threshold - MAJORITY_THRESHOLD) > 1e-9:
        print(f"error: --majority-threshold {args.majority_threshold} != {MAJORITY_THRESHOLD}. "
              "eval_run_all.py fixes it at 0.25 and exposes no flag; a different value would "
              "require calling eval/evaluate_block_metrics.py directly, which this reuse-only "
              "driver does not do.")
        return 2

    runs = args.runs or (NANO_RUNS + SMALL_RUNS if args.include_small else NANO_RUNS)
    need_infer = not args.skip_inference

    print(f"runs requested ({len(runs)}): {', '.join(runs)}")
    print(f"datasets: {', '.join(args.datasets)}")
    print(f"runs-dir={args.runs_dir}  indist-results-dir={args.indist_results_dir}")
    print(f"stage-dir={args.stage_dir}  results-dir={args.results_dir}  "
          f"eval-results-dir={args.eval_results_dir}")
    print(f"device={args.device}  conf={args.conf}  skip_inference={args.skip_inference}  "
          f"overwrite={args.overwrite}  limit={args.limit}  dry_run={args.dry_run}\n")

    runs = preflight(runs, need_infer, args.runs_dir, args.indist_results_dir)
    if not runs:
        print("no usable runs after preflight; nothing to do.")
        return 1
    print(f"usable runs ({len(runs)}): {', '.join(runs)}\n")

    run_records = []   # (dataset, run, stage_infer_status, valid_link_status)
    ds_records = []     # (dataset, eval_status, aggregate_status)

    for key in args.datasets:
        print("=" * 88)
        print(f"DATASET: {key}  ({DATASETS[key]})")
        print("=" * 88)
        try:
            if need_infer:
                n = stage_dataset(key, args.stage_dir, args.overwrite, args.dry_run)
                print(f"staged {n} symlink(s) -> {args.stage_dir / key / 'test'}")

            for run in runs:
                run_out = args.results_dir / key / run
                if not args.dry_run:
                    run_out.mkdir(parents=True, exist_ok=True)

                infer_status = "skipped"
                if need_infer:
                    cmd = [
                        args.python, "-m", "src.inference.infer",
                        "--run-name", run, "--split", "test",
                        "--device", args.device, "--conf", str(args.conf),
                        "--runs-dir", str(args.runs_dir.resolve()),
                        "--netcdf-root", str((args.stage_dir / key).resolve()),
                        "--results-dir", str((args.results_dir / key).resolve()),
                    ]
                    if args.overwrite:
                        cmd.append("--overwrite")
                    if args.limit:
                        cmd += ["--limit", str(args.limit)]
                    print(f"\n[{key}/{run}] inference")
                    rc = run_cmd(cmd, args.dry_run)
                    infer_status = "ok" if rc == 0 else f"FAIL(exit={rc})"
                    if rc != 0:
                        run_records.append((key, run, infer_status, "-"))
                        continue

                link_status = ensure_symlink(
                    run_out / "valid", args.indist_results_dir / run / "valid",
                    args.overwrite, args.dry_run,
                )
                print(f"[{key}/{run}] valid symlink: {link_status}")
                run_records.append((key, run, infer_status, link_status))

            # --- block eval for the whole dataset (existing eval_run_all.py) ---
            print(f"\n[{key}] block eval (eval_run_all.py)")
            rc = run_cmd(
                [args.python, str(EVAL_RUN_ALL),
                 "--results-dir", str((args.results_dir / key).resolve()),
                 "--eval-results-dir", str((args.eval_results_dir / key).resolve())],
                args.dry_run,
            )
            eval_status = "ok" if rc == 0 else f"FAIL(exit={rc})"

            aggregate_status = "skipped"
            if rc == 0:
                print(f"\n[{key}] aggregate (aggregate_eval_metrics.py)")
                rc2 = run_cmd(
                    [args.python, str(AGGREGATE),
                     "--eval-results-dir", str((args.eval_results_dir / key).resolve())],
                    args.dry_run,
                )
                aggregate_status = "ok" if rc2 == 0 else f"FAIL(exit={rc2})"

            ds_records.append((key, eval_status, aggregate_status))

        except Exception as exc:  # noqa: BLE001 - one dataset must not abort the rest
            import traceback
            print(f"[{key}] ERROR: {exc}")
            traceback.print_exc()
            ds_records.append((key, f"ERROR({exc})", "-"))

    if not args.no_combined_summary:
        print("\n" + "=" * 88)
        write_combined_summary(args.eval_results_dir, args.datasets, args.dry_run)

    # --- report ---
    print("\n" + "=" * 88)
    print("PER-RUN STATUS")
    print("=" * 88)
    print(f"{'dataset':<16}{'run':<22}{'infer':<16}{'valid symlink'}")
    for key, run, infer_status, link_status in run_records:
        print(f"{key:<16}{run:<22}{infer_status:<16}{link_status}")

    print("\n" + "=" * 88)
    print("PER-DATASET STATUS")
    print("=" * 88)
    print(f"{'dataset':<16}{'block eval':<20}{'aggregate'}")
    for key, eval_status, aggregate_status in ds_records:
        print(f"{key:<16}{eval_status:<20}{aggregate_status}")

    failures = [r for r in run_records if "FAIL" in r[2] or "WARN" in str(r[3])]
    failures += [d for d in ds_records if "FAIL" in d[1] or "ERROR" in d[1] or "FAIL" in d[2]]
    if failures:
        print("\nFAILURES / WARNINGS:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nall requested work completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
