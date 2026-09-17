"""Run the cnn-transformer block-level evaluation (README Steps 3-5) on the
labeled out-of-distribution datasets in ``/workspace/OOD/`` instead of the
in-distribution BoB test split.

For each requested run (``cnntrans_<variant>_<seed>``) and each OOD dataset:

  1. stage the dataset's ``labeled_data/*.nc`` as symlinks under
     ``<stage-dir>/<key>/test/`` (so ``src.inference.infer`` can be pointed at it
     with ``--netcdf-root`` / ``--split test``);
  2. run ``python src/inference/infer.py`` to write the slim per-file prediction
     NetCDFs (``internal_wave_bbox_confidence`` + carried-over ``block_mask`` /
     ``gaussian_filter_highpass``) under ``<results-dir>/<key>/<run>/test/netcdf/``;
  3. symlink ``<results-dir>/<key>/<run>/valid`` -> the run's existing
     in-distribution ``cnn-transformer/results/<run>/valid`` so the confidence
     operating point is still selected on the in-distribution validation set
     (keeping the OOD numbers comparable to the published in-distribution ones);
  4. score each run with ``eval/evaluate_block_metrics.py`` -- the exact command
     ``cnn-transformer/eval_run_all.py`` builds, with OOD paths substituted --
     writing ``block_metrics_{overall,per_file}.csv`` + ``eval.log``;
  5. aggregate the 3 seeds per variant into ``summary_by_variant.csv`` (inline,
     mirroring ``cnn-transformer/aggregate_eval_metrics.py``).

Finally it concatenates the per-dataset summaries into
``<eval-results-dir>/summary_all_datasets.csv`` with a leading ``dataset`` column.

Nothing under ``src/`` or the sibling scripts is modified. The two cnn-transformer
helpers (``eval_run_all.py`` / ``aggregate_eval_metrics.py``) take no path flags,
so this driver calls ``eval/evaluate_block_metrics.py`` directly (still the only
thing doing metric work) and reproduces the ~15-line seed aggregation inline.

Examples
--------
    # verification: 1 run, 1 dataset, 3 files
    python run_ood_block_eval.py --runs cnntrans_gfh_123 --datasets amazon_mouth --limit 3

    # full run: gfh + frgb-analogue shg, 3 seeds, all 3 OOD datasets
    python run_ood_block_eval.py --device 0

    # re-score existing inference output without re-running inference
    python run_ood_block_eval.py --skip-inference
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

CNNTRANS_DIR = Path(__file__).resolve().parent
REPO_ROOT = CNNTRANS_DIR.parent
EVAL_SCRIPT = REPO_ROOT / "eval" / "evaluate_block_metrics.py"
OOD_SRC_ROOT = Path("/workspace/OOD")

DATASETS = {
    "amazon_mouth": "SWOT_L3_Amazon_Mouth_Dataset",
    "mascarene_ridge": "SWOT_L3_Mascarene_Ridge_Dataset",
    "south_china_sea": "SWOT_L3_South_china_Sea_Dataset",
}

# gfh = single-channel gaussian_filter_highpass  (analogue of YOLO's `gauss`)
# shg = 3-channel ssha_quality_masked+rolling_mean_highpass+gradient_magnitude
#       (composite analogue of YOLO's `frgb`)
DEFAULT_RUNS = [f"cnntrans_{v}_{s}" for v in ("gfh", "shg") for s in (123, 456, 789)]

RUN_NAME_RE = re.compile(r"^cnntrans_(?P<variant>[a-z0-9]+)_(?P<seed>\d+)$")

# Exactly the columns cnn-transformer/aggregate_eval_metrics.py rolls up.
METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "specificity", "f1", "f0_5", "iou",
    "balanced_accuracy", "label_positive_rate", "pred_positive_rate",
    "false_positive_rate", "false_discovery_rate", "tp", "fp", "tn", "fn",
    "total_blocks", "ap", "pr_auc", "roc_auc", "threshold", "valid_blocks",
    "ground_truth_positive_blocks", "predicted_positive_blocks",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--runs", nargs="+", default=None,
                   help="Explicit run-name list (default: cnntrans_{gfh,shg}_{123,456,789}).")
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS),
                   help="Which OOD datasets to process (default: all three).")
    p.add_argument("--device", default="0",
                   help="Passed to infer.py --device (CUDA index, 'cuda', 'cpu', ...).")
    p.add_argument("--runs-dir", type=Path, default=CNNTRANS_DIR / "runs",
                   help="Where <run>/best.pt lives; forwarded to infer.py --runs-dir.")
    p.add_argument("--indist-results-dir", type=Path, default=CNNTRANS_DIR / "results",
                   help="In-distribution inference outputs whose <run>/valid set selects the "
                        "confidence threshold.")
    p.add_argument("--stage-dir", type=Path, default=CNNTRANS_DIR / "ood_staging",
                   help="Root of the symlink farm: <stage-dir>/<key>/test/*.nc.")
    p.add_argument("--results-dir", type=Path, default=CNNTRANS_DIR / "results_ood",
                   help="Inference output root: <results-dir>/<key>/<run>/test/netcdf/.")
    p.add_argument("--eval-results-dir", type=Path, default=CNNTRANS_DIR / "eval_results_ood",
                   help="Block-metrics output root: <eval-results-dir>/<key>/<run>/*.csv.")
    p.add_argument("--majority-threshold", default="0.25",
                   help="Ground-truth block_mask binarization fraction (project convention: 0.25).")
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
                   help="Interpreter used for the sub-processes (default: this one).")
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


def preflight(runs, need_infer: bool, runs_dir: Path, indist_results_dir: Path):
    if need_infer and importlib.util.find_spec("torch") is None:
        print("WARNING: `torch` is not importable; inference will fail.\n")

    usable = []
    for run in runs:
        if not RUN_NAME_RE.match(run):
            print(f"  drop {run}: not a cnntrans_<variant>_<seed> name")
            continue
        weights = runs_dir / run / "best.pt"
        valid = indist_results_dir / run / "valid" / "netcdf"
        if not weights.is_file():
            print(f"  drop {run}: no checkpoint at {weights}")
            continue
        if not (valid.is_dir() and any(valid.glob("*.nc"))):
            print(f"  drop {run}: no in-distribution valid/netcdf at {valid} "
                  "(needed to select the confidence threshold)")
            continue
        usable.append(run)
    return usable


def run_block_eval(python: str, run: str, key: str, args, dry: bool) -> int:
    """One call to eval/evaluate_block_metrics.py -- the exact command
    cnn-transformer/eval_run_all.py builds, with OOD paths."""
    out_dir = args.eval_results_dir / key / run
    if not dry:
        out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python, str(EVAL_SCRIPT),
        "--test-dir", str((args.results_dir / key / run / "test" / "netcdf").resolve()),
        "--val-dir", str((args.results_dir / key / run / "valid" / "netcdf").resolve()),
        "--majority-threshold", str(args.majority_threshold),
        "--overall-csv", str(out_dir / "block_metrics_overall.csv"),
        "--per-file-csv", str(out_dir / "block_metrics_per_file.csv"),
    ]
    print("  $ " + " ".join(cmd))
    if dry:
        return 0
    with open(out_dir / "eval.log", "w") as log_f:
        return subprocess.run(cmd, cwd=CNNTRANS_DIR, stdout=log_f,
                              stderr=subprocess.STDOUT).returncode


def aggregate_dataset(eval_key_dir: Path, dry: bool) -> str:
    """Inline reproduction of cnn-transformer/aggregate_eval_metrics.py, scoped to
    one OOD dataset's eval_results tree."""
    import pandas as pd

    if not eval_key_dir.is_dir():
        return "skipped (no eval_results dir)"

    rows = []
    for run_dir in sorted(p for p in eval_key_dir.iterdir() if p.is_dir()):
        csv_path = run_dir / "block_metrics_overall.csv"
        if not csv_path.is_file():
            continue
        m = RUN_NAME_RE.match(run_dir.name)
        if not m:
            continue
        overall = pd.read_csv(csv_path).iloc[0]
        row = {"run_name": run_dir.name, "variant": m["variant"], "seed": int(m["seed"])}
        row.update({k: overall[k] for k in METRIC_COLUMNS if k in overall})
        rows.append(row)

    if not rows:
        return "skipped (no per-run block_metrics_overall.csv)"

    df = pd.DataFrame(rows)
    counts = df.groupby("variant")["seed"].count()
    short = counts[counts != 3]
    if not short.empty:
        print("  note: variants without exactly 3 seed results:")
        print("  " + short.to_string().replace("\n", "\n  "))

    metric_cols = [c for c in METRIC_COLUMNS if c in df.columns]
    summary_rows = []
    for variant, group in df.groupby("variant"):
        r = {"variant": variant, "n_seeds": len(group)}
        for col in metric_cols:
            v = group[col].astype(float)
            r[f"{col}_mean"] = v.mean()
            r[f"{col}_std"] = v.std(ddof=1)
        summary_rows.append(r)

    summary = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)
    out = eval_key_dir / "summary_by_variant.csv"
    print(f"  wrote {out}")
    if not dry:
        summary.to_csv(out, index=False)
    return "ok"


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

    runs = args.runs or DEFAULT_RUNS
    need_infer = not args.skip_inference

    print(f"runs requested ({len(runs)}): {', '.join(runs)}")
    print(f"datasets: {', '.join(args.datasets)}")
    print(f"runs-dir={args.runs_dir}  indist-results-dir={args.indist_results_dir}")
    print(f"stage-dir={args.stage_dir}  results-dir={args.results_dir}  "
          f"eval-results-dir={args.eval_results_dir}")
    print(f"device={args.device}  majority_threshold={args.majority_threshold}  "
          f"skip_inference={args.skip_inference}  overwrite={args.overwrite}  "
          f"limit={args.limit}  dry_run={args.dry_run}\n")

    runs = preflight(runs, need_infer, args.runs_dir, args.indist_results_dir)
    if not runs:
        print("no usable runs after preflight; nothing to do.")
        return 1
    print(f"usable runs ({len(runs)}): {', '.join(runs)}\n")

    run_records = []   # (dataset, run, infer_status, valid_link_status, eval_status)
    ds_records = []    # (dataset, aggregate_status)

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
                        args.python, str(CNNTRANS_DIR / "src" / "inference" / "infer.py"),
                        "--run-name", run, "--split", "test",
                        "--device", args.device,
                        "--runs-dir", str(args.runs_dir.resolve()),
                        "--netcdf-root", str((args.stage_dir / key).resolve()),
                        "--results-dir", str((args.results_dir / key).resolve()),
                    ]
                    if args.overwrite:
                        cmd.append("--overwrite")
                    if args.limit:
                        cmd += ["--limit", str(args.limit)]
                    print(f"\n[{key}/{run}] inference")
                    print("  $ " + " ".join(cmd))
                    rc = 0 if args.dry_run else subprocess.run(cmd, cwd=CNNTRANS_DIR).returncode
                    infer_status = "ok" if rc == 0 else f"FAIL(exit={rc})"
                    if rc != 0:
                        run_records.append((key, run, infer_status, "-", "-"))
                        continue

                link_status = ensure_symlink(
                    run_out / "valid", args.indist_results_dir / run / "valid",
                    args.overwrite, args.dry_run,
                )
                print(f"[{key}/{run}] valid symlink: {link_status}")

                print(f"[{key}/{run}] block eval")
                rc = run_block_eval(args.python, run, key, args, args.dry_run)
                eval_status = "ok" if rc == 0 else f"FAIL(exit={rc})"
                run_records.append((key, run, infer_status, link_status, eval_status))

            print(f"\n[{key}] aggregate")
            agg = aggregate_dataset(args.eval_results_dir / key, args.dry_run)
            ds_records.append((key, agg))

        except Exception as exc:  # noqa: BLE001 - one dataset must not abort the rest
            import traceback
            print(f"[{key}] ERROR: {exc}")
            traceback.print_exc()
            ds_records.append((key, f"ERROR({exc})"))

    if not args.no_combined_summary:
        print("\n" + "=" * 88)
        write_combined_summary(args.eval_results_dir, args.datasets, args.dry_run)

    # --- report ---
    print("\n" + "=" * 88)
    print("PER-RUN STATUS")
    print("=" * 88)
    print(f"{'dataset':<16}{'run':<20}{'infer':<16}{'valid link':<28}{'block eval'}")
    for key, run, infer_status, link_status, eval_status in run_records:
        print(f"{key:<16}{run:<20}{infer_status:<16}{link_status:<28}{eval_status}")

    print("\n" + "=" * 88)
    print("PER-DATASET STATUS")
    print("=" * 88)
    print(f"{'dataset':<16}{'aggregate'}")
    for key, agg in ds_records:
        print(f"{key:<16}{agg}")

    failures = [r for r in run_records
                if "FAIL" in r[2] or "WARN" in str(r[3]) or "FAIL" in str(r[4])]
    failures += [d for d in ds_records if "ERROR" in d[1] or "FAIL" in d[1]]
    if failures:
        print("\nFAILURES / WARNINGS:")
        for f in failures:
            print(f"  {f}")
        return 1

    print("\nall requested work completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
