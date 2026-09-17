"""Run eval/evaluate_block_metrics.py once per trained YOLO11 model (30
models: 10 variants x 3 seeds), scoring each model's test-set inference
output against its own validation-set output.

Confidence threshold is not a flag here -- evaluate_block_metrics.py selects
it internally from the pooled validation set's max-F1 point and applies it
to the test set. --majority-threshold IS required (binarizes the ground
truth block_mask into per-block positive/negative labels) and is fixed at
0.25 per explicit user direction.

Example
-------
    python eval_run_all.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

YOLO11_DIR = Path(__file__).resolve().parent
REPO_ROOT = YOLO11_DIR.parent
DEFAULT_RESULTS_DIR = YOLO11_DIR / "results"
EVAL_SCRIPT = REPO_ROOT / "eval" / "evaluate_block_metrics.py"
DEFAULT_EVAL_RESULTS_DIR = YOLO11_DIR / "eval_results"

MAJORITY_THRESHOLD = "0.25"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Defaults to yolo11/results -- override to eval a sister sweep, "
             "e.g. yolo11/results_customaug.",
    )
    parser.add_argument(
        "--eval-results-dir", type=Path, default=DEFAULT_EVAL_RESULTS_DIR,
        help="Defaults to yolo11/eval_results -- override to match --results-dir, "
             "e.g. yolo11/eval_results_customaug.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_dir = args.results_dir
    eval_results_dir = args.eval_results_dir

    run_names = sorted(
        p.name
        for p in results_dir.iterdir()
        if (p / "test" / "netcdf").is_dir() and (p / "valid" / "netcdf").is_dir()
    )
    print(f"found {len(run_names)} runs with results")

    n_ok, n_fail = 0, 0
    for i, run_name in enumerate(run_names, start=1):
        out_dir = eval_results_dir / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--test-dir", str(results_dir / run_name / "test" / "netcdf"),
            "--val-dir", str(results_dir / run_name / "valid" / "netcdf"),
            "--majority-threshold", MAJORITY_THRESHOLD,
            "--overall-csv", str(out_dir / "block_metrics_overall.csv"),
            "--per-file-csv", str(out_dir / "block_metrics_per_file.csv"),
        ]

        log_path = out_dir / "eval.log"
        print(f"[{i}/{len(run_names)}] {run_name} ...", end=" ", flush=True)
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

        if result.returncode == 0:
            n_ok += 1
            print("ok")
        else:
            n_fail += 1
            print(f"FAILED (exit {result.returncode}), see {log_path}")

    print(f"done. ok={n_ok} failed={n_fail} total={len(run_names)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
