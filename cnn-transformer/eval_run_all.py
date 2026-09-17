"""Run eval/evaluate_block_metrics.py once per trained cnn-transformer model
(30 models: 10 variants x 3 seeds), scoring each model's test-set inference
output against its own valid-set output.

Confidence threshold is not a flag here -- evaluate_block_metrics.py selects
it internally from the pooled validation set's max-F1 point (searched within
[--min-conf-threshold, --max-conf-threshold] = [0.05, 0.95]) and applies it
to the test set. --majority-threshold IS required (binarizes the ground
truth block_mask into per-block positive/negative labels) and is fixed at
0.25, matching the fixed convention already used for yolo11.

Example
-------
    python eval_run_all.py
"""

import subprocess
import sys
from pathlib import Path

CNNTRANS_DIR = Path(__file__).resolve().parent
REPO_ROOT = CNNTRANS_DIR.parent
RESULTS_DIR = CNNTRANS_DIR / "results"
EVAL_SCRIPT = REPO_ROOT / "eval" / "evaluate_block_metrics.py"
EVAL_RESULTS_DIR = CNNTRANS_DIR / "eval_results"

MAJORITY_THRESHOLD = "0.25"


def main():
    run_names = sorted(
        p.name
        for p in RESULTS_DIR.iterdir()
        if (p / "test" / "netcdf").is_dir() and (p / "valid" / "netcdf").is_dir()
    )
    print(f"found {len(run_names)} runs with results")

    n_ok, n_fail = 0, 0
    for i, run_name in enumerate(run_names, start=1):
        out_dir = EVAL_RESULTS_DIR / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--test-dir", str(RESULTS_DIR / run_name / "test" / "netcdf"),
            "--val-dir", str(RESULTS_DIR / run_name / "valid" / "netcdf"),
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
