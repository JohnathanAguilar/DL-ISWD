"""Aggregate the 30 per-run block_metrics_overall.csv files (10 variants x 3
seeds) into one mean/std row per variant.

Example
-------
    python aggregate_eval_metrics.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

YOLO11_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_RESULTS_DIR = YOLO11_DIR / "eval_results"

if str(YOLO11_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO11_DIR))

from src.inference.variants import parse_run_name  # noqa: E402

# Every numeric column in block_metrics_overall.csv worth reporting mean/std
# for. Excludes only the two path-string columns (val_set_directory,
# test_set_directory) and block_height/block_width/majority_threshold, which
# are constants (20, 20, 0.25) identical across every run, not metrics.
METRIC_COLUMNS = [
    "val_files_found", "val_valid_blocks", "val_ground_truth_positive_blocks",
    "files_found", "successful_files", "failed_files",
    "valid_blocks", "ground_truth_positive_blocks", "predicted_positive_blocks",
    "threshold", "accuracy", "precision", "recall", "specificity", "f1", "f0_5",
    "iou", "balanced_accuracy", "label_positive_rate", "pred_positive_rate",
    "false_positive_rate", "false_discovery_rate", "tp", "fp", "tn", "fn",
    "total_blocks", "ap", "pr_auc", "roc_auc",
]

DISPLAY_COLUMNS = [
    "accuracy", "precision", "recall", "f1", "ap", "pr_auc", "roc_auc", "iou",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-results-dir", type=Path, default=DEFAULT_EVAL_RESULTS_DIR,
        help="Defaults to yolo11/eval_results -- override to aggregate a sister sweep, "
             "e.g. yolo11/eval_results_customaug.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    eval_results_dir = args.eval_results_dir
    summary_csv = eval_results_dir / "summary_by_variant.csv"

    rows = []
    for run_dir in sorted(eval_results_dir.iterdir()):
        csv_path = run_dir / "block_metrics_overall.csv"
        if not csv_path.is_file():
            continue

        run_name = run_dir.name
        variant, seed = parse_run_name(run_name)
        # parse_run_name drops the model-size prefix (yolo11n/yolo11s), but
        # yolo11n_gauss_* and yolo11s_gauss_* both parse to variant="gauss" --
        # without this, grouping by variant alone would pool nano and small
        # seed-results into one blended row. Derive it separately here so
        # they group independently.
        model_size = run_name.split("_")[0]

        row = pd.read_csv(csv_path).iloc[0].to_dict()
        row["run_name"] = run_name
        row["model_size"] = model_size
        row["variant"] = variant
        row["seed"] = seed
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} run results across {df['variant'].nunique()} variants")

    counts = df.groupby(["model_size", "variant"])["seed"].count()
    short = counts[counts != 3]
    if not short.empty:
        print("WARNING: these model_size/variant combos do not have exactly 3 seed results:")
        print(short)

    summary_rows = []
    for (model_size, variant), group in df.groupby(["model_size", "variant"]):
        row = {"model_size": model_size, "variant": variant, "n_seeds": len(group)}
        for col in METRIC_COLUMNS:
            values = group[col].astype(float)
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["model_size", "variant"]).reset_index(drop=True)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"\nwrote {summary_csv}")

    print("\n" + "=" * 100)
    print("SUMMARY BY VARIANT (mean +/- std across 3 seeds)")
    print("=" * 100)

    display = pd.DataFrame({"model_size": summary["model_size"], "variant": summary["variant"]})
    for col in DISPLAY_COLUMNS:
        display[col] = [
            f"{m:.3f} +/- {s:.3f}"
            for m, s in zip(summary[f"{col}_mean"], summary[f"{col}_std"])
        ]

    print(display.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
