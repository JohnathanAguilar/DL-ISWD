"""Aggregate the 30 per-run eval_results/<run>/block_metrics_overall.csv
files (10 variants x 3 seeds) into one mean/std row per variant.

Example
-------
    python aggregate_eval_metrics.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

CNNTRANS_DIR = Path(__file__).resolve().parent
EVAL_RESULTS_DIR = CNNTRANS_DIR / "eval_results"
SUMMARY_CSV = EVAL_RESULTS_DIR / "summary_by_variant.csv"

RUN_NAME_RE = re.compile(r"^cnntrans_(?P<variant>[a-z0-9]+)_(?P<seed>\d+)$")

METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "specificity", "f1", "f0_5", "iou",
    "balanced_accuracy", "label_positive_rate", "pred_positive_rate",
    "false_positive_rate", "false_discovery_rate", "tp", "fp", "tn", "fn",
    "total_blocks", "ap", "pr_auc", "roc_auc", "threshold", "valid_blocks",
    "ground_truth_positive_blocks", "predicted_positive_blocks",
]


def parse_run_name(run_name):
    m = RUN_NAME_RE.match(run_name)
    if not m:
        raise ValueError(f"Run name {run_name!r} doesn't match cnntrans_<variant>_<seed>")
    return m["variant"], int(m["seed"])


def main():
    rows = []
    for run_dir in sorted(EVAL_RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "block_metrics_overall.csv"
        if not csv_path.is_file():
            continue

        variant, seed = parse_run_name(run_dir.name)
        overall = pd.read_csv(csv_path).iloc[0]
        row = {"run_name": run_dir.name, "variant": variant, "seed": seed}
        row.update({k: overall[k] for k in METRIC_COLUMNS if k in overall})
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} run results across {df['variant'].nunique()} variants")

    counts = df.groupby("variant")["seed"].count()
    short = counts[counts != 3]
    if not short.empty:
        print("WARNING: these variants do not have exactly 3 seed results:")
        print(short)

    metric_cols_present = [c for c in METRIC_COLUMNS if c in df.columns]

    summary_rows = []
    for variant, group in df.groupby("variant"):
        row = {"variant": variant, "n_seeds": len(group)}
        for col in metric_cols_present:
            values = group[col].astype(float)
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("variant").reset_index(drop=True)

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nwrote {SUMMARY_CSV}")

    print("\n" + "=" * 100)
    print("SUMMARY BY VARIANT (mean +/- std across 3 seeds)")
    print("=" * 100)
    sub = summary.sort_values("f1_mean", ascending=False)
    display_cols = ["precision", "recall", "f1", "iou", "ap", "roc_auc"]
    display = pd.DataFrame({"variant": sub["variant"]})
    for col in display_cols:
        if f"{col}_mean" not in sub.columns:
            continue
        display[col] = [
            f"{m:.3f} +/- {s:.3f}"
            for m, s in zip(sub[f"{col}_mean"], sub[f"{col}_std"])
        ]
    print(display.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
