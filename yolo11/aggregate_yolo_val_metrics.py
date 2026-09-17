"""Aggregate the 60 per-run yolo_val_results/<run>/{test,val}_metrics.json
files (10 variants x 3 seeds x 2 splits) into one mean/std row per variant
per split.

Example
-------
    python aggregate_yolo_val_metrics.py
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

YOLO11_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = YOLO11_DIR / "yolo_val_results"

if str(YOLO11_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO11_DIR))

from src.inference.variants import parse_run_name  # noqa: E402

METRIC_COLUMNS = ["precision", "recall", "f1", "map50", "map50_95", "fitness"]
SPLITS = ["test", "val"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Defaults to yolo11/yolo_val_results -- override to aggregate a sister "
             "sweep, e.g. yolo11/yolo_val_results_customaug.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results_dir = args.results_dir
    summary_csv = results_dir / "summary_by_variant.csv"

    rows = []
    for run_dir in sorted(results_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        variant, seed = parse_run_name(run_name)

        for split in SPLITS:
            json_path = run_dir / f"{split}_metrics.json"
            if not json_path.is_file():
                continue
            metrics = json.loads(json_path.read_text())
            row = {"run_name": run_name, "variant": variant, "seed": seed, "split": split}
            row.update({k: metrics[k] for k in METRIC_COLUMNS})
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"loaded {len(df)} run/split results across {df['variant'].nunique()} variants")

    counts = df.groupby(["variant", "split"])["seed"].count()
    short = counts[counts != 3]
    if not short.empty:
        print("WARNING: these variant/split combos do not have exactly 3 seed results:")
        print(short)

    summary_rows = []
    for (variant, split), group in df.groupby(["variant", "split"]):
        row = {"variant": variant, "split": split, "n_seeds": len(group)}
        for col in METRIC_COLUMNS:
            values = group[col].astype(float)
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_std"] = values.std(ddof=1)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["split", "variant"]).reset_index(drop=True)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"\nwrote {summary_csv}")

    for split in SPLITS:
        print("\n" + "=" * 100)
        print(f"SUMMARY BY VARIANT -- split={split} (mean +/- std across 3 seeds)")
        print("=" * 100)
        sub = summary[summary["split"] == split].sort_values("f1_mean", ascending=False)
        display = pd.DataFrame({"variant": sub["variant"]})
        for col in METRIC_COLUMNS:
            display[col] = [
                f"{m:.3f} +/- {s:.3f}"
                for m, s in zip(sub[f"{col}_mean"], sub[f"{col}_std"])
            ]
        print(display.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
