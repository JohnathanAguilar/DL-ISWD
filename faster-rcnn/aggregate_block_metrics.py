"""Aggregate per-seed block-level metrics into mean/std by variant.

Verbatim from faster-rcnn/README.md Step 4. Run from faster-rcnn/.
Writes eval_results/summary_by_variant.csv.
"""

import re
import pandas as pd
from pathlib import Path

EVAL_RESULTS_DIR = Path("eval_results")
RUN_NAME_RE = re.compile(r"frcnn_(?P<variant>[a-z]+)_s(?P<seed>\d+)")

rows = []
for run_dir in sorted(EVAL_RESULTS_DIR.iterdir()):
    m = RUN_NAME_RE.match(run_dir.name)
    if not m:
        continue
    overall = pd.read_csv(run_dir / "block_metrics_overall.csv").iloc[0]
    rows.append({"variant": m["variant"], "seed": int(m["seed"]), **overall.to_dict()})

df = pd.DataFrame(rows)
metric_columns = [
    c for c in df.columns
    if c not in ("variant", "seed") and pd.api.types.is_numeric_dtype(df[c])
]

summary_rows = []
for variant, group in df.groupby("variant"):
    row = {"variant": variant, "n_seeds": len(group)}
    for col in metric_columns:
        values = group[col].astype(float)
        row[f"{col}_mean"] = values.mean()
        row[f"{col}_std"] = values.std(ddof=1)
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(EVAL_RESULTS_DIR / "summary_by_variant.csv", index=False)
print(f"wrote {EVAL_RESULTS_DIR / 'summary_by_variant.csv'} ({len(summary_rows)} variant rows)")
