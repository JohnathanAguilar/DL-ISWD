#!/usr/bin/env python3
"""Cross-model block-level comparison of the two representations every model
family shares: a Gaussian high-pass single-channel input and a multichannel /
scientific-RGB-composite input. Compares YOLO11n, YOLO11s, Faster R-CNN,
RT-DETR, and the CNN-Transformer.

All five families are scored by the same shared eval/evaluate_block_metrics.py
pipeline, so precision/recall/f1/ap are directly comparable across them. Reads
the raw per-seed block_metrics_overall.csv for each (model, variant, seed) --
NEVER each family's summary_by_variant.csv, some of which (Faster R-CNN,
RT-DETR) were found to be internally inconsistent with their own per-seed
files during exploration (looks like a column-shift bug in those aggregator
scripts). Mean + sample std (ddof=1) across the 3 seeds are computed here,
matching the convention in yolo11/eval_figures and cnn-transformer/eval_figures.

Variant-code mapping (same underlying channels, different short codes per
model):
  model            Gaussian HP   multichannel   results dir
  YOLO11n           gauss         frgb           yolo11/eval_results_customaug
  YOLO11s           gauss         frgb           yolo11/eval_results_customaug
  Faster R-CNN      gh            frgb           faster-rcnn/eval_results
  RT-DETR           ghp           frgb           rt-detr/eval_results_lr5e5
  CNN-Transformer   gfh           shg            cnn-transformer/eval_results

Writes two PNGs to model_comparison_figures/:
  gaussian_hp_across_models.png   -- x = 5 models, bars = P/R/F1/AP
  multichannel_across_models.png  -- x = 5 models, bars = P/R/F1/AP

Style follows ood_comparison/plot_ood_deltas.py /
yolo11/eval_figures/plot_eval_metrics.py.

Example
-------
    python3 model_comparison_figures/plot_eval_metrics.py
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGDIR = HERE
FIGDIR.mkdir(parents=True, exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e2", "#ffffff"
# One hue per metric: red, amber, green, blue.
CAT = ["#cf3a34", "#e8a33d", "#2f9e5b", "#2f6fb2"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 16, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})

METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("ap", "AP")]
METRIC_KEYS = [k for k, _ in METRICS]

SEEDS = [123, 456, 789]

# key -> (results_dir, run_name_fn(variant, seed), display label)
MODELS = [
    ("yolo11n", ROOT / "yolo11" / "eval_results_customaug",
     lambda v, s: f"yolo11n_{v}_s{s}", "YOLO11n"),
    ("yolo11s", ROOT / "yolo11" / "eval_results_customaug",
     lambda v, s: f"yolo11s_{v}_s{s}", "YOLO11s"),
    ("frcnn", ROOT / "faster-rcnn" / "eval_results",
     lambda v, s: f"frcnn_{v}_s{s}", "Faster R-CNN"),
    ("rtdetr", ROOT / "rt-detr" / "eval_results_lr5e5",
     lambda v, s: f"rt_detr_{v}_s{s}", "RT-DETR"),
    ("cnntrans", ROOT / "cnn-transformer" / "eval_results",
     lambda v, s: f"cnntrans_{v}_{s}", "CNN-Transformer"),
]
MODEL_ORDER = [key for key, *_ in MODELS]
MODEL_LABEL = {key: label for key, _, _, label in MODELS}

# Same underlying channels, different short codes per model.
VARIANT_CODE = {
    "yolo11n": {"gaussian": "gauss", "multichannel": "frgb"},
    "yolo11s": {"gaussian": "gauss", "multichannel": "frgb"},
    "frcnn": {"gaussian": "gh", "multichannel": "frgb"},
    "rtdetr": {"gaussian": "ghp", "multichannel": "frgb"},
    "cnntrans": {"gaussian": "gfh", "multichannel": "shg"},
}
CONCEPT_LABEL = {"gaussian": "Gaussian high-pass", "multichannel": "multichannel"}


def load_one(results_dir, run_name_fn, variant):
    """seeds' block_metrics_overall.csv -> {"mean": {...}, "std": {...}, "n": int}."""
    cols = {k: [] for k in METRIC_KEYS}
    for seed in SEEDS:
        p = results_dir / run_name_fn(variant, seed) / "block_metrics_overall.csv"
        if not p.is_file():
            print(f"WARNING: missing {p}")
            continue
        with p.open() as fh:
            row = next(csv.DictReader(fh))
        for k in METRIC_KEYS:
            cols[k].append(float(row[k]))

    n = len(cols[METRIC_KEYS[0]])
    if n == 0:
        raise SystemExit(f"no seed files found under {results_dir} for variant {variant!r}")
    if n != 3:
        print(f"WARNING: {results_dir.name}/*{variant}* has {n} seed file(s), expected 3")

    return {
        "mean": {k: float(np.mean(v)) for k, v in cols.items()},
        "std": {k: float(np.std(v, ddof=1)) if n > 1 else 0.0 for k, v in cols.items()},
        "n": n,
    }


def load_stats():
    """{"gaussian": {model_key: stats}, "multichannel": {model_key: stats}}."""
    stats = {"gaussian": {}, "multichannel": {}}
    for key, results_dir, run_name_fn, _label in MODELS:
        for concept in ("gaussian", "multichannel"):
            variant = VARIANT_CODE[key][concept]
            stats[concept][key] = load_one(results_dir, run_name_fn, variant)
    return stats


def print_table(stats_for_concept, concept):
    hdr = f"{'model':<20}" + "".join(f"{lbl:>22}" for _, lbl in METRICS)
    print("\n" + "=" * len(hdr))
    print(f"Block-level metrics -- {CONCEPT_LABEL[concept]} (mean +/- std across seeds)")
    print("=" * len(hdr))
    print(hdr)
    for key in MODEL_ORDER:
        s = stats_for_concept[key]
        row = f"{MODEL_LABEL[key] + ' (' + VARIANT_CODE[key][concept] + ')':<20}"
        for k in METRIC_KEYS:
            row += f"{s['mean'][k]:>13.3f} +/- {s['std'][k]:.3f}"
        print(row)


def grouped_bar(stats_for_concept, concept, out_name, title):
    order = MODEL_ORDER
    x = np.arange(len(order))
    n = len(METRICS)
    w = 0.8 / n  # n bars per group
    fig, ax = plt.subplots(figsize=(16, 9))  # exact 16:9

    hi = 0.0
    for i, (key, lbl) in enumerate(METRICS):
        means = [stats_for_concept[m]["mean"][key] for m in order]
        stds = [stats_for_concept[m]["std"][key] for m in order]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, means, w, color=CAT[i], label=lbl,
                      yerr=stds, error_kw=dict(ecolor=MUTED, lw=1, capsize=2), zorder=2)
        for b, m, s in zip(bars, means, stds):
            ax.annotate(f"{m:.2f}", (b.get_x() + b.get_width() / 2, m + s),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=10, color=INK2)
        hi = max(hi, max(m + s for m, s in zip(means, stds)))

    ax.set_ylim(0, max(1.0, hi) + 0.08)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{MODEL_LABEL[m]}\n({VARIANT_CODE[m][concept]})" for m in order], fontsize=16)
    ax.set_ylabel("score  (mean ± std across 3 seeds)", fontsize=17)
    ax.set_title(title, fontsize=20, color=INK, pad=14)
    ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              fontsize=17)

    # keep the saved canvas exactly 16:9 -- no bbox_inches="tight" crop
    fig.tight_layout()
    out = FIGDIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    stats = load_stats()
    for concept, out_name, title in [
        ("gaussian", "gaussian_hp_across_models.png",
         "Block-level metrics: Gaussian high-pass — across model families"),
        ("multichannel", "multichannel_across_models.png",
         "Block-level metrics: multichannel — across model families"),
    ]:
        print_table(stats[concept], concept)
        print("wrote", grouped_bar(stats[concept], concept, out_name, title))
        print()


if __name__ == "__main__":
    main()
