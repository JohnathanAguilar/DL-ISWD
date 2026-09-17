#!/usr/bin/env python3
"""YOLO11n box-level test-set metrics per preprocessing variant, seeds 123/456/789.

Reads yolo11/yolo_val_results_customaug/yolo11n_<variant>_s*/test_metrics.json
(10 variants x 3 seeds). Metrics: precision, recall, f1, map50. Per-variant
mean + sample std (ddof=1) across the 3 seeds, matching
aggregate_yolo_val_metrics.py.

Two variant groups, split by input family; each rendered as two PNGs:
  <stem>.png       grouped BAR chart -- percent change of each metric relative
                   to the SSHA baseline (variant 'ssha'), SSHA itself dropped,
                   error bars = propagated std.
  <stem>_line.png  LINE chart -- raw absolute 0-1 scores, SSHA included.

  highpass_test_metrics  -- bar: rollmean, stepconv, gauss, fft, gradmag
  composite_test_metrics -- bar: frgb, nohp, nossha, nograd

Style follows ood_comparison/plot_ood_deltas.py.

Example
-------
    cd yolo11 && python3 eval_figures/plot_eval_metrics.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE.parent / "yolo_val_results_customaug"
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

METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
           ("map50", "mAP@50")]
METRIC_KEYS = [k for k, _ in METRICS]

# Full spelled-out name of the SWOT field / composite each variant is trained on.
VARIANT_LABEL = {
    "ssha": "SSHA\n(quality-masked)",
    "rollmean": "Rolling-mean\nhigh-pass",
    "stepconv": "Stepped-convolution\nhigh-pass",
    "gauss": "Gaussian-filter\nhigh-pass",
    "fft": "FFT\nhigh-pass",
    "gradmag": "Gradient\nmagnitude",
    "frgb": "Scientific RGB\n(all channels)",
    "nohp": "Scientific RGB\nno Gaussian high-pass\n(G = 0)",
    "nossha": "Scientific RGB\nno SSHA\n(R = 0)",
    "nograd": "Scientific RGB\nno gradient magnitude\n(B = 0)",
}
BASELINE = "ssha"
# Line charts (raw absolute scores) keep the baseline row.
LINE_HIGHPASS = ["ssha", "rollmean", "stepconv", "gauss", "fft", "gradmag"]
LINE_COMPOSITE = ["frgb", "nohp", "nossha", "nograd"]
# Bar charts (percent change vs SSHA) drop the baseline itself.
BAR_HIGHPASS = ["rollmean", "stepconv", "gauss", "fft", "gradmag"]
BAR_COMPOSITE = ["frgb", "nohp", "nossha", "nograd"]
ALL_VARIANTS = LINE_HIGHPASS + LINE_COMPOSITE


def load_stats():
    """variant -> {"mean": {metric: float}, "std": {metric: float}, "n": int}."""
    stats = {}
    for variant in ALL_VARIANTS:
        paths = sorted(RESULTS_DIR.glob(f"yolo11n_{variant}_s*/test_metrics.json"))
        if len(paths) != 3:
            print(f"WARNING: {variant} has {len(paths)} seed file(s), expected 3")
        if not paths:
            raise SystemExit(f"no test_metrics.json found for variant {variant!r} under {RESULTS_DIR}")

        cols = {k: [] for k in METRIC_KEYS}
        for p in paths:
            m = json.loads(p.read_text())
            for k in METRIC_KEYS:
                cols[k].append(float(m[k]))

        stats[variant] = {
            "mean": {k: float(np.mean(v)) for k, v in cols.items()},
            "std": {k: float(np.std(v, ddof=1)) for k, v in cols.items()},
            "n": len(paths),
        }
    return stats


def print_table(stats):
    hdr = f"{'variant':<16}" + "".join(f"{lbl:>22}" for _, lbl in METRICS)
    print("\n" + "=" * len(hdr))
    print("YOLO11n test-set metrics by preprocessing variant (mean +/- std across seeds)")
    print("=" * len(hdr))
    print(hdr)
    for variant in ALL_VARIANTS:
        s = stats[variant]
        row = f"{variant:<16}"
        for k in METRIC_KEYS:
            row += f"{s['mean'][k]:>13.3f} +/- {s['std'][k]:.3f}"
        print(row)


def pct_change(stats, baseline):
    """Percent change of each metric vs the baseline variant, with the std of a
    ratio of two independent seed-means propagated to first order.

    variant (!= baseline) -> {"mean": {metric: pct}, "std": {metric: pct_std}}.
    """
    base = stats[baseline]
    out = {}
    for variant, s in stats.items():
        if variant == baseline:
            continue
        row = {"mean": {}, "std": {}}
        for k in METRIC_KEYS:
            mv, sv = s["mean"][k], s["std"][k]
            mb, sb = base["mean"][k], base["std"][k]
            ratio = mv / mb
            rel_err = np.hypot(sv / mv, sb / mb)
            row["mean"][k] = 100.0 * (ratio - 1.0)
            row["std"][k] = 100.0 * ratio * rel_err
        out[variant] = row
    return out


def print_pct_table(pct, baseline):
    hdr = f"{'variant':<16}" + "".join(f"{lbl:>22}" for _, lbl in METRICS)
    print("\n" + "=" * len(hdr))
    print(f"YOLO11n test-set metrics -- % change vs '{baseline}' baseline (propagated std)")
    print("=" * len(hdr))
    print(hdr)
    for variant in ALL_VARIANTS:
        if variant == baseline:
            continue
        p = pct[variant]
        row = f"{variant:<16}"
        for k in METRIC_KEYS:
            row += f"{p['mean'][k]:>+12.1f}% +/- {p['std'][k]:.1f}%"
        print(row)


def grouped_bar_pct(pct, order, out_name, title):
    """Grouped bars of percent change vs the SSHA baseline (see pct_change)."""
    x = np.arange(len(order))
    n = len(METRICS)
    w = 0.8 / n  # n bars per group
    fig, ax = plt.subplots(figsize=(16, 9))  # exact 16:9

    lo, hi = 0.0, 0.0
    for i, (key, lbl) in enumerate(METRICS):
        means = np.array([pct[v]["mean"][key] for v in order])
        stds = np.array([pct[v]["std"][key] for v in order])
        bars = ax.bar(x + (i - (n - 1) / 2) * w, means, w, color=CAT[i], label=lbl,
                      yerr=stds, error_kw=dict(ecolor=MUTED, lw=1, capsize=2), zorder=2)
        lo = min(lo, float((means - stds).min()))
        hi = max(hi, float((means + stds).max()))
        for b, v in zip(bars, means):
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3 if v >= 0 else -12),
                        ha="center", fontsize=10, color=INK2)

    pad = 0.15 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0, color=INK, lw=2, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in order], fontsize=16)
    ax.set_ylabel("% change vs SSHA baseline  (mean ± propagated std, 3 seeds)", fontsize=17)
    ax.set_title(title, fontsize=20, color=INK, pad=14)
    ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              fontsize=17)

    # keep the saved canvas exactly 16:9 -- no bbox_inches="tight" crop
    fig.tight_layout()
    out = FIGDIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def line_chart(stats, order, out_name, title):
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(16, 9))  # exact 16:9

    for i, (key, lbl) in enumerate(METRICS):
        means = np.array([stats[v]["mean"][key] for v in order])
        stds = np.array([stats[v]["std"][key] for v in order])
        ax.fill_between(x, means - stds, means + stds, color=CAT[i], alpha=0.15, zorder=1)
        ax.plot(x, means, color=CAT[i], lw=2.5, marker="o", ms=9, label=lbl, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=CAT[i], elinewidth=1.5,
                    capsize=4, zorder=2)

    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.35, len(order) - 0.65)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABEL[v] for v in order], fontsize=16)
    ax.set_ylabel("score  (mean ± std across 3 seeds)", fontsize=17)
    ax.set_title(title, fontsize=20, color=INK, pad=14)
    ax.legend(frameon=False, ncol=len(METRICS), loc="upper center",
              bbox_to_anchor=(0.5, -0.14), fontsize=17)

    fig.tight_layout()
    out = FIGDIR / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    stats = load_stats()
    print_table(stats)
    pct = pct_change(stats, BASELINE)
    print_pct_table(pct, BASELINE)
    print()
    groups = [
        ("highpass_test_metrics", BAR_HIGHPASS, LINE_HIGHPASS,
         "single-channel high-pass variants"),
        ("composite_test_metrics", BAR_COMPOSITE, LINE_COMPOSITE,
         "RGB-composite channel ablations"),
    ]
    for stem, bar_order, line_order, subtitle in groups:
        print("wrote", grouped_bar_pct(
            pct, bar_order, f"{stem}.png",
            f"YOLO11n metrics: % change vs SSHA baseline — {subtitle}"))
        print("wrote", line_chart(
            stats, line_order, f"{stem}_line.png",
            f"YOLO11n test-set metrics — {subtitle}"))


if __name__ == "__main__":
    main()
