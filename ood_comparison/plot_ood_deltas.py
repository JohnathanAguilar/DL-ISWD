#!/usr/bin/env python3
"""Bar charts of the OOD-region minus og-test-set deltas.

Reads ood_vs_ogtest_diff.csv (from build_ood_comparison.py); writes four bar-chart
PNGs to ood_comparison/figures/.

Metrics: precision, recall, f1, ap.
Delta = <metric>_ood_mean - <metric>_ogtest_mean   (negative => OOD worse than BoB test).
Error bars = <metric>_delta_std = sqrt(ood_std^2 + ogtest_std^2)  (independent-seed approx).

Naming follows the paper (Fig. 3 / Table S4): model names CNN-Transformer /
YOLO11n; representations "Gaussian HP" (gfh, gauss), "multichannel" (shg),
"scientific RGB" (frgb). Config order is the two CNN-Transformer variants first
(Gaussian HP, then multichannel), then the two YOLO11n variants (Gaussian HP,
then scientific RGB).

Colour:
  config-labelled figures (delta_grouped_by_metric, delta_by_model) : keyed by
      model family -- the CNN-Transformer in a blue pair (#1b4f9c Gaussian HP /
      #7fb0e6 multichannel), YOLO11n in an orange pair (#b5410e Gaussian HP /
      #f2a65a scientific RGB).
  delta_ranked_diverging : sign colour (#e34948 red = OOD worse, #2a78d6 blue =
      OOD better); the config family palette shows only in the y-tick labels.
  delta_mean_by_region : categorical slots #2a78d6 blue, #eb6834 orange,
      #1baf7a green (one per study region).
The signed CSV is the table view (relief for the sub-3:1 slots on white).
"""
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "ood_vs_ogtest_diff.csv")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e2", "#ffffff"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
POS, NEG = "#2a78d6", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})

METRICS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("ap", "AP")]
REGIONS = ["amazon_mouth", "mascarene_ridge", "south_china_sea"]
REGION_LBL = {"amazon_mouth": "Amazon Mouth", "mascarene_ridge": "Mascarene Ridge",
              "south_china_sea": "South China Sea"}
REGION_2L = {"amazon_mouth": "Amazon\nMouth", "mascarene_ridge": "Mascarene\nRidge",
             "south_china_sea": "South China\nSea"}
REGION_AB = {"amazon_mouth": "AM", "mascarene_ridge": "MR", "south_china_sea": "SCS"}

df = pd.read_csv(CSV)
df["config"] = df["model"].map({"cnn-transformer": "CNN-Tr", "yolo11": "YOLO11"}) + " · " + df["variant"]

# Model/representation naming + palette, shared by every config-labelled figure.
# Order follows the paper (Fig. 3 / Table S4): the two CNN-Transformer configs
# first (Gaussian high-pass, then the multichannel input), then the two YOLO11n
# configs (Gaussian high-pass, then scientific RGB). Colour is keyed by model
# family -- a blue pair for the CNN-Transformer, an orange pair for YOLO11n --
# with the darker shade marking the Gaussian-high-pass representation.
CONFIG_ORDER = ["CNN-Tr · gfh", "CNN-Tr · shg", "YOLO11 · gauss", "YOLO11 · frgb"]
CONFIG_LABEL = {
    "CNN-Tr · gfh":   "CNN–Transformer · Gaussian HP",
    "CNN-Tr · shg":   "CNN–Transformer · multichannel",
    "YOLO11 · gauss": "YOLO11n · Gaussian HP",
    "YOLO11 · frgb":  "YOLO11n · scientific RGB",
}
CONFIG_COLOR = {
    "CNN-Tr · gfh":   "#1b4f9c",   # CNN-Transformer -- dark blue  (Gaussian HP)
    "CNN-Tr · shg":   "#7fb0e6",   # CNN-Transformer -- light blue (multichannel)
    "YOLO11 · gauss": "#b5410e",   # YOLO11n -- dark orange (Gaussian HP)
    "YOLO11 · frgb":  "#f2a65a",   # YOLO11n -- light amber (scientific RGB)
}
# Per-model representation axis (model name shown separately), for fig_by_model.
MODEL_LABEL = {"cnn-transformer": "CNN–Transformer", "yolo11": "YOLO11n"}
MODEL_VARIANTS = {"cnn-transformer": ["gfh", "shg"], "yolo11": ["gauss", "frgb"]}
VARIANT_LABEL = {"gfh": "Gaussian HP", "shg": "multichannel",
                 "gauss": "Gaussian HP", "frgb": "scientific RGB"}
VARIANT_COLOR = {"gfh": "#1b4f9c", "shg": "#7fb0e6",
                 "gauss": "#b5410e", "frgb": "#f2a65a"}


def hlabels(ax, bars, vals):
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for b, v in zip(bars, vals):
        w = b.get_width()
        off = 0.012 * span
        ax.text(w + (off if w >= 0 else -off), b.get_y() + b.get_height() / 2,
                f"{v:+.3f}", va="center", ha="left" if w >= 0 else "right",
                fontsize=7.5, color=INK2, clip_on=False)


def zero_y(ax):
    ax.axhline(0, color=INK, lw=2, zorder=3)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# 1. grouped: panel per metric, x = region, 4 bars = model/variant --------------
def fig_grouped_by_metric():
    piv = {m: df.pivot_table(index="region", columns="config", values=f"{m}_delta") for m, _ in METRICS}
    err = {m: df.pivot_table(index="region", columns="config", values=f"{m}_delta_std") for m, _ in METRICS}
    lo = min(v.values.min() for v in piv.values())
    hi = max(v.values.max() for v in piv.values())
    pad = 0.10 * (hi - lo)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=True)
    x = np.arange(len(REGIONS))
    w = 0.19
    for ax, (m, lbl) in zip(axes.flat, METRICS):
        ax.set_ylim(lo - pad, hi + pad)
        for i, cfg in enumerate(CONFIG_ORDER):
            vals = piv[m].reindex(REGIONS)[cfg].values
            errs = err[m].reindex(REGIONS)[cfg].values
            ax.bar(x + (i - 1.5) * w, vals, w, color=CONFIG_COLOR[cfg],
                   yerr=errs, error_kw=dict(ecolor=MUTED, lw=1, capsize=2), zorder=2)
        zero_y(ax)
        ax.set_title(lbl, fontsize=12, color=INK, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels([REGION_2L[r] for r in REGIONS])
        ax.set_ylabel("Δ  (OOD − BoB test)")

    handles = [Patch(facecolor=CONFIG_COLOR[c], label=CONFIG_LABEL[c]) for c in CONFIG_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("OOD generalisation gap by study region — detection metrics vs. Bay-of-Bengal test set",
                 y=1.0, fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(FIGDIR, "delta_grouped_by_metric.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# 2. faceted: rows = model, cols = metric; x = region, bars = variant ----------
def fig_by_model():
    models = ["cnn-transformer", "yolo11"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharey="row")
    x = np.arange(len(REGIONS))
    w = 0.34
    for r, mdl in enumerate(models):
        mdl_lbl = MODEL_LABEL[mdl]
        variants = MODEL_VARIANTS[mdl]
        sub = df[df["model"] == mdl]
        row_lo = min(sub[f"{m}_delta"].min() for m, _ in METRICS)
        row_hi = max(sub[f"{m}_delta"].max() for m, _ in METRICS)
        rpad = 0.30 * (row_hi - row_lo)
        for c, (m, lbl) in enumerate(METRICS):
            ax = axes[r, c]
            ax.set_ylim(row_lo - rpad, row_hi + rpad)
            for i, var in enumerate(variants):
                s = sub[sub["variant"] == var].set_index("region").reindex(REGIONS)
                ax.bar(x + (i - 0.5) * w, s[f"{m}_delta"].values, w, color=VARIANT_COLOR[var],
                       yerr=s[f"{m}_delta_std"].values,
                       error_kw=dict(ecolor=MUTED, lw=1, capsize=2), zorder=2)
            zero_y(ax)
            ax.set_xticks(x)
            ax.set_xticklabels([REGION_AB[r_] for r_ in REGIONS], fontsize=9)
            if r == 0:
                ax.set_title(lbl, fontsize=12, color=INK, pad=6)
            if c == 0:
                ax.set_ylabel(f"{mdl_lbl}\nΔ (OOD − BoB test)", fontsize=10)
        # per-row legend (each model has its own 2 representations) in the Recall
        # panel's empty half
        rh = [Patch(facecolor=VARIANT_COLOR[v], label=VARIANT_LABEL[v]) for v in variants]
        axes[r, 1].legend(handles=rh, frameon=False, fontsize=9,
                          loc="lower center", ncol=2, title=f"{mdl_lbl} representation")
    fig.suptitle("OOD deltas per model and preprocessing variant across the three study regions\n"
                 "AM = Amazon Mouth · MR = Mascarene Ridge · SCS = South China Sea",
                 y=1.02, fontsize=12.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIGDIR, "delta_by_model.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# 3. mean delta across the 4 configs: x = metric, bars = region ---------------
def fig_mean_by_region():
    rows = []
    for m, _ in METRICS:
        for reg in REGIONS:
            v = df[df["region"] == reg][f"{m}_delta"]
            rows.append(dict(metric=m, region=reg, mean=v.mean(), std=v.std(ddof=1)))
    g = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(METRICS))
    w = 0.26
    ax.set_ylim(g["mean"].min() - 0.11, g["mean"].max() + 0.09)
    for i, reg in enumerate(REGIONS):
        s = g[g["region"] == reg].set_index("metric").reindex([m for m, _ in METRICS])
        bars = ax.bar(x + (i - 1) * w, s["mean"].values, w, color=CAT[i],
                      yerr=s["std"].values, error_kw=dict(ecolor=MUTED, lw=1, capsize=3),
                      label=REGION_LBL[reg], zorder=2)
        for b, v in zip(bars, s["mean"].values):
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + (0.008 if h >= 0 else -0.008),
                    f"{v:+.3f}", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=8.5, color=INK2)
    zero_y(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRICS])
    ax.set_ylabel("mean Δ  (OOD − BoB test),  averaged over the 4 model/variant runs")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07))
    ax.set_title("How far each study region falls below in-distribution performance",
                 fontsize=13, color=INK, pad=10)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "delta_mean_by_region.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


# 4. ranked horizontal diverging bars: one panel per metric, stacked ----------
def fig_ranked_diverging():
    fig, axes = plt.subplots(4, 1, figsize=(10, 15))
    for ax, (m, lbl) in zip(axes, METRICS):
        s = df.copy()
        s["label"] = s["config"].map(CONFIG_LABEL) + " · " + s["region"].map(REGION_AB)
        s = s.sort_values(f"{m}_delta")
        vals = s[f"{m}_delta"].values
        colors = [POS if v >= 0 else NEG for v in vals]
        y = np.arange(len(s))
        bars = ax.barh(y, vals, color=colors, height=0.66, zorder=2)
        hlabels(ax, bars, vals)
        ax.axvline(0, color=INK, lw=2, zorder=3)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(s["label"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(lbl, fontsize=12, color=INK, pad=6, loc="left")
        ax.set_xlabel("Δ (OOD − BoB test)")
        xm = np.abs(vals).max() * 1.4
        ax.set_xlim(-xm, xm)
    handles = [Patch(facecolor=NEG, label="OOD worse"), Patch(facecolor=POS, label="OOD better")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("Every model × variant × region ranked by delta   "
                 "(AM Amazon Mouth · MR Mascarene Ridge · SCS South China Sea)",
                 y=1.012, fontsize=12.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(FIGDIR, "delta_ranked_diverging.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (fig_grouped_by_metric, fig_by_model, fig_mean_by_region, fig_ranked_diverging):
        print("wrote", fn())
