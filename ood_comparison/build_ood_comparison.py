#!/usr/bin/env python3
"""Combine YOLO11 + CNN-Transformer OOD block-level eval results, and difference
each OOD study region against the original Bay-of-Bengal test-split ("og test set").

Inputs (all pre-aggregated seed mean/std CSVs already on disk):
  CNN-Transformer OOD : cnn-transformer/eval_results_ood/summary_all_datasets.csv
  CNN-Transformer BoB : cnn-transformer/eval_results/summary_by_variant.csv
  YOLO11 OOD          : yolo11/eval_results_ood/summary_all_datasets.csv
  YOLO11 BoB baseline : yolo11/eval_results_customaug/summary_by_variant.csv
                        (customaug models were the ones scored OOD)

Outputs (this directory):
  ood_eval_combined.csv   one row per (model, variant, region): <m>_mean/<m>_std
  ood_vs_ogtest_diff.csv  one row per (model, variant, region):
                          <m>_ood_mean, <m>_ogtest_mean,
                          <m>_delta      = ood_mean - ogtest_mean
                          <m>_delta_std  = sqrt(ood_std^2 + ogtest_std^2)
                                           (independent-seed approximation)
"""
import os

import numpy as np
import pandas as pd

REPO = "/workspace/isw-detector"
OUT_DIR = os.path.join(REPO, "ood_comparison")

CT_OOD = os.path.join(REPO, "cnn-transformer/eval_results_ood/summary_all_datasets.csv")
CT_BOB = os.path.join(REPO, "cnn-transformer/eval_results/summary_by_variant.csv")
YOLO_OOD = os.path.join(REPO, "yolo11/eval_results_ood/summary_all_datasets.csv")
YOLO_BOB = os.path.join(REPO, "yolo11/eval_results_customaug/summary_by_variant.csv")

CORE_METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1",
    "iou", "ap", "pr_auc", "roc_auc", "threshold",
]

CHANNEL_CONFIG = {
    "gfh": "gaussian_filter_highpass",
    "gauss": "gaussian_filter_highpass",
    "shg": "ssha+rollmean+gradmag (3ch)",
    "frgb": "scientific_rgb (ssha/gauss_hp/gradmag)",
}

ID_COLS = ["model", "model_size", "variant", "channel_config", "region", "n_seeds"]


def _mean_std_cols():
    cols = []
    for m in CORE_METRICS:
        cols += [f"{m}_mean", f"{m}_std"]
    return cols


def load_ood(path, model, model_size):
    df = pd.read_csv(path)
    df = df.rename(columns={"dataset": "region"})
    df["model"] = model
    df["model_size"] = model_size
    df["channel_config"] = df["variant"].map(CHANNEL_CONFIG)
    keep = ID_COLS + _mean_std_cols()
    return df[keep]


def load_bob(path, model, model_size, variants):
    df = pd.read_csv(path)
    if "model_size" in df.columns:
        df = df[df["model_size"] == model_size]
    df = df[df["variant"].isin(variants)].copy()
    df["model"] = model
    df["model_size"] = model_size
    keep = ["model", "model_size", "variant", "n_seeds"] + _mean_std_cols()
    return df[keep]


# ---- CSV 1: combined OOD results ------------------------------------------------
ct_ood = load_ood(CT_OOD, "cnn-transformer", "cnntrans")
yolo_ood = load_ood(YOLO_OOD, "yolo11", "yolo11n")
combined = pd.concat([ct_ood, yolo_ood], ignore_index=True)
combined = combined.sort_values(["model", "variant", "region"]).reset_index(drop=True)
combined.to_csv(os.path.join(OUT_DIR, "ood_eval_combined.csv"), index=False)

# ---- CSV 2: OOD region minus og (BoB) test set -------------------------------
ct_bob = load_bob(CT_BOB, "cnn-transformer", "cnntrans", ["gfh", "shg"])
yolo_bob = load_bob(YOLO_BOB, "yolo11", "yolo11n", ["frgb", "gauss"])
bob = pd.concat([ct_bob, yolo_bob], ignore_index=True)

rows = []
for _, o in combined.iterrows():
    b = bob[(bob["model"] == o["model"]) & (bob["variant"] == o["variant"])]
    assert len(b) == 1, f"baseline join failed for {o['model']}/{o['variant']}"
    b = b.iloc[0]
    row = {c: o[c] for c in ID_COLS}
    for m in CORE_METRICS:
        om, os_ = float(o[f"{m}_mean"]), float(o[f"{m}_std"])
        bm, bs = float(b[f"{m}_mean"]), float(b[f"{m}_std"])
        row[f"{m}_ood_mean"] = om
        row[f"{m}_ogtest_mean"] = bm
        row[f"{m}_delta"] = om - bm
        row[f"{m}_delta_std"] = float(np.sqrt(os_ ** 2 + bs ** 2))
    rows.append(row)

diff = pd.DataFrame(rows).sort_values(["model", "variant", "region"]).reset_index(drop=True)
diff.to_csv(os.path.join(OUT_DIR, "ood_vs_ogtest_diff.csv"), index=False)

print(f"ood_eval_combined.csv   : {combined.shape[0]} rows x {combined.shape[1]} cols")
print(f"ood_vs_ogtest_diff.csv  : {diff.shape[0]} rows x {diff.shape[1]} cols")
print(f"NaNs -> combined: {int(combined.isna().sum().sum())}  diff: {int(diff.isna().sum().sum())}")
