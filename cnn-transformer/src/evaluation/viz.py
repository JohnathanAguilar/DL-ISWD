from __future__ import annotations

import json
import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _ensure_parent_dir(path: Optional[str]) -> None:
    if path is None:
        return

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _write_metrics_sidecar(
    save_path: Optional[str],
    metrics: Optional[Mapping[str, Any]],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Write a JSON file with the same basename as an image, containing that
    image's metrics. This keeps every saved sample plot self-describing even
    if it is later copied out of its evaluation directory.
    """
    if save_path is None or metrics is None:
        return

    _ensure_parent_dir(save_path)

    sidecar_path = os.path.splitext(save_path)[0] + ".json"
    payload = dict(extra or {})
    payload.update(metrics)

    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def _save_or_show(
    save_path: Optional[str],
    show: bool,
    dpi: int = 160,
) -> None:
    if save_path is not None:
        _ensure_parent_dir(save_path)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def _valid_values_from_sample_results(
    sample_results: Sequence[Mapping[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect y_true and y_prob over valid blocks from sample_results.
    """
    y_true_parts = []
    y_prob_parts = []

    for sample in sample_results:
        labels = np.asarray(sample["labels"])
        probs = np.asarray(sample["probs"])
        valid = np.asarray(sample["valid"]).astype(bool)

        if labels.shape != probs.shape or labels.shape != valid.shape:
            raise ValueError(
                "Sample labels/probs/valid shapes must match. Got "
                f"labels={labels.shape}, probs={probs.shape}, valid={valid.shape}"
            )

        if valid.sum() == 0:
            continue

        y_true_parts.append(labels[valid].astype(bool))
        y_prob_parts.append(probs[valid].astype(np.float32))

    if not y_true_parts:
        return np.array([], dtype=bool), np.array([], dtype=np.float32)

    y_true = np.concatenate(y_true_parts, axis=0)
    y_prob = np.concatenate(y_prob_parts, axis=0)

    return y_true, y_prob


def _roc_curve_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ROC curve arrays without sklearn.

    Returns:
        fpr, tpr
    """
    y_true = y_true.astype(bool, copy=False)
    y_score = y_score.astype(np.float64, copy=False)

    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())

    if y_true.size == 0 or n_pos == 0 or n_neg == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    tpr = [0.0]
    fpr = [0.0]

    tp = 0
    fp = 0
    i = 0

    while i < len(y_sorted):
        score = score_sorted[i]

        while i < len(y_sorted) and score_sorted[i] == score:
            if y_sorted[i]:
                tp += 1
            else:
                fp += 1
            i += 1

        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)

    return np.asarray(fpr, dtype=np.float64), np.asarray(tpr, dtype=np.float64)


def _pr_curve_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build precision-recall curve arrays without sklearn.

    Returns:
        recall, precision
    """
    y_true = y_true.astype(bool, copy=False)
    y_score = y_score.astype(np.float64, copy=False)

    n_pos = int(y_true.sum())

    if y_true.size == 0 or n_pos == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    recall = [0.0]
    precision = [1.0]

    tp = 0
    fp = 0
    i = 0

    while i < len(y_sorted):
        score = score_sorted[i]

        while i < len(y_sorted) and score_sorted[i] == score:
            if y_sorted[i]:
                tp += 1
            else:
                fp += 1
            i += 1

        rec = tp / n_pos
        prec = tp / max(tp + fp, 1)

        recall.append(rec)
        precision.append(prec)

    return np.asarray(recall, dtype=np.float64), np.asarray(precision, dtype=np.float64)


def plot_confusion_matrix(
    metrics: Mapping[str, Any],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plot a 2x2 confusion matrix from evaluation metrics.

    Matrix layout:
        rows:    true label [0, 1]
        columns: predicted label [0, 1]
    """
    tn = int(metrics.get("tn", 0))
    fp = int(metrics.get("fp", 0))
    fn = int(metrics.get("fn", 0))
    tp = int(metrics.get("tp", 0))

    matrix = np.array(
        [
            [tn, fp],
            [fn, tp],
        ],
        dtype=np.int64,
    )

    plt.figure(figsize=(5.5, 4.8))
    plt.imshow(matrix, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for row in range(2):
        for col in range(2):
            value = matrix[row, col]
            plt.text(
                col,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.colorbar(label="Count")
    plt.tight_layout()

    _save_or_show(save_path, show)


def plot_metric_bar(
    metrics: Mapping[str, Any],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "Evaluation Metrics",
) -> None:
    """
    Plot key scalar metrics as a bar chart.
    """
    keys = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "iou",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
    ]

    labels = []
    values = []

    for key in keys:
        if key in metrics:
            value = float(metrics[key])
            if np.isfinite(value):
                labels.append(key)
                values.append(value)

    if not values:
        raise ValueError("No plottable metric keys were found.")

    plt.figure(figsize=(10.5, 4.8))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title(title)
    plt.xticks(rotation=35, ha="right")

    for idx, value in enumerate(values):
        plt.text(
            idx,
            min(value + 0.02, 1.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    _save_or_show(save_path, show)


def plot_auc_bar(
    metrics: Mapping[str, Any],
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "AUC Scores",
) -> None:
    """
    Plot ROC-AUC and PR-AUC as a small bar chart.
    """
    raw_values = {
        "ROC-AUC": float(metrics.get("roc_auc", float("nan"))),
        "PR-AUC": float(metrics.get("pr_auc", float("nan"))),
    }

    labels = []
    values = []

    for label, value in raw_values.items():
        if np.isfinite(value):
            labels.append(label)
            values.append(value)

    if not values:
        raise ValueError(
            "ROC-AUC and PR-AUC are undefined. This usually happens when "
            "the test set contains only one class."
        )

    plt.figure(figsize=(5.5, 4.5))
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUC")
    plt.title(title)

    for idx, value in enumerate(values):
        plt.text(
            idx,
            min(value + 0.025, 1.02),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    plt.tight_layout()

    _save_or_show(save_path, show)


def plot_auc_curves(
    sample_results: Sequence[Mapping[str, Any]],
    metrics: Optional[Mapping[str, Any]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    title: str = "AUC Curves",
) -> None:
    """
    Plot ROC and Precision-Recall curves.

    Saves one chart with two panels:
        left: ROC curve
        right: Precision-Recall curve
    """
    y_true, y_prob = _valid_values_from_sample_results(sample_results)

    if y_true.size == 0:
        raise ValueError("No valid values found in sample_results.")

    fpr, tpr = _roc_curve_arrays(y_true, y_prob)
    recall, precision = _pr_curve_arrays(y_true, y_prob)

    roc_auc = float("nan")
    pr_auc = float("nan")

    if metrics is not None:
        roc_auc = float(metrics.get("roc_auc", float("nan")))
        pr_auc = float(metrics.get("pr_auc", float("nan")))

    plt.figure(figsize=(11.5, 5.0))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    if fpr.size > 0 and tpr.size > 0:
        label = "ROC"
        if np.isfinite(roc_auc):
            label += f" AUC = {roc_auc:.3f}"

        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Random")
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "ROC-AUC undefined\nonly one class present",
            ha="center",
            va="center",
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.subplot(1, 2, 2)
    if recall.size > 0 and precision.size > 0:
        label = "PR"
        if np.isfinite(pr_auc):
            label += f" AUC = {pr_auc:.3f}"

        plt.plot(recall, precision, linewidth=2, label=label)

        positive_rate = float(y_true.mean())
        plt.axhline(
            positive_rate,
            linestyle="--",
            linewidth=1,
            label=f"Baseline = {positive_rate:.3f}",
        )

        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "PR-AUC undefined\nno positive labels",
            ha="center",
            va="center",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.tight_layout()

    _save_or_show(save_path, show)


def plot_probability_histogram(
    sample_results: Sequence[Mapping[str, Any]],
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = False,
    bins: int = 40,
    title: str = "Predicted Probability Distribution",
) -> None:
    """
    Plot probability histograms for true-negative-class and true-positive-class blocks.
    """
    y_true, y_prob = _valid_values_from_sample_results(sample_results)

    if y_true.size == 0:
        raise ValueError("No valid values found in sample_results.")

    probs_neg = y_prob[~y_true]
    probs_pos = y_prob[y_true]

    plt.figure(figsize=(8.5, 5.0))

    if probs_neg.size > 0:
        plt.hist(
            probs_neg,
            bins=bins,
            alpha=0.55,
            label="True label 0",
        )

    if probs_pos.size > 0:
        plt.hist(
            probs_pos,
            bins=bins,
            alpha=0.55,
            label="True label 1",
        )

    plt.axvline(
        float(threshold),
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.2f}",
    )

    plt.xlabel("Predicted probability")
    plt.ylabel("Valid block count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    _save_or_show(save_path, show)


def plot_sample_result(
    sample: Mapping[str, Any],
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    Plot one sample's labels, probabilities, binary predictions, and errors.

    Expected sample keys:
        id
        labels: [Bh, Bw]
        probs:  [Bh, Bw]
        pred:   [Bh, Bw]
        valid:  [Bh, Bw]
        metrics: optional dict of this sample's metrics (accuracy, precision,
            recall, f1, roc_auc, pr_auc, tp/fp/tn/fn, etc.). When present and
            save_path is given, it is written as a JSON file next to the
            image, e.g. "00005_sample.png" -> "00005_sample.json".
    """
    sample_id = str(sample.get("id", "sample"))

    labels = np.asarray(sample["labels"])
    probs = np.asarray(sample["probs"])
    pred = np.asarray(sample["pred"]).astype(bool)
    valid = np.asarray(sample["valid"]).astype(bool)

    if labels.shape != probs.shape or labels.shape != pred.shape or labels.shape != valid.shape:
        raise ValueError(
            "Sample labels/probs/pred/valid shapes must match. Got "
            f"labels={labels.shape}, probs={probs.shape}, "
            f"pred={pred.shape}, valid={valid.shape}"
        )

    labels_bool = labels.astype(bool)

    labels_plot = labels.astype(np.float32)
    probs_plot = probs.astype(np.float32)
    pred_plot = pred.astype(np.float32)

    labels_plot = np.where(valid, labels_plot, np.nan)
    probs_plot = np.where(valid, probs_plot, np.nan)
    pred_plot = np.where(valid, pred_plot, np.nan)

    errors = np.zeros_like(probs_plot, dtype=np.float32)
    errors[np.logical_and(valid, labels_bool == pred)] = 0.0
    errors[np.logical_and(valid, labels_bool != pred)] = 1.0
    errors[~valid] = np.nan

    fig_title = title if title is not None else f"Sample Result: {sample_id}"

    plt.figure(figsize=(14.0, 4.2))
    plt.suptitle(fig_title)

    plt.subplot(1, 4, 1)
    plt.imshow(labels_plot, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title("Ground Truth")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 2)
    plt.imshow(probs_plot, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title("Probability")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 3)
    plt.imshow(pred_plot, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title(f"Prediction >= {threshold:.2f}")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.subplot(1, 4, 4)
    plt.imshow(errors, interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.title("Errors")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()

    _save_or_show(save_path, show)
    _write_metrics_sidecar(
        save_path,
        sample.get("metrics"),
        extra={"id": sample_id},
    )


def plot_top_error_samples(
    sample_results: Sequence[Mapping[str, Any]],
    save_dir: str,
    threshold: float = 0.5,
    top_k: int = 8,
) -> None:
    """
    Save sample plots for the samples with the lowest per-sample F1 score.

    Samples with zero valid positives can have F1=0 even if they are mostly correct,
    so use this as a quick diagnostic rather than a definitive ranking.
    """
    os.makedirs(save_dir, exist_ok=True)

    ranked = sorted(
        sample_results,
        key=lambda sample: float(sample.get("metrics", {}).get("f1", 0.0)),
    )

    for idx, sample in enumerate(ranked[:top_k]):
        sample_id = str(sample.get("id", "sample")).replace("/", "_").replace("\\", "_")
        save_path = os.path.join(save_dir, f"lowest_f1_{idx:03d}_{sample_id}.png")

        plot_sample_result(
            sample=sample,
            threshold=threshold,
            save_path=save_path,
            show=False,
            title=f"Lowest F1 #{idx + 1}: {sample_id}",
        )


def plot_dense_comparison(
    image: np.ndarray,
    patch_labels: np.ndarray,
    dense_by_method: Mapping[str, np.ndarray],
    block_size: int,
    save_path: Optional[str] = None,
    show: bool = False,
    title: Optional[str] = None,
    zoom: Optional[Tuple[int, int, int, int]] = None,
    guide_index: int = -1,
    layout: str = "auto",
    panel_size: float = 3.0,
    input_cmap: Optional[str] = None,
    map_cmap: Optional[str] = None,
    overlay_on_input: bool = False,
    overlay_alpha: float = 0.55,
) -> None:
    """
    Qualitative figure: input, patch-level ground truth, and one panel per dense
    reconstruction method.

    The expected reading is that nearest gives square blocks, bilinear and
    bicubic give a generic blur that ignores the image, and the learned head
    produces confidence that follows the image structure.

    Args:
        image:
            Input channels [C, Ht, Wt]. Panel 1 shows channel `guide_index`.
        patch_labels:
            Block-level ground truth [Bh, Bw]; drawn at pixel resolution so the
            panels line up.
        dense_by_method:
            Ordered mapping of method name -> dense map [Ht, Wt] in [0, 1].
            For example: {"Nearest": ..., "Bilinear": ..., "Learned": ...}.
        block_size:
            Original block size in pixels, used to expand patch_labels.
        zoom:
            Optional (y0, y1, x0, x1) pixel window. When given, an extra row (or
            column) shows the same panels restricted to that window, which is
            where an internal-wave signature crossing block boundaries is visible.
        layout:
            "columns" puts methods side by side, "rows" stacks them and rotates
            each panel 90 degrees, and "auto" (default) picks rows whenever the
            image is more than twice as tall as it is wide.

            A SWOT pass is roughly 700x60, about 12:1. Six of those side by side
            get squeezed to slivers, so tall inputs are laid out as rotated
            horizontal strips instead, which also suits a two-column paper.
        panel_size:
            Base panel edge in inches; the other dimension follows the data
            aspect ratio so pixels stay square.
        input_cmap:
            Colormap for the "Input" panel. None uses matplotlib's default.
        map_cmap:
            Colormap for the ground-truth and dense-method panels, which all
            share the same [0, 1] scale. None uses matplotlib's default.
        overlay_on_input:
            When True, each dense-method panel (not "Input" or "Patch ground
            truth") is drawn as the input image with the method's confidence
            map layered on top at `overlay_alpha`, instead of the confidence
            map alone.
        overlay_alpha:
            Opacity of the confidence map when `overlay_on_input` is True.
    """
    image = np.asarray(image)
    patch_labels = np.asarray(patch_labels)

    if image.ndim != 3:
        raise ValueError(f"image must have shape [C, Ht, Wt]; got {image.shape}")
    if patch_labels.ndim != 2:
        raise ValueError(f"patch_labels must have shape [Bh, Bw]; got {patch_labels.shape}")
    if not dense_by_method:
        raise ValueError("dense_by_method must contain at least one method")

    guide = image[guide_index]
    Ht, Wt = guide.shape

    gt_dense = np.kron(
        patch_labels.astype(np.float32),
        np.ones((block_size, block_size), dtype=np.float32),
    )

    if gt_dense.shape != (Ht, Wt):
        raise ValueError(
            f"Expanded ground truth {gt_dense.shape} does not match image {(Ht, Wt)}; "
            f"check block_size={block_size}."
        )

    for name, dense in dense_by_method.items():
        dense = np.asarray(dense)
        if dense.shape != (Ht, Wt):
            raise ValueError(
                f"Dense map '{name}' has shape {dense.shape}, expected {(Ht, Wt)}"
            )

    panels = [("Input", guide, None), ("Patch ground truth", gt_dense, (0.0, 1.0))]
    panels += [(name, np.asarray(d), (0.0, 1.0)) for name, d in dense_by_method.items()]

    if layout not in ("auto", "rows", "columns"):
        raise ValueError(f"layout must be 'auto', 'rows', or 'columns'; got {layout!r}")

    if layout == "auto":
        layout = "rows" if Ht > 2 * Wt else "columns"

    stack_rows = layout == "rows"
    n_views = 2 if zoom is not None else 1
    n_panels = len(panels)

    # Keep pixels square: the panel's short edge is panel_size and the long edge
    # follows the data aspect ratio, clamped so one strip cannot blow up the page.
    display_aspect = (Wt / Ht) if stack_rows else (Ht / Wt)
    display_aspect = float(np.clip(display_aspect, 0.15, 6.0))

    if stack_rows:
        n_rows, n_cols = n_panels, n_views
        fig_w = panel_size * 2.2 * n_cols
        fig_h = max(1.0, panel_size * 2.2 * display_aspect) * n_rows
    else:
        n_rows, n_cols = n_views, n_panels
        fig_w = panel_size * n_cols
        fig_h = max(1.6, panel_size * display_aspect) * n_rows

    fig_h = min(fig_h, 60.0)

    plt.figure(figsize=(fig_w, fig_h))

    if title is not None:
        plt.suptitle(title)

    views = [(None, "")]
    if zoom is not None:
        views.append((zoom, " (zoom)"))

    for panel_i, (name, data, vlim) in enumerate(panels):
        is_dense_method = panel_i >= 2  # 0 = Input, 1 = Patch ground truth
        for view_i, (window, suffix) in enumerate(views):
            if stack_rows:
                index = panel_i * n_cols + view_i + 1
            else:
                index = view_i * n_cols + panel_i + 1

            plt.subplot(n_rows, n_cols, index)

            view = data
            bg = guide
            if window is not None:
                y0, y1, x0, x1 = window
                view = data[y0:y1, x0:x1]
                bg = guide[y0:y1, x0:x1]

            if stack_rows:
                view = view.T
                bg = bg.T

            if overlay_on_input and is_dense_method:
                bg_kwargs = {"interpolation": "nearest"}
                if input_cmap is not None:
                    bg_kwargs["cmap"] = input_cmap
                plt.imshow(bg, **bg_kwargs)

                fg_kwargs = {"interpolation": "nearest", "alpha": overlay_alpha}
                if vlim is not None:
                    fg_kwargs["vmin"], fg_kwargs["vmax"] = vlim
                if map_cmap is not None:
                    fg_kwargs["cmap"] = map_cmap
                plt.imshow(view, **fg_kwargs)
            else:
                kwargs = {"interpolation": "nearest"}
                if vlim is not None:
                    kwargs["vmin"], kwargs["vmax"] = vlim

                cmap = input_cmap if panel_i == 0 else map_cmap
                if cmap is not None:
                    kwargs["cmap"] = cmap

                plt.imshow(view, **kwargs)

            plt.title(f"{name}{suffix}", fontsize=10)
            plt.xticks([])
            plt.yticks([])
            plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()

    _save_or_show(save_path, show)


def plot_best_samples(
    sample_results: Sequence[Mapping[str, Any]],
    save_dir: str,
    threshold: float = 0.5,
    top_k: int = 8,
) -> None:
    """
    Save sample plots for the samples with the highest per-sample F1 score.
    """
    os.makedirs(save_dir, exist_ok=True)

    ranked = sorted(
        sample_results,
        key=lambda sample: float(sample.get("metrics", {}).get("f1", 0.0)),
        reverse=True,
    )

    for idx, sample in enumerate(ranked[:top_k]):
        sample_id = str(sample.get("id", "sample")).replace("/", "_").replace("\\", "_")
        save_path = os.path.join(save_dir, f"highest_f1_{idx:03d}_{sample_id}.png")

        plot_sample_result(
            sample=sample,
            threshold=threshold,
            save_path=save_path,
            show=False,
            title=f"Highest F1 #{idx + 1}: {sample_id}",
        )