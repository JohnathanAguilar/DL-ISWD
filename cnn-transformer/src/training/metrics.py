# src/training/metrics.py

import numpy as np
import torch

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        fbeta_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    SKLEARN_OK = True

except Exception:
    SKLEARN_OK = False


def compute_metrics(
    labels_v: torch.Tensor,
    probs_v: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """
    Compute binary classification metrics for valid block tokens.

    Parameters
    ----------
    labels_v : torch.Tensor
        Ground-truth binary labels with shape [N].

    probs_v : torch.Tensor
        Predicted positive-class probabilities with shape [N].

    threshold : float
        Probability threshold used to convert probabilities into
        binary predictions.

    Returns
    -------
    dict
        Dictionary containing threshold-independent and
        threshold-dependent evaluation metrics.
    """

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "f0_5",
        "iou",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "label_positive_rate",
        "pred_positive_rate",
    ]

    metrics = {
        name: float("nan")
        for name in metric_names
    }

    if not SKLEARN_OK:
        return metrics

    # Move tensors to CPU and convert to NumPy.
    y_true = (
        labels_v
        .detach()
        .reshape(-1)
        .cpu()
        .numpy()
        .astype(np.int32)
    )

    y_prob = (
        probs_v
        .detach()
        .reshape(-1)
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    if y_true.size == 0:
        return metrics

    # Convert probabilities to binary predictions.
    y_pred = (y_prob >= threshold).astype(np.int32)

    # --------------------------------------------------------
    # Basic class-distribution metrics
    # --------------------------------------------------------

    metrics["label_positive_rate"] = float(
        np.mean(y_true == 1)
    )

    metrics["pred_positive_rate"] = float(
        np.mean(y_pred == 1)
    )

    # --------------------------------------------------------
    # Threshold-independent metrics
    # --------------------------------------------------------

    # ROC AUC requires both classes to be present.
    if np.unique(y_true).size == 2:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_prob)
            )
        except ValueError:
            pass

    # Average precision is the standard scalar PR-AUC summary
    # commonly used for imbalanced binary classification.
    if np.any(y_true == 1):
        try:
            metrics["pr_auc"] = float(
                average_precision_score(y_true, y_prob)
            )
        except ValueError:
            pass

    # --------------------------------------------------------
    # Threshold-dependent metrics
    # --------------------------------------------------------

    metrics["accuracy"] = float(
        accuracy_score(y_true, y_pred)
    )

    metrics["precision"] = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics["recall"] = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics["f1"] = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics["f0_5"] = float(
        fbeta_score(
            y_true,
            y_pred,
            beta=0.5,
            zero_division=0,
        )
    )

    metrics["balanced_accuracy"] = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    # --------------------------------------------------------
    # Confusion-matrix metrics
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity_denominator = tn + fp

    if specificity_denominator > 0:
        metrics["specificity"] = float(
            tn / specificity_denominator
        )

    iou_denominator = tp + fp + fn

    if iou_denominator > 0:
        metrics["iou"] = float(
            tp / iou_denominator
        )
    else:
        metrics["iou"] = 0.0

    return metrics