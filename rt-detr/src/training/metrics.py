"""Detection metrics matching the YOLO evaluation protocol.

Two families of numbers are produced:

1. Fixed-confidence-threshold TP/FP/FN counts, from which precision, recall and
   F-beta are computed by hand. Matching is class-aware, greedy by descending
   score, and uses a single IoU threshold (0.50 by default).
2. Curve-based AP: AP@0.50 and AP@[0.50:0.05:0.95] with 101-point
   interpolation, i.e. the COCO definition that Ultralytics' mAP50 /
   mAP50-95 approximate.
"""

import numpy as np
import torch
from torchvision.ops import box_iou

IOU_SWEEP = np.round(np.arange(0.50, 0.96, 0.05), 2)


def filter_by_confidence(predictions, conf):
    """Drop predictions below ``conf`` from a cached prediction list."""
    filtered = []

    for pred in predictions:
        keep = pred["scores"] >= conf
        filtered.append(
            {
                "boxes": pred["boxes"][keep],
                "scores": pred["scores"][keep],
                "labels": pred["labels"][keep],
            }
        )

    return filtered


def count_tp_fp_fn(predictions, targets, iou_thr=0.5):
    """Greedy, class-aware matching at a fixed confidence threshold.

    ``predictions`` must already be confidence-filtered.
    """
    tp = 0
    fp = 0
    fn = 0

    for pred, gt in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]

        if len(pred_boxes) == 0:
            fn += len(gt_boxes)
            continue

        if len(gt_boxes) == 0:
            fp += len(pred_boxes)
            continue

        order = torch.argsort(pred_scores, descending=True)
        ious = box_iou(pred_boxes[order], gt_boxes)
        same_class = pred_labels[order].unsqueeze(1) == gt_labels.unsqueeze(0)
        ious = torch.where(same_class, ious, torch.zeros_like(ious))

        matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for i in range(len(order)):
            row = ious[i].clone()
            row[matched] = -1.0

            best_iou, best_j = row.max(dim=0)

            if best_iou >= iou_thr:
                matched[best_j] = True
                tp += 1
            else:
                fp += 1

        fn += int((~matched).sum())

    return tp, fp, fn


def precision_recall_fbeta(tp, fp, fn, beta=1.0):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    denom = (beta**2 * precision) + recall
    if denom <= 0:
        fbeta = 0.0
    else:
        fbeta = (1 + beta**2) * precision * recall / denom

    return precision, recall, fbeta


def fbeta_key(beta):
    """Metric key used in the saved JSON: ``f1`` for beta=1, ``f0.5`` for beta=0.5."""
    return f"f{beta:g}"


def fixed_threshold_metrics(predictions, targets, conf, iou_thr=0.5, beta=1.0):
    """Full metric record at one confidence threshold."""
    filtered = filter_by_confidence(predictions, conf)
    tp, fp, fn = count_tp_fp_fn(filtered, targets, iou_thr=iou_thr)
    precision, recall, fbeta = precision_recall_fbeta(tp, fp, fn, beta=beta)

    return {
        "conf": float(conf),
        "eval_iou": float(iou_thr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        fbeta_key(beta): float(fbeta),
    }


def average_precision(predictions, targets, iou_thr=0.5, class_id=None):
    """101-point interpolated AP for a single class (or all classes pooled)."""
    n_positives = 0
    scored = []  # (score, image_index, prediction_index)
    iou_cache = {}

    for img_idx, (pred, gt) in enumerate(zip(predictions, targets)):
        gt_boxes = gt["boxes"]
        gt_labels = gt["labels"]

        if class_id is not None and len(gt_boxes):
            gt_keep = gt_labels == class_id
            gt_boxes = gt_boxes[gt_keep]

        n_positives += len(gt_boxes)

        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        if class_id is not None and len(pred_boxes):
            pred_keep = pred["labels"] == class_id
            pred_boxes = pred_boxes[pred_keep]
            pred_scores = pred_scores[pred_keep]

        if len(pred_boxes) == 0:
            iou_cache[img_idx] = (torch.zeros((0, len(gt_boxes))), len(gt_boxes))
            continue

        if len(gt_boxes) == 0:
            ious = torch.zeros((len(pred_boxes), 0))
        else:
            ious = box_iou(pred_boxes, gt_boxes)

        iou_cache[img_idx] = (ious, len(gt_boxes))

        for pred_idx in range(len(pred_boxes)):
            scored.append((float(pred_scores[pred_idx]), img_idx, pred_idx))

    if n_positives == 0:
        return 0.0

    if not scored:
        return 0.0

    scored.sort(key=lambda item: -item[0])

    matched = {
        img_idx: torch.zeros(n_gt, dtype=torch.bool)
        for img_idx, (_, n_gt) in iou_cache.items()
    }

    tps = np.zeros(len(scored), dtype=np.float64)
    fps = np.zeros(len(scored), dtype=np.float64)

    for rank, (_, img_idx, pred_idx) in enumerate(scored):
        ious, n_gt = iou_cache[img_idx]

        if n_gt == 0:
            fps[rank] = 1
            continue

        row = ious[pred_idx].clone()
        row[matched[img_idx]] = -1.0

        best_iou, best_j = row.max(dim=0)

        if best_iou >= iou_thr:
            matched[img_idx][best_j] = True
            tps[rank] = 1
        else:
            fps[rank] = 1

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)

    recalls = tp_cum / n_positives
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)

    # Make the precision envelope monotonically decreasing, then sample it at
    # 101 evenly spaced recall levels (COCO convention).
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    recall_levels = np.linspace(0, 1, 101)
    indices = np.searchsorted(recalls, recall_levels, side="left")

    sampled = np.zeros_like(recall_levels)
    valid = indices < len(precisions)
    sampled[valid] = precisions[indices[valid]]

    return float(sampled.mean())


def ap_metrics(predictions, targets, class_ids=(1,)):
    """mAP@0.50 and mAP@[0.50:0.95], averaged over classes."""
    ap50_per_class = []
    ap_sweep_per_class = []

    for class_id in class_ids:
        aps = [
            average_precision(predictions, targets, iou_thr=thr, class_id=class_id)
            for thr in IOU_SWEEP
        ]
        ap50_per_class.append(aps[0])
        ap_sweep_per_class.append(float(np.mean(aps)))

    return {
        "mAP50": float(np.mean(ap50_per_class)),
        "mAP50-95": float(np.mean(ap_sweep_per_class)),
        "per_class_AP50": [float(v) for v in ap50_per_class],
        "per_class_AP50-95": [float(v) for v in ap_sweep_per_class],
    }
