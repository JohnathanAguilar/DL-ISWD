"""20x20-block detection evaluation, shared across every model / preprocessing variant.

Reproduces the evaluation done by hand, once per model/variant, in
``eval/all_model_block_evaluations.ipynb`` -- the same cell body was copied
8 times with only the input directories and majority threshold edited. This
script parameterizes that cell so each variant is one invocation instead of
one hand-edited copy.

For every valid 20x20 block, the ground-truth ``block_mask`` becomes a binary
majority-vote label and the prediction's ``internal_wave_bbox_confidence``
becomes a mean confidence score. A single confidence threshold is selected
once from the pooled VALIDATION set via Ultralytics' curve-based confidence
sweep (the max-F1 point of ``box.mp`` / ``box.mr``), then applied unchanged
to the TEST set for every metric below -- the pooled confusion matrix, every
per-file confusion matrix, and precision/recall/F1. The test set never
influences its own operating point.

Example
-------
    python evaluate_block_metrics.py \\
        --test-dir test_data/full_rgb --val-dir val_data/full_rgb \\
        --majority-threshold 0.25

    python evaluate_block_metrics.py \\
        --test-dir test_data/gaussian --val-dir val_data/gaussian \\
        --majority-threshold 0.50

Outputs land in ``<output-dir>/<name of --test-dir>/`` unless --per-file-csv
/ --overall-csv override them explicitly:
    block_metrics_per_file.csv   one row per input NetCDF file
    block_metrics_overall.csv    one row, pooled across every valid block
"""

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--test-dir", required=True, type=Path,
                        help="Directory of test-set NetCDF files (NETCDF_DIR).")
    parser.add_argument("--val-dir", required=True, type=Path,
                        help="Directory of validation-set NetCDF files, used only to "
                             "select the confidence threshold (VAL_NETCDF_DIR).")

    parser.add_argument(
        "--majority-threshold", required=True, type=float,
        help="Fraction of positive finite pixels a block needs to count as a "
             "positive ground-truth block. The notebook used 0.25 for the "
             "full-RGB-composite variants and 0.50 for the Gaussian-highpass "
             "variants -- there is no safe shared default, so this must be set "
             "explicitly.",
    )

    parser.add_argument("--ground-truth-var", default="block_mask")
    parser.add_argument("--prediction-var", default="internal_wave_bbox_confidence")
    parser.add_argument("--background-var", default="gaussian_filter_highpass")

    parser.add_argument("--block-height", type=int, default=20)
    parser.add_argument("--block-width", type=int, default=20)

    parser.add_argument(
        "--pixel-positive-threshold", type=float, default=0.50,
        help="Ground-truth block_mask pixel value at/above which a pixel counts "
             "as positive. The prediction is a continuous confidence, not a "
             "pixel mask, so this only applies to the ground truth.",
    )

    parser.add_argument(
        "--ignore-blocks-with-no-background-data", action="store_true",
        help="Drop blocks with no finite Gaussian high-pass pixel from evaluation.",
    )

    parser.add_argument(
        "--min-conf-threshold", type=float, default=0.05,
        help="Lower bound for the max-F1 confidence-threshold search. Keeps the "
             "selected operating point from collapsing to a degenerate near-0 "
             "threshold (predict every block positive) when a model's curve is "
             "flat or uninformative.",
    )
    parser.add_argument(
        "--max-conf-threshold", type=float, default=0.95,
        help="Upper bound for the max-F1 confidence-threshold search.",
    )

    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/test_results"),
        help="Base directory for auto-derived --per-file-csv / --overall-csv paths.",
    )
    parser.add_argument(
        "--per-file-csv", type=Path, default=None,
        help="Defaults to <output-dir>/<test-dir name>/block_metrics_per_file.csv.",
    )
    parser.add_argument(
        "--overall-csv", type=Path, default=None,
        help="Defaults to <output-dir>/<test-dir name>/block_metrics_overall.csv.",
    )

    return parser.parse_args(argv)


# ============================================================
# Array helpers
# ============================================================

def get_2d_array(data_array):
    """Convert an xarray DataArray into a squeezed 2D NumPy array."""
    array = np.asarray(data_array.values)
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(
            f"Expected {data_array.name!r} to be 2D after squeezing, but got shape {array.shape}."
        )

    return array


def crop_to_complete_blocks(array, block_height, block_width):
    """Crop a 2D array so its dimensions are exact multiples of the block dimensions.

    Incomplete blocks along the bottom and right edges are discarded.
    """
    array = np.asarray(array)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}.")

    height, width = array.shape

    cropped_height = (height // block_height) * block_height
    cropped_width = (width // block_width) * block_width

    if cropped_height == 0 or cropped_width == 0:
        raise ValueError(f"Array shape {array.shape} is too small for {block_height}x{block_width} blocks.")

    return array[:cropped_height, :cropped_width]


def binary_mask_to_majority_blocks(
    mask, block_height=20, block_width=20, pixel_positive_threshold=0.5, majority_threshold=0.5
):
    """Convert a pixel-level mask into a binary block grid.

    A block is positive when more than majority_threshold of its finite
    pixels are positive. With majority_threshold=0.5, an exact 50/50 tie
    becomes 0.

    Returns
    -------
    block_mask : np.ndarray
        Binary block-level mask.
    valid_blocks : np.ndarray
        Boolean mask identifying blocks with at least one finite pixel.
    positive_fraction : np.ndarray
        Fraction of finite positive pixels in every block. This is used as
        the continuous score for PR AUC and ROC AUC.
    """
    mask = np.asarray(mask, dtype=np.float32)
    mask = crop_to_complete_blocks(mask, block_height=block_height, block_width=block_width)

    height, width = mask.shape
    block_rows = height // block_height
    block_columns = width // block_width

    finite_pixels = np.isfinite(mask)
    positive_pixels = finite_pixels & (mask >= pixel_positive_threshold)

    finite_blocks = finite_pixels.reshape(block_rows, block_height, block_columns, block_width)
    positive_blocks = positive_pixels.reshape(block_rows, block_height, block_columns, block_width)

    finite_count = finite_blocks.sum(axis=(1, 3))
    positive_count = positive_blocks.sum(axis=(1, 3))

    valid_blocks = finite_count > 0

    positive_fraction = np.zeros((block_rows, block_columns), dtype=np.float32)
    positive_fraction[valid_blocks] = positive_count[valid_blocks] / finite_count[valid_blocks]

    block_mask = (positive_fraction > majority_threshold).astype(np.uint8)
    block_mask[~valid_blocks] = 0

    return block_mask, valid_blocks, positive_fraction


def background_to_valid_blocks(background, block_height=20, block_width=20):
    """Return a block mask indicating which blocks contain at least one finite background pixel."""
    background = np.asarray(background, dtype=np.float32)
    background = crop_to_complete_blocks(background, block_height=block_height, block_width=block_width)

    height, width = background.shape
    block_rows = height // block_height
    block_columns = width // block_width

    finite_pixels = np.isfinite(background)
    finite_blocks = finite_pixels.reshape(block_rows, block_height, block_columns, block_width)
    finite_count = finite_blocks.sum(axis=(1, 3))

    return finite_count > 0


def confidence_mask_to_average_blocks(confidence, block_height=20, block_width=20):
    """Convert a pixel-level YOLO confidence mask into a block-level average confidence grid.

    Each block's score is the mean of its finite confidence pixels (0.0
    outside every detection, up to 1.0 inside a detection; overlapping
    detections already use the maximum confidence per pixel upstream).

    Returns
    -------
    block_mean_confidence : np.ndarray
        Average YOLO confidence per block, in [0, 1].
    valid_blocks : np.ndarray
        Boolean mask identifying blocks with at least one finite pixel.
    """
    confidence = np.asarray(confidence, dtype=np.float32)
    confidence = crop_to_complete_blocks(confidence, block_height=block_height, block_width=block_width)

    height, width = confidence.shape
    block_rows = height // block_height
    block_columns = width // block_width

    finite_pixels = np.isfinite(confidence)
    confidence_filled = np.where(finite_pixels, confidence, 0.0)

    finite_blocks = finite_pixels.reshape(block_rows, block_height, block_columns, block_width)
    confidence_blocks = confidence_filled.reshape(block_rows, block_height, block_columns, block_width)

    finite_count = finite_blocks.sum(axis=(1, 3))
    confidence_sum = confidence_blocks.sum(axis=(1, 3))

    valid_blocks = finite_count > 0

    block_mean_confidence = np.zeros((block_rows, block_columns), dtype=np.float32)
    block_mean_confidence[valid_blocks] = confidence_sum[valid_blocks] / finite_count[valid_blocks]

    return block_mean_confidence, valid_blocks


# ============================================================
# Binary metrics
# ============================================================

def calculate_binary_metrics(y_true, y_prob, threshold):
    """Calculate binary classification metrics at one threshold."""
    y_true = np.asarray(y_true, dtype=bool).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32).reshape(-1)

    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true shape {y_true.shape} does not match y_prob shape {y_prob.shape}.")

    y_pred = y_prob >= threshold

    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())

    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    beta_squared = 0.5**2
    f0_5 = (
        (1.0 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        if (beta_squared * precision + recall) > 0
        else 0.0
    )

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    label_positive_rate = float(y_true.mean()) if total > 0 else 0.0
    pred_positive_rate = float(y_pred.mean()) if total > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_discovery_rate = fp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "f0_5": float(f0_5),
        "iou": float(iou),
        "balanced_accuracy": float(balanced_accuracy),
        "label_positive_rate": float(label_positive_rate),
        "pred_positive_rate": float(pred_positive_rate),
        "false_positive_rate": float(false_positive_rate),
        "false_discovery_rate": float(false_discovery_rate),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_blocks": total,
    }


# ============================================================
# PR AUC and ROC AUC
# ============================================================

def calculate_auc_metrics(y_true, y_score):
    """Calculate PR AUC and ROC AUC from continuous block scores.

    y_score is each predicted 20x20 block's mean YOLO confidence.
    """
    y_true = np.asarray(y_true, dtype=np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)

    if y_true.shape != y_score.shape:
        raise ValueError(f"y_true shape {y_true.shape} does not match y_score shape {y_score.shape}.")

    finite_mask = np.isfinite(y_score)
    y_true = y_true[finite_mask]
    y_score = y_score[finite_mask]

    if y_true.size == 0:
        return {"pr_auc": np.nan, "roc_auc": np.nan}

    unique_labels = np.unique(y_true)

    # PR AUC requires at least one positive label to be meaningful.
    if np.any(y_true == 1):
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall_curve[::-1], precision_curve[::-1])
    else:
        pr_auc = np.nan

    # ROC AUC requires both positive and negative labels.
    roc_auc = roc_auc_score(y_true, y_score) if unique_labels.size == 2 else np.nan

    return {"pr_auc": float(pr_auc), "roc_auc": float(roc_auc)}


# ============================================================
# Ultralytics-style curve-based precision/recall/F1/AP
# ============================================================
#
# This reproduces the exact algorithm Ultralytics uses for
# `results.box.mp` / `results.box.mr` (ultralytics/utils/metrics.py:
# `smooth`, `compute_ap`, `ap_per_class`, `Metric.mp`/`Metric.mr`), rather
# than fixing an evaluation threshold up front. Every block is treated as
# one "detection": its ground-truth label is the correctness signal (no IoU
# matching is needed since a block prediction can only correspond to one
# ground-truth block), so this is the single-class reduction of
# `ap_per_class` applied to each block's average YOLO confidence (from
# `confidence_mask_to_average_blocks`) as the score.

def ultralytics_smooth(y, f=0.05):
    """Box filter of fraction f. Matches ultralytics.utils.metrics.smooth."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ultralytics_compute_ap(recall, precision):
    """101-point COCO interpolation. Matches ultralytics.utils.metrics.compute_ap."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))

    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = _trapezoid(np.interp(x, mrec, mpre), x)

    return ap, mpre, mrec


def ultralytics_box_pr(y_true, y_prob, eps=1e-16, conf_min=0.0, conf_max=1.0):
    """Single-class reduction of ultralytics.utils.metrics.ap_per_class.

    Each block is one "detection" with confidence y_prob; tp[i] = 1 iff
    that block's ground-truth label is positive. Reproduces the
    sort -> cumulative TP/FP -> interpolate onto a 1000-pt confidence grid
    -> smooth(F1, 0.1) -> argmax pipeline that yields box.mp / box.mr /
    box.ap for a single-class YOLO model.

    The smoothed F1 curve is always computed over the full [0, 1] confidence
    grid (so smoothing isn't distorted by a truncated window, and AP -- which
    integrates over the full curve -- is unaffected), but the argmax that
    picks the operating point is restricted to [conf_min, conf_max]. This
    keeps the selected threshold from landing on a degenerate boundary value
    (e.g. 0.0, which predicts every block positive) when a model's curve is
    otherwise flat or uninformative.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    if y_true.sum() == 0:
        raise ValueError("Need at least one positive label to compute box.mp / box.mr.")

    order = np.argsort(-y_prob)  # sort by confidence, descending
    tp = y_true[order]
    conf = y_prob[order]

    n_l = tp.sum()  # number of ground-truth positives, i.e. nt[ci] in ap_per_class

    x = np.linspace(0, 1, 1000)

    fpc = (1 - tp).cumsum(0)
    tpc = tp.cumsum(0)

    recall = tpc / (n_l + eps)
    r_curve = np.interp(-x, -conf, recall, left=0)

    precision = tpc / (tpc + fpc)
    p_curve = np.interp(-x, -conf, precision, left=1)

    ap, mpre, mrec = ultralytics_compute_ap(recall, precision)

    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

    smoothed_f1 = ultralytics_smooth(f1_curve, 0.1)
    in_range = (x >= conf_min) & (x <= conf_max)
    if not in_range.any():
        raise ValueError(f"No confidence grid points fall inside [{conf_min}, {conf_max}].")
    i = np.where(in_range, smoothed_f1, -np.inf).argmax()  # max-F1 index within [conf_min, conf_max]

    return {
        "p": float(p_curve[i]),  # == box.mp for this single "class"
        "r": float(r_curve[i]),  # == box.mr
        "f1": float(f1_curve[i]),
        "ap": float(ap),
        "conf_at_max_f1": float(x[i]),
        "x": x,
        "p_curve": p_curve,
        "r_curve": r_curve,
        "f1_curve": f1_curve,
    }


def box_pr_at_confidence(box_pr, confidence):
    """Read precision/recall/F1 off an already-computed `ultralytics_box_pr` curve at an
    arbitrary confidence, instead of at that curve's own max-F1 point. Used to score the
    test-set curve at the confidence selected from the validation set, exactly like
    Ultralytics reads box.mp / box.mr off its confidence grid at a chosen index -- just
    with that index chosen by the validation set rather than by argmax on the same data
    being scored.
    """
    i = int(np.argmin(np.abs(box_pr["x"] - confidence)))

    return {
        "p": float(box_pr["p_curve"][i]),
        "r": float(box_pr["r_curve"][i]),
        "f1": float(box_pr["f1_curve"][i]),
    }


# ============================================================
# Load every NetCDF file in a directory and compute its block-level
# ground-truth labels and prediction scores.
# ============================================================

def load_blocks_from_directory(nc_files, dataset_label, args):
    """Run the full block-extraction pipeline over every file in nc_files.

    Returns per-file records (needed for per-file confusion matrices),
    failed rows, and the pooled labels/scores across every valid block in
    every successfully-loaded file.
    """
    file_records = []
    failed_rows = []

    all_labels = []
    all_scores = []

    successful_files = 0
    failed_files = 0

    for file_index, nc_path in enumerate(nc_files, start=1):
        print("\n" + "-" * 80)
        print(f"[{dataset_label}] [{file_index}/{len(nc_files)}] {nc_path.name}")

        try:
            with xr.open_dataset(nc_path) as ds:
                required_variables = [args.ground_truth_var, args.prediction_var, args.background_var]
                missing_variables = [v for v in required_variables if v not in ds]

                if missing_variables:
                    raise KeyError(f"Missing variables: {missing_variables}")

                ground_truth_pixels = get_2d_array(ds[args.ground_truth_var]).astype(np.float32)
                prediction_confidence_pixels = get_2d_array(ds[args.prediction_var]).astype(np.float32)
                background = get_2d_array(ds[args.background_var]).astype(np.float32)

            # --------------------------------------------------------
            # Validate pixel-level alignment
            # --------------------------------------------------------
            if ground_truth_pixels.shape != prediction_confidence_pixels.shape:
                raise ValueError(
                    "Mask shapes do not match.\n"
                    f"{args.ground_truth_var}: {ground_truth_pixels.shape}\n"
                    f"{args.prediction_var}: {prediction_confidence_pixels.shape}"
                )

            if background.shape != ground_truth_pixels.shape:
                raise ValueError(
                    "Background shape does not match mask shape.\n"
                    f"{args.background_var}: {background.shape}\n"
                    f"Masks: {ground_truth_pixels.shape}"
                )

            # --------------------------------------------------------
            # Convert ground truth into majority-vote blocks and the
            # prediction into average-confidence blocks
            # --------------------------------------------------------
            ground_truth_blocks, ground_truth_valid_blocks, ground_truth_positive_fraction = (
                binary_mask_to_majority_blocks(
                    mask=ground_truth_pixels,
                    block_height=args.block_height,
                    block_width=args.block_width,
                    pixel_positive_threshold=args.pixel_positive_threshold,
                    majority_threshold=args.majority_threshold,
                )
            )

            prediction_block_confidence, prediction_valid_blocks = confidence_mask_to_average_blocks(
                confidence=prediction_confidence_pixels,
                block_height=args.block_height,
                block_width=args.block_width,
            )

            background_valid_blocks = background_to_valid_blocks(
                background=background, block_height=args.block_height, block_width=args.block_width
            )

            # --------------------------------------------------------
            # Build valid evaluation mask
            # --------------------------------------------------------
            valid_blocks = ground_truth_valid_blocks & prediction_valid_blocks

            if args.ignore_blocks_with_no_background_data:
                valid_blocks &= background_valid_blocks

            if not valid_blocks.any():
                raise ValueError("No valid 20x20 blocks were available.")

            # --------------------------------------------------------
            # Flatten valid blocks
            # --------------------------------------------------------
            y_true = ground_truth_blocks[valid_blocks].astype(np.uint8)

            # Each block's prediction score is its mean YOLO confidence.
            y_score = prediction_block_confidence[valid_blocks].astype(np.float32)

            original_height, original_width = ground_truth_pixels.shape
            cropped_height = (original_height // args.block_height) * args.block_height
            cropped_width = (original_width // args.block_width) * args.block_width

            file_records.append(
                {
                    "file": nc_path.name,
                    "original_height": original_height,
                    "original_width": original_width,
                    "cropped_height": cropped_height,
                    "cropped_width": cropped_width,
                    "block_rows": ground_truth_blocks.shape[0],
                    "block_columns": ground_truth_blocks.shape[1],
                    "y_true": y_true,
                    "y_score": y_score,
                }
            )

            # Add this file's valid blocks to the pooled set.
            all_labels.append(y_true)
            all_scores.append(y_score)

            successful_files += 1

            print(f"Valid blocks: {y_true.size} | Ground-truth positive: {int(y_true.sum())}")

        except Exception as error:
            failed_files += 1

            print(f"FAILED: {error}")
            traceback.print_exc()

            failed_rows.append(
                {
                    "file": nc_path.name,
                    "status": "failed",
                    "original_height": np.nan,
                    "original_width": np.nan,
                    "cropped_height": np.nan,
                    "cropped_width": np.nan,
                    "block_rows": np.nan,
                    "block_columns": np.nan,
                    "valid_blocks": 0,
                    "ground_truth_positive_blocks": 0,
                    "predicted_positive_blocks": 0,
                    "threshold": np.nan,
                    "accuracy": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "specificity": np.nan,
                    "f1": np.nan,
                    "f0_5": np.nan,
                    "iou": np.nan,
                    "balanced_accuracy": np.nan,
                    "label_positive_rate": np.nan,
                    "pred_positive_rate": np.nan,
                    "false_positive_rate": np.nan,
                    "false_discovery_rate": np.nan,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "tp": 0,
                    "fp": 0,
                    "tn": 0,
                    "fn": 0,
                    "total_blocks": 0,
                    "error": str(error),
                }
            )

    if not all_labels:
        raise ValueError(f"No files were successfully evaluated for the {dataset_label} set.")

    pooled_labels = np.concatenate(all_labels)
    pooled_scores = np.concatenate(all_scores)

    return {
        "file_records": file_records,
        "failed_rows": failed_rows,
        "labels": pooled_labels,
        "scores": pooled_scores,
        "successful_files": successful_files,
        "failed_files": failed_files,
    }


def main(argv=None):
    args = parse_args(argv)

    if args.per_file_csv is None:
        args.per_file_csv = args.output_dir / args.test_dir.name / "block_metrics_per_file.csv"
    if args.overall_csv is None:
        args.overall_csv = args.output_dir / args.test_dir.name / "block_metrics_overall.csv"

    # ============================================================
    # Validate input directories
    # ============================================================
    if not args.val_dir.exists():
        raise FileNotFoundError(f"Validation NetCDF directory does not exist:\n{args.val_dir}")

    val_nc_files = sorted(args.val_dir.glob("*.nc"))
    if not val_nc_files:
        raise FileNotFoundError(f"No .nc files were found in:\n{args.val_dir}")

    if not args.test_dir.exists():
        raise FileNotFoundError(f"NetCDF directory does not exist:\n{args.test_dir}")

    nc_files = sorted(args.test_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files were found in:\n{args.test_dir}")

    print("=" * 80)
    print("20x20 BLOCK-MASK TEST-SET EVALUATION")
    print("=" * 80)
    print(f"Validation NetCDF directory: {args.val_dir}")
    print(f"Validation NetCDF files found: {len(val_nc_files)}")
    print(f"Test NetCDF directory: {args.test_dir}")
    print(f"Test NetCDF files found: {len(nc_files)}")
    print(f"Block size: {args.block_height}x{args.block_width}")
    print(f"Ground-truth block rule: positive when more than {args.majority_threshold:.0%} of finite pixels are positive")
    print(
        "Prediction block score: mean YOLO confidence "
        f"({args.prediction_var}) over the block's finite pixels"
    )
    print(
        "Confusion-matrix threshold: selected once from the pooled VALIDATION "
        "set via Ultralytics' curve-based confidence sweep, then applied "
        "unchanged to the test set for every result below"
    )

    # ============================================================
    # Pass 1: load the validation set and the test set. Per-file
    # confusion-matrix metrics for the test set are not computed yet --
    # they need the shared threshold selected below from the validation set.
    # ============================================================
    val_data = load_blocks_from_directory(val_nc_files, "validation", args)

    val_labels = val_data["labels"]
    val_scores = val_data["scores"]

    test_data_blocks = load_blocks_from_directory(nc_files, "test", args)

    file_records = test_data_blocks["file_records"]
    failed_rows = test_data_blocks["failed_rows"]
    test_labels = test_data_blocks["labels"]
    test_scores = test_data_blocks["scores"]
    successful_files = test_data_blocks["successful_files"]
    failed_files = test_data_blocks["failed_files"]

    # ============================================================
    # Select the shared confidence threshold from the pooled VALIDATION set,
    # then score the pooled TEST set's curve at that threshold.
    # ============================================================
    val_box_pr = ultralytics_box_pr(
        y_true=val_labels, y_prob=val_scores,
        conf_min=args.min_conf_threshold, conf_max=args.max_conf_threshold,
    )
    selected_threshold = val_box_pr["conf_at_max_f1"]

    print("\n" + "=" * 80)
    print("VALIDATION-SELECTED CONFIDENCE THRESHOLD")
    print("=" * 80)
    print(f"Confidence @ max smoothed F1 (pooled validation set): {selected_threshold:.6f}")
    print(
        "This threshold is now applied to the test set for every result "
        "below, including every per-file confusion matrix -- the test set "
        "never influences its own operating point."
    )

    test_box_pr = ultralytics_box_pr(
        y_true=test_labels, y_prob=test_scores,
        conf_min=args.min_conf_threshold, conf_max=args.max_conf_threshold,
    )
    test_at_val_threshold = box_pr_at_confidence(test_box_pr, selected_threshold)

    # Confusion-matrix diagnostics (accuracy, specificity, IoU, TP/FP/TN/FN, ...)
    # at the confidence threshold the validation-set curve-sweep selected.
    overall_metrics = calculate_binary_metrics(y_true=test_labels, y_prob=test_scores, threshold=selected_threshold)

    # Ultralytics reads precision/recall/F1 directly off the interpolated,
    # smoothed curve at that confidence -- these replace the raw >=threshold
    # values above so they match box.mp / box.mr exactly. AP integrates over
    # every threshold on the test curve, so it is unaffected by which single
    # confidence was selected.
    overall_metrics["precision"] = test_at_val_threshold["p"]
    overall_metrics["recall"] = test_at_val_threshold["r"]
    overall_metrics["f1"] = test_at_val_threshold["f1"]
    overall_metrics["ap"] = test_box_pr["ap"]

    overall_auc_metrics = calculate_auc_metrics(y_true=test_labels, y_score=test_scores)

    overall_results = pd.DataFrame(
        [
            {
                "val_set_directory": str(args.val_dir),
                "val_files_found": len(val_nc_files),
                "val_valid_blocks": int(val_labels.size),
                "val_ground_truth_positive_blocks": int(val_labels.sum()),
                "test_set_directory": str(args.test_dir),
                "files_found": len(nc_files),
                "successful_files": successful_files,
                "failed_files": failed_files,
                "block_height": args.block_height,
                "block_width": args.block_width,
                "majority_threshold": args.majority_threshold,
                "valid_blocks": int(test_labels.size),
                "ground_truth_positive_blocks": int(test_labels.sum()),
                "predicted_positive_blocks": int(overall_metrics["tp"] + overall_metrics["fp"]),
                **overall_metrics,
                **overall_auc_metrics,
            }
        ]
    )

    args.overall_csv.parent.mkdir(parents=True, exist_ok=True)
    overall_results.to_csv(args.overall_csv, index=False)

    # ============================================================
    # Pass 2: compute per-file confusion-matrix metrics at the shared
    # selected_threshold, now that it is known.
    # ============================================================
    per_file_rows = []

    for record in file_records:
        y_true = record["y_true"]
        y_score = record["y_score"]

        metrics = calculate_binary_metrics(y_true=y_true, y_prob=y_score, threshold=selected_threshold)
        auc_metrics = calculate_auc_metrics(y_true=y_true, y_score=y_score)

        per_file_rows.append(
            {
                "file": record["file"],
                "status": "success",
                "original_height": record["original_height"],
                "original_width": record["original_width"],
                "cropped_height": record["cropped_height"],
                "cropped_width": record["cropped_width"],
                "block_rows": record["block_rows"],
                "block_columns": record["block_columns"],
                "valid_blocks": int(y_true.size),
                "ground_truth_positive_blocks": int(y_true.sum()),
                "predicted_positive_blocks": int(metrics["tp"] + metrics["fp"]),
                **metrics,
                **auc_metrics,
                "error": "",
            }
        )

    per_file_rows.extend(failed_rows)

    # Restore the original NETCDF_DIR listing order.
    per_file_rows.sort(key=lambda row: row["file"])

    per_file_results = pd.DataFrame(per_file_rows)

    args.per_file_csv.parent.mkdir(parents=True, exist_ok=True)
    per_file_results.to_csv(args.per_file_csv, index=False)

    # ============================================================
    # Print per-file results
    # ============================================================
    display_columns = [
        "file", "status", "valid_blocks", "ground_truth_positive_blocks", "predicted_positive_blocks",
        "accuracy", "precision", "recall", "specificity", "f1", "f0_5", "iou", "pr_auc", "roc_auc",
        "balanced_accuracy", "tp", "fp", "tn", "fn",
    ]

    print("\n" + "=" * 80)
    print("PER-FILE RESULTS")
    print("=" * 80)
    print(per_file_results[display_columns].to_string(index=False))

    # ============================================================
    # Print overall pooled test-set results
    # ============================================================
    print("\n" + "=" * 80)
    print("OVERALL TEST-SET RESULTS")
    print("=" * 80)

    print(f"Validation files found: {len(val_nc_files)}")
    print(f"Validation valid blocks: {val_labels.size}")
    print()

    print(f"Files found:           {len(nc_files)}")
    print(f"Successful files:      {successful_files}")
    print(f"Failed files:          {failed_files}")
    print(f"Valid 20x20 blocks:    {test_labels.size}")
    print(f"Ground-truth positive: {int(test_labels.sum())}")
    print(f"Predicted positive (at validation-selected confidence): {overall_metrics['tp'] + overall_metrics['fp']}")
    print()

    print(
        "Precision / Recall / F1 below come from Ultralytics' curve-based "
        "confidence sweep (box.mp / box.mr), read off the TEST-set curve at "
        "the confidence selected from the VALIDATION set. AP integrates the "
        "TEST-set curve over every threshold, so it needs no selected "
        "confidence."
    )
    print(f"Confidence (selected on validation, applied to test): {overall_metrics['threshold']:.6f}")
    print()

    print(f"Accuracy:              {overall_metrics['accuracy']:.6f}")
    print(f"Precision (box.mp):    {overall_metrics['precision']:.6f}")
    print(f"Recall (box.mr):       {overall_metrics['recall']:.6f}")
    print(f"Specificity:           {overall_metrics['specificity']:.6f}")
    print(f"F1:                    {overall_metrics['f1']:.6f}")
    print(f"AP (box.ap):           {overall_metrics['ap']:.6f}")
    print(f"F0.5 (at selected conf): {overall_metrics['f0_5']:.6f}")
    print(f"IoU:                   {overall_metrics['iou']:.6f}")
    print(f"PR AUC:                {overall_auc_metrics['pr_auc']:.6f}")
    print(f"ROC AUC:               {overall_auc_metrics['roc_auc']:.6f}")
    print(f"Balanced accuracy:     {overall_metrics['balanced_accuracy']:.6f}")
    print(f"False-positive rate:   {overall_metrics['false_positive_rate']:.6f}")
    print(f"False-discovery rate:  {overall_metrics['false_discovery_rate']:.6f}")
    print(f"Label positive rate:   {overall_metrics['label_positive_rate']:.6f}")
    print(f"Pred positive rate:    {overall_metrics['pred_positive_rate']:.6f}")
    print()

    print(f"TP: {overall_metrics['tp']}")
    print(f"FP: {overall_metrics['fp']}")
    print(f"TN: {overall_metrics['tn']}")
    print(f"FN: {overall_metrics['fn']}")
    print()

    print(f"Per-file CSV: {args.per_file_csv}")
    print(f"Overall CSV:  {args.overall_csv}")

    return overall_results


if __name__ == "__main__":
    main()
