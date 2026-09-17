from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


try:
    # Typical layout when running from src/:
    #   src/inference/inference.py
    #   src/evaluation/eval.py
    from inference.inference import load_model
except ImportError:
    # Useful if imported as a package from project root.
    from src.inference.inference import load_model  # type: ignore


try:
    from evaluation.viz import (
        plot_auc_bar,
        plot_auc_curves,
        plot_confusion_matrix,
        plot_metric_bar,
        plot_probability_histogram,
        plot_sample_result,
    )
except ImportError:
    try:
        from src.evaluation.viz import (  # type: ignore
            plot_auc_bar,
            plot_auc_curves,
            plot_confusion_matrix,
            plot_metric_bar,
            plot_probability_histogram,
            plot_sample_result,
        )
    except ImportError:
        # Visualization imports are optional. Evaluation can still run without them.
        plot_auc_bar = None
        plot_auc_curves = None
        plot_confusion_matrix = None
        plot_metric_bar = None
        plot_probability_histogram = None
        plot_sample_result = None


TensorLike = Union[torch.Tensor, np.ndarray]


@dataclass
class BinaryCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        y_true = y_true.astype(bool, copy=False)
        y_pred = y_pred.astype(bool, copy=False)

        self.tp += int(np.logical_and(y_true, y_pred).sum())
        self.fp += int(np.logical_and(~y_true, y_pred).sum())
        self.tn += int(np.logical_and(~y_true, ~y_pred).sum())
        self.fn += int(np.logical_and(y_true, ~y_pred).sum())

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def _binary_auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute binary ROC-AUC without sklearn.

    Returns NaN if AUC is undefined, e.g. only one class is present.
    """
    y_true = y_true.astype(bool, copy=False)
    y_score = y_score.astype(np.float64, copy=False)

    if y_true.size == 0:
        return float("nan")

    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Handle tied scores by assigning average ranks.
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1

        if j - i > 1:
            avg_rank = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg_rank

        i = j

    sum_ranks_pos = float(ranks[y_true].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc)


def _binary_auc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """
    Compute binary PR-AUC / average precision without sklearn.

    Returns NaN if there are no positive labels.
    """
    y_true = y_true.astype(bool, copy=False)
    y_score = y_score.astype(np.float64, copy=False)

    if y_true.size == 0:
        return float("nan")

    n_pos = int(y_true.sum())

    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp_cum = np.cumsum(y_sorted).astype(np.float64)
    fp_cum = np.cumsum(~y_sorted).astype(np.float64)

    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1.0)

    # Average precision: mean precision at each true positive.
    ap = float(precision[y_sorted].sum() / n_pos)

    return ap


def _metrics_from_counts(
    counts: BinaryCounts,
    threshold: float,
    loss_sum: float = 0.0,
    loss_count: int = 0,
) -> Dict[str, float]:
    tp = counts.tp
    fp = counts.fp
    tn = counts.tn
    fn = counts.fn
    total = counts.total

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    # F-beta score with beta=0.5. This weights precision more heavily
    # than recall: beta^2 = 0.25.
    beta = 0.5
    beta_sq = beta ** 2
    f0_5 = _safe_div(
        (1.0 + beta_sq) * precision * recall,
        beta_sq * precision + recall,
    )

    iou = _safe_div(tp, tp + fp + fn)
    balanced_accuracy = 0.5 * (recall + specificity)

    label_positive_rate = _safe_div(tp + fn, total)
    pred_positive_rate = _safe_div(tp + fp, total)

    metrics: Dict[str, float] = {
        "threshold": float(threshold),
        "total_valid_blocks": float(total),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f0_5": f0_5,
        "iou": iou,
        "balanced_accuracy": balanced_accuracy,
        "label_positive_rate": label_positive_rate,
        "pred_positive_rate": pred_positive_rate,
    }

    if loss_count > 0:
        metrics["bce_loss"] = float(loss_sum / loss_count)

    return metrics


def _as_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _move_to_device(
    value: Any,
    device: torch.device,
) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return {key: _move_to_device(value, device) for key, value in batch.items()}


def _get_batch_block_mask(batch: Mapping[str, Any]) -> Optional[torch.Tensor]:
    """
    Get a block mask from the batch.

    Preferred key:
        "block_attn_mask"

    Fallback key:
        "valid"

    The model should treat True values as real/valid blocks.
    """
    block_mask = batch.get("block_attn_mask", None)

    if block_mask is None:
        block_mask = batch.get("valid", None)

    if block_mask is None:
        return None

    if not isinstance(block_mask, torch.Tensor):
        block_mask = torch.as_tensor(block_mask)

    return block_mask.bool()


def _flatten_batch_grid(x: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, Bh, Bw] -> [B, Bh*Bw].

    If already [B, N], keep as-is.
    """
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected tensor with shape [B, Bh, Bw] or [B, N], got {tuple(x.shape)}")


def _get_probs_from_model_output(out: Mapping[str, Any]) -> torch.Tensor:
    """
    Pull probabilities from a model output dict.

    Expected:
        out["probs"]

    Supported shapes:
        [B, N]
        [B, Bh, Bw]
    """
    if "probs" not in out:
        raise KeyError(
            "Model output must contain key 'probs'. "
            f"Available keys: {list(out.keys())}"
        )

    probs = out["probs"]

    if not isinstance(probs, torch.Tensor):
        probs = torch.as_tensor(probs)

    if probs.ndim == 3:
        probs = probs.reshape(probs.shape[0], -1)
    elif probs.ndim != 2:
        raise ValueError(
            f"Expected out['probs'] to have shape [B, N] or [B, Bh, Bw], "
            f"got {tuple(probs.shape)}"
        )

    return probs


def _binary_cross_entropy_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-7,
) -> Tuple[float, int]:
    """
    Compute BCE from probabilities on valid tokens only.

    Returns:
        loss_sum:
            Sum of BCE over valid tokens.
        count:
            Number of valid tokens.
    """
    probs = probs.float().clamp(eps, 1.0 - eps)
    labels = labels.float()
    valid_mask = valid_mask.bool()

    if valid_mask.sum().item() == 0:
        return 0.0, 0

    bce = -(labels * torch.log(probs) + (1.0 - labels) * torch.log(1.0 - probs))
    bce_valid = bce[valid_mask]

    return float(bce_valid.sum().detach().cpu().item()), int(bce_valid.numel())


def _infer_in_channels_from_loader(test_loader: Iterable[Any]) -> Optional[int]:
    """
    Try to infer the number of image channels from test_loader.dataset[0]["image"].

    This lets evaluation automatically build a 1-channel, 2-channel, or 3-channel
    model config when the checkpoint/config does not already specify in_channels.
    """
    dataset = getattr(test_loader, "dataset", None)
    if dataset is None:
        return None

    try:
        if len(dataset) == 0:
            return None

        first_item = dataset[0]
        image = first_item.get("image", None)

        if isinstance(image, np.ndarray) and image.ndim == 3:
            return int(image.shape[0])

        if isinstance(image, torch.Tensor) and image.ndim == 3:
            return int(image.shape[0])

    except Exception:
        return None

    return None


def _grid_shape_for_sample(
    batch: Mapping[str, Any],
    sample_index: int,
    fallback_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Get the real grid shape for one sample.

    The collator stores:
        batch["grid_shape"] = tensor [B, 2]

    If not present, use labels/valid fallback shape.
    """
    grid_shape = batch.get("grid_shape", None)

    if grid_shape is None:
        return fallback_shape

    if isinstance(grid_shape, torch.Tensor):
        Bh = int(grid_shape[sample_index, 0].detach().cpu().item())
        Bw = int(grid_shape[sample_index, 1].detach().cpu().item())
        return Bh, Bw

    grid_shape_arr = np.asarray(grid_shape)
    Bh = int(grid_shape_arr[sample_index, 0])
    Bw = int(grid_shape_arr[sample_index, 1])
    return Bh, Bw


def _ids_from_batch(batch: Mapping[str, Any], batch_size: int) -> List[str]:
    ids = batch.get("ids", None)

    if ids is None:
        return [f"sample_{i}" for i in range(batch_size)]

    return [str(x) for x in ids]


def _sample_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    counts = BinaryCounts()
    counts.update(y_true=y_true, y_pred=y_pred)

    loss_sum = 0.0
    loss_count = 0

    if y_true.size > 0:
        eps = 1e-7
        p = np.clip(y_prob.astype(np.float32), eps, 1.0 - eps)
        y = y_true.astype(np.float32)
        bce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        loss_sum = float(np.sum(bce))
        loss_count = int(bce.size)

    metrics = _metrics_from_counts(
        counts,
        threshold=threshold,
        loss_sum=loss_sum,
        loss_count=loss_count,
    )

    metrics["roc_auc"] = _binary_auc_roc(y_true=y_true, y_score=y_prob)
    metrics["pr_auc"] = _binary_auc_pr(y_true=y_true, y_score=y_prob)

    return metrics


def _write_metrics_json(
    metrics: Mapping[str, Any],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, allow_nan=True)


def _write_sample_metrics_csv(
    sample_results: Sequence[Mapping[str, Any]],
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metric_keys = [
        "id",
        "total_valid_blocks",
        "tp",
        "fp",
        "tn",
        "fn",
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
        "bce_loss",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()

        for result in sample_results:
            row = {"id": result["id"]}
            metrics = result.get("metrics", {})

            for key in metric_keys:
                if key == "id":
                    continue
                row[key] = metrics.get(key, "")

            writer.writerow(row)


def _save_sample_npz_files(
    sample_results: Sequence[Mapping[str, Any]],
    save_dir: str,
) -> None:
    sample_dir = os.path.join(save_dir, "samples_npz")
    os.makedirs(sample_dir, exist_ok=True)

    for idx, result in enumerate(sample_results):
        safe_id = str(result["id"]).replace("/", "_").replace("\\", "_")
        out_path = os.path.join(sample_dir, f"{idx:05d}_{safe_id}.npz")

        np.savez_compressed(
            out_path,
            probs=np.asarray(result["probs"], dtype=np.float32),
            pred=np.asarray(result["pred"], dtype=np.bool_),
            labels=np.asarray(result["labels"], dtype=np.uint8),
            valid=np.asarray(result["valid"], dtype=np.bool_),
        )


def evaluate_test_loader(
    model_ckpt: str,
    device: Union[str, torch.device],
    test_loader: Iterable[Mapping[str, Any]],
    block_size=None,
    threshold: float = 0.5,
    model_cfg_kwargs: Optional[dict[str, Any]] = None,
    strict: bool = True,
    save_dir: Optional[str] = None,
    make_visualizations: bool = True,
    max_visualized_samples: int = 8,
    save_sample_arrays: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained BlocksTokenModel on a test_loader.

    This function uses load_model(...) from inference.py, but evaluates directly
    from the already-built test_loader.

    Expected batch keys from your collator:
        images: torch.FloatTensor [B, C, H, W]
        labels: torch.Tensor [B, Bh, Bw]
        valid: torch.BoolTensor [B, Bh, Bw]
        block_attn_mask: optional torch.BoolTensor [B, Bh, Bw] or [B, Bh*Bw]
        ids: list[str]
        grid_shape: torch.LongTensor [B, 2]

    Args:
        model_ckpt:
            Path to checkpoint, for example ".../best.pt".
        device:
            torch.device or string: "cuda", "mps", or "cpu".
        test_loader:
            DataLoader returned by your loader code.
        block_size:
            Optional block size override. If None, load_model uses checkpoint config.
        threshold:
            Probability threshold for binary predictions.
        model_cfg_kwargs:
            Optional ModelConfig overrides. If "in_channels" is not provided, this
            function tries to infer it from test_loader.dataset[0]["image"].
        strict:
            Passed to model.load_state_dict inside inference.py.
        save_dir:
            If provided, writes metrics, CSV, sample arrays, and visualizations.
        make_visualizations:
            If True and save_dir is provided, writes PNG plots using viz.py.
        max_visualized_samples:
            Number of per-sample visualization PNGs to save.
        save_sample_arrays:
            If True and save_dir is provided, saves per-sample .npz arrays.

    Returns:
        result dict with:
            metrics:
                Overall metrics across all valid blocks, including roc_auc and pr_auc.
            sample_results:
                Per-sample predictions, probabilities, labels, valid mask, and metrics.
            cfg:
                ModelConfig object returned by load_model.
    """
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    device = _as_device(device)

    cfg_kwargs = dict(model_cfg_kwargs or {})
    inferred_in_channels = _infer_in_channels_from_loader(test_loader)

    if inferred_in_channels is not None:
        cfg_kwargs.setdefault("in_channels", inferred_in_channels)

    model, cfg = load_model(
        model_ckpt=model_ckpt,
        device=device,
        block_size=block_size,
        model_cfg_kwargs=cfg_kwargs,
        strict=strict,
    )

    model.eval()

    overall_counts = BinaryCounts()
    overall_loss_sum = 0.0
    overall_loss_count = 0

    all_y_true_parts: List[np.ndarray] = []
    all_y_prob_parts: List[np.ndarray] = []

    sample_results: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(test_loader):
            batch = _move_batch_to_device(batch_cpu, device)

            if "images" not in batch:
                raise KeyError("Batch is missing required key 'images'")
            if "labels" not in batch:
                raise KeyError("Batch is missing required key 'labels'")
            if "valid" not in batch:
                raise KeyError("Batch is missing required key 'valid'")

            images = batch["images"]
            labels_grid = batch["labels"]
            valid_grid = batch["valid"].bool()

            if not isinstance(images, torch.Tensor):
                raise TypeError(f"batch['images'] must be a torch.Tensor, got {type(images)}")
            if not isinstance(labels_grid, torch.Tensor):
                labels_grid = torch.as_tensor(labels_grid, device=device)
            if not isinstance(valid_grid, torch.Tensor):
                valid_grid = torch.as_tensor(valid_grid, device=device).bool()

            block_mask = _get_batch_block_mask(batch)
            if block_mask is not None:
                block_mask = block_mask.to(device, non_blocking=True).bool()

            out = model(
                images,
                block_mask=block_mask,
                return_heatmap=False,
            )

            probs_flat = _get_probs_from_model_output(out).to(device)
            labels_flat = _flatten_batch_grid(labels_grid).to(device)
            valid_flat = _flatten_batch_grid(valid_grid).to(device).bool()

            if probs_flat.shape != labels_flat.shape:
                raise ValueError(
                    "Model probability shape does not match label shape after flattening: "
                    f"probs {tuple(probs_flat.shape)}, labels {tuple(labels_flat.shape)}. "
                    "Check block_size, model config, and collator padding."
                )

            if valid_flat.shape != labels_flat.shape:
                raise ValueError(
                    "Valid mask shape does not match label shape after flattening: "
                    f"valid {tuple(valid_flat.shape)}, labels {tuple(labels_flat.shape)}."
                )

            pred_flat = probs_flat >= threshold

            batch_loss_sum, batch_loss_count = _binary_cross_entropy_from_probs(
                probs=probs_flat,
                labels=labels_flat,
                valid_mask=valid_flat,
            )
            overall_loss_sum += batch_loss_sum
            overall_loss_count += batch_loss_count

            y_true_valid = labels_flat[valid_flat].detach().cpu().numpy().astype(bool)
            y_pred_valid = pred_flat[valid_flat].detach().cpu().numpy().astype(bool)
            y_prob_valid = probs_flat[valid_flat].detach().cpu().numpy().astype(np.float32)

            all_y_true_parts.append(y_true_valid)
            all_y_prob_parts.append(y_prob_valid)

            overall_counts.update(
                y_true=y_true_valid,
                y_pred=y_pred_valid,
            )

            labels_grid_cpu = labels_grid.detach().cpu()
            valid_grid_cpu = valid_grid.detach().cpu()
            probs_flat_cpu = probs_flat.detach().cpu()
            pred_flat_cpu = pred_flat.detach().cpu()

            B = int(images.shape[0])
            ids = _ids_from_batch(batch_cpu, batch_size=B)

            Bh_max = int(labels_grid_cpu.shape[1])
            Bw_max = int(labels_grid_cpu.shape[2])

            probs_grid_cpu = probs_flat_cpu.reshape(B, Bh_max, Bw_max)
            pred_grid_cpu = pred_flat_cpu.reshape(B, Bh_max, Bw_max)

            for sample_i in range(B):
                Bh_i, Bw_i = _grid_shape_for_sample(
                    batch_cpu,
                    sample_index=sample_i,
                    fallback_shape=(Bh_max, Bw_max),
                )

                labels_i = labels_grid_cpu[sample_i, :Bh_i, :Bw_i].numpy()
                valid_i = valid_grid_cpu[sample_i, :Bh_i, :Bw_i].numpy().astype(bool)
                probs_i = probs_grid_cpu[sample_i, :Bh_i, :Bw_i].numpy().astype(np.float32)
                pred_i = pred_grid_cpu[sample_i, :Bh_i, :Bw_i].numpy().astype(bool)

                y_true_i = labels_i[valid_i].astype(bool)
                y_pred_i = pred_i[valid_i].astype(bool)
                y_prob_i = probs_i[valid_i].astype(np.float32)

                metrics_i = _sample_metrics(
                    y_true=y_true_i,
                    y_pred=y_pred_i,
                    y_prob=y_prob_i,
                    threshold=threshold,
                )

                sample_results.append(
                    {
                        "id": ids[sample_i],
                        "batch_index": batch_idx,
                        "sample_index": sample_i,
                        "grid_shape": (Bh_i, Bw_i),
                        "labels": labels_i.astype(np.uint8, copy=False),
                        "valid": valid_i.astype(np.bool_, copy=False),
                        "probs": probs_i.astype(np.float32, copy=False),
                        "pred": pred_i.astype(np.bool_, copy=False),
                        "metrics": metrics_i,
                    }
                )

    overall_metrics = _metrics_from_counts(
        overall_counts,
        threshold=threshold,
        loss_sum=overall_loss_sum,
        loss_count=overall_loss_count,
    )

    if all_y_true_parts:
        all_y_true = np.concatenate(all_y_true_parts, axis=0)
        all_y_prob = np.concatenate(all_y_prob_parts, axis=0)

        overall_metrics["roc_auc"] = _binary_auc_roc(
            y_true=all_y_true,
            y_score=all_y_prob,
        )
        overall_metrics["pr_auc"] = _binary_auc_pr(
            y_true=all_y_true,
            y_score=all_y_prob,
        )
    else:
        overall_metrics["roc_auc"] = float("nan")
        overall_metrics["pr_auc"] = float("nan")

    result: Dict[str, Any] = {
        "metrics": overall_metrics,
        "sample_results": sample_results,
        "cfg": cfg,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        _write_metrics_json(
            overall_metrics,
            os.path.join(save_dir, "metrics.json"),
        )

        _write_sample_metrics_csv(
            sample_results,
            os.path.join(save_dir, "sample_metrics.csv"),
        )

        if save_sample_arrays:
            _save_sample_npz_files(sample_results, save_dir)

        if make_visualizations:
            save_evaluation_visualizations(
                result=result,
                save_dir=save_dir,
                max_visualized_samples=max_visualized_samples,
            )

    return result


def save_evaluation_visualizations(
    result: Mapping[str, Any],
    save_dir: str,
    max_visualized_samples: int = 8,
) -> None:
    """
    Save standard evaluation visualizations using functions from viz.py.
    """
    if (
        plot_auc_bar is None
        or plot_auc_curves is None
        or plot_confusion_matrix is None
        or plot_metric_bar is None
        or plot_probability_histogram is None
        or plot_sample_result is None
    ):
        raise ImportError(
            "Could not import visualization functions from evaluation.viz. "
            "Make sure viz.py exists in the evaluation folder."
        )

    os.makedirs(save_dir, exist_ok=True)

    metrics = result["metrics"]
    sample_results = result["sample_results"]

    plot_confusion_matrix(
        metrics=metrics,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
        show=False,
    )

    plot_metric_bar(
        metrics=metrics,
        save_path=os.path.join(save_dir, "metric_bar.png"),
        show=False,
    )

    plot_probability_histogram(
        sample_results=sample_results,
        threshold=float(metrics["threshold"]),
        save_path=os.path.join(save_dir, "probability_histogram.png"),
        show=False,
    )

    plot_auc_curves(
        sample_results=sample_results,
        metrics=metrics,
        save_path=os.path.join(save_dir, "auc_curves.png"),
        show=False,
    )

    plot_auc_bar(
        metrics=metrics,
        save_path=os.path.join(save_dir, "auc_bar.png"),
        show=False,
    )

    sample_viz_dir = os.path.join(save_dir, "sample_plots")
    os.makedirs(sample_viz_dir, exist_ok=True)

    n = min(int(max_visualized_samples), len(sample_results))

    for i in range(n):
        sample = sample_results[i]
        safe_id = str(sample["id"]).replace("/", "_").replace("\\", "_")

        plot_sample_result(
            sample=sample,
            threshold=float(metrics["threshold"]),
            save_path=os.path.join(sample_viz_dir, f"{i:05d}_{safe_id}.png"),
            show=False,
        )


def print_metrics(metrics: Mapping[str, Any]) -> None:
    """
    Pretty-print key evaluation metrics.
    """
    keys = [
        "total_valid_blocks",
        "bce_loss",
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
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    print("Evaluation metrics")
    print("-" * 40)

    for key in keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if isinstance(value, float):
            if key in {"tp", "fp", "tn", "fn", "total_valid_blocks"}:
                print(f"{key:24s}: {int(value)}")
            else:
                print(f"{key:24s}: {value:.6f}")
        else:
            print(f"{key:24s}: {value}")


def evaluate_and_print(
    model_ckpt: str,
    device: Union[str, torch.device],
    test_loader: Iterable[Mapping[str, Any]],
    block_size=None,
    threshold: float = 0.5,
    model_cfg_kwargs: Optional[dict[str, Any]] = None,
    strict: bool = True,
    save_dir: Optional[str] = None,
    make_visualizations: bool = True,
    max_visualized_samples: int = 8,
) -> Dict[str, Any]:
    """
    Convenience wrapper around evaluate_test_loader(...) that prints metrics.
    """
    result = evaluate_test_loader(
        model_ckpt=model_ckpt,
        device=device,
        test_loader=test_loader,
        block_size=block_size,
        threshold=threshold,
        model_cfg_kwargs=model_cfg_kwargs,
        strict=strict,
        save_dir=save_dir,
        make_visualizations=make_visualizations,
        max_visualized_samples=max_visualized_samples,
    )

    print_metrics(result["metrics"])

    if save_dir is not None:
        print(f"\nSaved evaluation outputs to: {save_dir}")

    return result