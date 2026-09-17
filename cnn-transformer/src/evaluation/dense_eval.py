"""
Evaluation for pixel-resolution dense confidence maps.

Answers the two questions the dense refinement has to survive:

1. Did dense refinement preserve detection quality?
   Aggregate the dense map back over each original block and score it against
   the held-out block labels (`detection_fidelity_metrics`).

2. Does the dense output contain meaningful image-conditioned spatial structure
   beyond simple interpolation?
   `grid_artifact_ratio`, `image_structure_alignment`, `equivariance_error`,
   and `localized_perturbation_ratio`, each compared against the nearest /
   bilinear / bicubic / Gaussian baselines built by `upsample_patch_probs`.

None of these require dense ground truth.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


BASELINE_MODES = ("nearest", "bilinear", "bicubic", "gaussian")


# ----------------------------------------------------------------------
# Interpolation baselines
# ----------------------------------------------------------------------

def _gaussian_kernel1d(sigma: float, device, dtype) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur with reflect padding. x: [B, 1, H, W]."""
    if sigma <= 0:
        return x

    k = _gaussian_kernel1d(sigma, x.device, x.dtype)
    radius = (k.numel() - 1) // 2

    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, -1, 1))

    return x


def upsample_patch_probs(
    patch_probs: torch.Tensor,
    block_size: int,
    mode: str = "nearest",
    gaussian_sigma: Optional[float] = None,
) -> torch.Tensor:
    """
    Build a dense map from block confidences with a fixed interpolation rule.

    patch_probs: [B, Bh, Bw] or [B, 1, Bh, Bw]
    returns:     [B, 1, Bh*block_size, Bw*block_size]

    "gaussian" is nearest-neighbour upsampling followed by a Gaussian blur whose
    default sigma is half a block, which is the usual way people smooth away the
    blocky look without learning anything.
    """
    if patch_probs.ndim == 3:
        patch_probs = patch_probs.unsqueeze(1)
    if patch_probs.ndim != 4 or patch_probs.shape[1] != 1:
        raise ValueError(
            f"patch_probs must have shape [B, Bh, Bw] or [B, 1, Bh, Bw]; "
            f"got {tuple(patch_probs.shape)}"
        )

    patch_probs = patch_probs.float()

    if mode not in BASELINE_MODES:
        raise ValueError(f"mode must be one of {BASELINE_MODES}; got {mode!r}")

    if mode in ("nearest", "gaussian"):
        dense = F.interpolate(patch_probs, scale_factor=block_size, mode="nearest")
        if mode == "gaussian":
            sigma = gaussian_sigma if gaussian_sigma is not None else block_size / 2.0
            dense = gaussian_blur(dense, sigma)
    else:
        dense = F.interpolate(
            patch_probs,
            scale_factor=block_size,
            mode=mode,
            align_corners=False,
        )

    # Bicubic overshoots outside [0, 1], and a blur cannot leave it, but clamp
    # everything so every baseline is a valid confidence field.
    return dense.clamp(0.0, 1.0)


# ----------------------------------------------------------------------
# Gradient helpers
# ----------------------------------------------------------------------

def _forward_diffs(x: torch.Tensor):
    """
    Axis-separated forward differences.

    Returns (dy, dx) where dy[..., i, :] sits between rows i and i+1, and
    dx[..., :, j] sits between columns j and j+1.
    """
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = x[..., :, 1:] - x[..., :, :-1]
    return dy, dx


def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Per-pixel gradient magnitude, same shape as x, using forward differences
    with the last row/column replicated.
    """
    dy, dx = _forward_diffs(x)
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    return torch.sqrt(dy ** 2 + dx ** 2 + 1e-12)


# ----------------------------------------------------------------------
# 3. Grid-artifact metric
# ----------------------------------------------------------------------

def grid_artifact_ratio(
    dense: torch.Tensor,
    block_size: int,
    valid_pixels: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    How strongly confidence-map gradients align with the artificial block seams.

        R_grid = E_boundary / (E_interior + eps)

    A difference between rows i-1 and i lies on a seam exactly when i is a
    multiple of block_size; same for columns. Lower is better, and 1.0 means
    seams are indistinguishable from anywhere else.

    Nearest-neighbour upsampling is constant inside each block, so its
    E_interior is exactly zero and R_grid is infinite. That is the honest
    answer but awkward in a table, so `grid_artifact_fraction` is also
    reported: the share of total gradient mass that sits on seams, which is
    bounded in [0, 1], equals 1.0 for nearest, and drops toward the seams'
    natural share (about 1 / block_size) for an artifact-free map.

    dense: [B, 1, Ht, Wt]
    """
    dy, dx = _forward_diffs(dense)
    dy, dx = dy.abs(), dx.abs()

    Ht, Wt = dense.shape[-2:]

    # dy[..., i, :] separates rows i and i+1 -> seam when (i + 1) % block_size == 0
    row_idx = torch.arange(Ht - 1, device=dense.device)
    col_idx = torch.arange(Wt - 1, device=dense.device)
    row_seam = ((row_idx + 1) % block_size == 0).view(1, 1, -1, 1)
    col_seam = ((col_idx + 1) % block_size == 0).view(1, 1, 1, -1)

    if valid_pixels is not None:
        vy = valid_pixels[..., 1:, :] & valid_pixels[..., :-1, :]
        vx = valid_pixels[..., :, 1:] & valid_pixels[..., :, :-1]
    else:
        vy = torch.ones_like(dy, dtype=torch.bool)
        vx = torch.ones_like(dx, dtype=torch.bool)

    y_seam = vy & row_seam
    x_seam = vx & col_seam
    y_int = vy & ~row_seam
    x_int = vx & ~col_seam

    boundary_sum = (dy * y_seam).sum() + (dx * x_seam).sum()
    boundary_n = y_seam.sum() + x_seam.sum()

    interior_sum = (dy * y_int).sum() + (dx * x_int).sum()
    interior_n = y_int.sum() + x_int.sum()

    e_boundary = float(boundary_sum / boundary_n.clamp_min(1))
    e_interior = float(interior_sum / interior_n.clamp_min(1))

    total_mass = float(boundary_sum) + float(interior_sum)

    return {
        "grid_artifact_ratio": (
            float("inf") if e_interior <= 0.0 else e_boundary / e_interior
        ),
        "grid_artifact_fraction": (
            float("nan") if total_mass <= eps else float(boundary_sum) / total_mass
        ),
        "grid_boundary_gradient": e_boundary,
        "grid_interior_gradient": e_interior,
    }


# ----------------------------------------------------------------------
# 4. Image-structure alignment
# ----------------------------------------------------------------------

def image_structure_alignment(
    dense: torch.Tensor,
    guide: torch.Tensor,
    valid_pixels: Optional[torch.Tensor] = None,
    quantile: float = 0.9,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Do the places where the confidence map changes coincide with real image
    structure?

    Reports two complementary numbers, both higher-is-better:

    alignment_ratio
        Mean |grad I| at pixels in the top (1 - quantile) of |grad P|, divided
        by the mean |grad I| everywhere. 1.0 means the prediction's edges fall
        on ordinary image locations; above 1.0 means they concentrate on real
        structure.
    alignment_corr
        Pearson correlation between |grad P| and |grad I| over valid pixels.

    This is a structural alignment measure, not pixel segmentation accuracy.

    dense: [B, 1, Ht, Wt], guide: [B, 1, Ht, Wt] (e.g. high-pass SSHA)
    """
    gP = gradient_magnitude(dense)
    gI = gradient_magnitude(guide)

    if valid_pixels is not None:
        mask = valid_pixels.reshape(-1)
        gP = gP.reshape(-1)[mask]
        gI = gI.reshape(-1)[mask]
    else:
        gP = gP.reshape(-1)
        gI = gI.reshape(-1)

    if gP.numel() == 0:
        return {"alignment_ratio": float("nan"), "alignment_corr": float("nan")}

    # Accumulate in float64 for the correlation, which sums millions of terms
    # after mean-centering. MPS has no float64, so this runs on CPU regardless
    # of where the model lives; these are reduction-only ops on one batch.
    gP = gP.detach().cpu().double()
    gI = gI.detach().cpu().double()

    mean_gI = gI.mean()

    # torch.quantile caps out around 16M elements, so estimate the cut from a
    # random subsample rather than silently falling back to a different statistic.
    if gP.numel() <= 8_000_000:
        thresh = torch.quantile(gP, quantile)
    else:
        idx = torch.randint(0, gP.numel(), (8_000_000,), device=gP.device)
        thresh = torch.quantile(gP[idx], quantile)

    selected = gP >= thresh

    if selected.sum() == 0 or mean_gI <= eps:
        alignment_ratio = float("nan")
    else:
        alignment_ratio = float(gI[selected].mean() / (mean_gI + eps))

    # Pearson correlation
    gP_c = gP - gP.mean()
    gI_c = gI - gI.mean()
    denom = gP_c.norm() * gI_c.norm()
    alignment_corr = float((gP_c @ gI_c) / denom) if float(denom) > eps else float("nan")

    return {"alignment_ratio": alignment_ratio, "alignment_corr": alignment_corr}


# ----------------------------------------------------------------------
# 5. Equivariance / consistency test
# ----------------------------------------------------------------------

@torch.no_grad()
def equivariance_error(
    model,
    images: torch.Tensor,
    block_mask: Optional[torch.Tensor] = None,
    flip_dims: Sequence[int] = (-1,),
    valid_pixels: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Spatial consistency under a flip, with no dense labels required.

        P      = model(I)
        P_flip = model(flip(I))
        E      = mean |P - inverse_flip(P_flip)|

    Lower is better. flip_dims=(-1,) is a horizontal flip, (-2,) vertical.

    The image height/width must be exact multiples of block_size, otherwise the
    flip shifts the block tiling and the two runs are not comparable.
    """
    block_size = int(model.cfg.block_size)
    H, W = images.shape[-2:]

    if H % block_size or W % block_size:
        raise ValueError(
            f"Image size {(H, W)} is not a multiple of block_size {block_size}; "
            "crop the input before running the equivariance test."
        )

    model.eval()

    out = model(images, block_mask=block_mask, return_dense=True)
    P = out["dense_probs"]

    flipped_mask = None
    if block_mask is not None:
        flipped_mask = torch.flip(block_mask, dims=list(flip_dims))

    out_flip = model(torch.flip(images, dims=list(flip_dims)), block_mask=flipped_mask, return_dense=True)
    P_hat = torch.flip(out_flip["dense_probs"], dims=list(flip_dims))

    diff = (P - P_hat).abs()

    if valid_pixels is not None:
        n = valid_pixels.sum()
        if n == 0:
            return {"equivariance_mae": float("nan"), "equivariance_max": float("nan")}
        mae = float((diff * valid_pixels).sum() / n)
        max_err = float(diff[valid_pixels].max())
    else:
        mae = float(diff.mean())
        max_err = float(diff.max())

    return {"equivariance_mae": mae, "equivariance_max": max_err}


# ----------------------------------------------------------------------
# 6. Localized perturbation test
# ----------------------------------------------------------------------

def _gaussian_bump(
    shape: tuple,
    center: tuple,
    sigma: float,
    device,
    dtype,
) -> torch.Tensor:
    """[1, 1, H, W] Gaussian bump peaking at 1.0."""
    H, W = shape
    yy = torch.arange(H, device=device, dtype=dtype).view(-1, 1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, -1)
    cy, cx = center
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return torch.exp(-d2 / (2.0 * sigma ** 2)).view(1, 1, H, W)


@torch.no_grad()
def localized_perturbation_ratio(
    model,
    images: torch.Tensor,
    block_mask: Optional[torch.Tensor] = None,
    region_size: Optional[int] = None,
    amplitude: float = 3.0,
    seed: int = 0,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Does the dense map respond where the image was changed?

    Adds a localized Gaussian bump to the input, then reports

        R_local = mean dP inside the perturbed region
                / (mean dP outside + eps)

    Higher is better. Fixed interpolation of unchanged block confidences cannot
    produce a localized response, so its ratio sits near 1 except where the bump
    flips a whole block. `amplitude` is in units of the per-sample standard
    deviation of each channel, so it adapts to the data scale.

    region_size defaults to 3 blocks across, capped so it always fits; the
    centre is drawn so the whole region stays inside the image.
    """
    block_size = int(model.cfg.block_size)
    B, C, H, W = images.shape

    if region_size is None:
        region_size = 3 * block_size

    # Never let the region exceed a third of the smaller side, otherwise
    # "outside" has too few pixels for the ratio to mean anything.
    region_size = int(min(region_size, max(block_size, min(H, W) // 3)))

    if region_size < 1:
        raise ValueError(f"Cannot fit a perturbation region inside image {(H, W)}")

    model.eval()

    generator = torch.Generator(device="cpu").manual_seed(seed)
    sigma = max(region_size / 4.0, 1.0)

    # Keep the region fully inside the image; fall back to the centre when the
    # image is barely larger than the region.
    half = region_size // 2

    def _draw(extent: int) -> int:
        lo, hi = half, extent - half
        if hi <= lo:
            return extent // 2
        return int(torch.randint(lo, hi, (1,), generator=generator).item())

    cy = _draw(H)
    cx = _draw(W)

    bump = _gaussian_bump((H, W), (cy, cx), sigma, images.device, images.dtype)
    scale = images.reshape(B, C, -1).std(dim=-1).view(B, C, 1, 1)
    perturbed = images + amplitude * scale * bump

    P0 = model(images, block_mask=block_mask, return_dense=True)["dense_probs"]
    P1 = model(perturbed, block_mask=block_mask, return_dense=True)["dense_probs"]

    dP = (P1 - P0).abs()

    Ht, Wt = dP.shape[-2:]
    y0, y1 = max(0, cy - half), min(Ht, cy + half + 1)
    x0, x1 = max(0, cx - half), min(Wt, cx + half + 1)

    inside_mask = torch.zeros(1, 1, Ht, Wt, dtype=torch.bool, device=dP.device)
    inside_mask[..., y0:y1, x0:x1] = True

    inside = float((dP * inside_mask).sum() / inside_mask.sum().clamp_min(1))
    outside_n = (~inside_mask).sum() * dP.shape[0]
    outside = float((dP * ~inside_mask).sum() / outside_n.clamp_min(1))

    return {
        "perturbation_ratio": inside / (outside + eps),
        "perturbation_inside": inside,
        "perturbation_outside": outside,
        "perturbation_center": (cy, cx),
        "perturbation_region_size": region_size,
    }


# ----------------------------------------------------------------------
# 1. Detection fidelity
# ----------------------------------------------------------------------

def aggregate_dense(dense: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pool a dense map back to one mean-occupancy value per original block."""
    return F.avg_pool2d(dense, kernel_size=block_size, stride=block_size).squeeze(1)


def detection_fidelity_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Precision / recall / F1 / AUROC / AUPRC for aggregated block scores.

    scores and labels are 1-D and already restricted to valid blocks.
    """
    try:
        from training.metrics import compute_metrics
    except ImportError:
        from src.training.metrics import compute_metrics  # type: ignore

    return compute_metrics(labels_v=labels, probs_v=scores, threshold=threshold)


# Metrics that actually change as the threshold moves, so a sweep over them is
# meaningful. roc_auc, pr_auc, and label_positive_rate are threshold-independent
# by construction and would just pick an arbitrary threshold.
THRESHOLD_SELECTABLE_METRICS = (
    "accuracy", "precision", "recall", "specificity",
    "f1", "f0_5", "iou", "balanced_accuracy", "pred_positive_rate",
)


@torch.no_grad()
def collect_aggregated_scores(
    model,
    loader: Iterable[Mapping[str, Any]],
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the dense head over a loader and pool its output back to one mean-
    occupancy score per valid block, alongside the matching labels.

    This is the same aggregated quantity `evaluate_dense_loader` scores as the
    "learned" method, exposed on its own so a threshold can be selected on one
    loader (validation) before scoring another (test).

    Returns:
        labels: [N] float, in {0, 1}
        scores: [N] float, in [0, 1]
    """
    model = model.to(device)
    model.eval()

    block_size = int(model.cfg.block_size)
    scores_all, labels_all = [], []

    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        valid = batch["valid"].to(device).bool()
        block_mask = batch.get("block_attn_mask", batch["valid"]).to(device).bool()

        out = model(images, block_mask=block_mask, return_dense=True)
        agg = aggregate_dense(out["dense_probs"], block_size)

        scores_all.append(agg[valid])
        labels_all.append(labels[valid])

    if not scores_all:
        return torch.empty(0), torch.empty(0)

    return torch.cat(labels_all, dim=0).float(), torch.cat(scores_all, dim=0).float()


def select_best_threshold(
    labels_v: torch.Tensor,
    probs_v: torch.Tensor,
    metric_name: str,
    thresholds: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Sweep a probability threshold and pick the one maximizing metric_name.

    Mirrors the boxcar training notebook's BEST_METRIC threshold selection:
    pick a threshold on validation data, then apply that fixed threshold when
    reporting test-set numbers so the two don't leak into each other.

    Returns:
        {
          "threshold": best threshold found,
          "value": metric value at that threshold,
          "sweep": [{"threshold": t, metric_name: v}, ...] for every t tried,
        }
    """
    if metric_name not in THRESHOLD_SELECTABLE_METRICS:
        raise ValueError(
            f"{metric_name!r} does not vary with threshold and cannot be used to "
            f"select one. Choose one of: {THRESHOLD_SELECTABLE_METRICS}"
        )

    n_pos = int(labels_v.sum().item()) if labels_v.numel() else 0
    n_neg = labels_v.numel() - n_pos

    if n_pos == 0 or n_neg == 0:
        # With only one class present, sklearn's zero-division fallback makes
        # every threshold score identically (typically 0.0), which is finite
        # and would otherwise be returned as a misleadingly confident "best"
        # threshold. There is no signal to select on, so this is a hard error
        # rather than a silent arbitrary pick.
        raise ValueError(
            f"Cannot select a threshold: labels_v has {n_pos} positive and "
            f"{n_neg} negative blocks. Every threshold scores identically "
            "when only one class is present. Check the loader/split."
        )

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    try:
        from training.metrics import compute_metrics
    except ImportError:
        from src.training.metrics import compute_metrics  # type: ignore

    best_t, best_v = 0.5, float("-inf")
    sweep = []

    for t in thresholds:
        t = float(t)
        m = compute_metrics(labels_v=labels_v, probs_v=probs_v, threshold=t)
        v = float(m[metric_name])
        sweep.append({"threshold": t, metric_name: v})

        if np.isfinite(v) and v > best_v:
            best_v, best_t = v, t

    if not np.isfinite(best_v):
        raise ValueError(
            f"{metric_name} was NaN/inf at every threshold tried; check that "
            "labels_v contains both classes."
        )

    return {"threshold": best_t, "value": best_v, "metric": metric_name, "sweep": sweep}


# ----------------------------------------------------------------------
# Loader-level driver
# ----------------------------------------------------------------------

def _pixel_valid_from_blocks(valid: torch.Tensor, block_size: int) -> torch.Tensor:
    """[B, Bh, Bw] bool -> [B, 1, Bh*bs, Bw*bs] bool."""
    return F.interpolate(
        valid.unsqueeze(1).float(), scale_factor=block_size, mode="nearest"
    ).bool()


@torch.no_grad()
def evaluate_dense_loader(
    model,
    loader: Iterable[Mapping[str, Any]],
    device,
    threshold: float = 0.5,
    edge_channel: int = -1,
    include_baselines: bool = True,
    run_equivariance: bool = True,
    run_perturbation: bool = True,
    flip_dims: Sequence[int] = (-1,),
    perturbation_seed: int = 0,
    max_structural_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the full dense evaluation over a loader.

    Produces one entry per method: the learned dense head plus, when
    include_baselines is True, the nearest / bilinear / bicubic / Gaussian
    reconstructions of the *same* patch confidences. Every method is scored with
    the same detection-fidelity metrics and the same structural metrics, so the
    rows of the ablation table are directly comparable.

    The equivariance and perturbation tests need extra forward passes, so
    max_structural_batches caps how many batches they run on.

    Returns:
        {
          "methods": {name: {metrics...}},
          "n_batches": int,
          "n_valid_blocks": int,
        }
    """
    model = model.to(device)
    model.eval()

    block_size = int(model.cfg.block_size)

    method_names = ["learned"]
    if include_baselines:
        method_names += list(BASELINE_MODES)

    scores: Dict[str, list] = {name: [] for name in method_names}
    labels_all: list = []

    # Structural metrics are averaged over batches rather than pooled, because
    # each is already a per-image ratio.
    structural: Dict[str, Dict[str, list]] = {
        name: {} for name in method_names
    }
    equivariance: Dict[str, list] = {}
    perturbation: Dict[str, list] = {}

    def _accumulate(store: Dict[str, list], values: Mapping[str, Any]):
        for key, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                store.setdefault(key, []).append(float(value))

    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        valid = batch["valid"].to(device).bool()
        block_mask = batch.get("block_attn_mask", batch["valid"]).to(device).bool()

        out = model(images, block_mask=block_mask, return_dense=True)

        B = images.shape[0]
        Bh, Bw = int(out["Bh"]), int(out["Bw"])
        Ht, Wt = Bh * block_size, Bw * block_size

        patch_probs = out["probs"].reshape(B, Bh, Bw)
        valid_pixels = _pixel_valid_from_blocks(valid, block_size)

        ch = edge_channel % images.shape[1]
        guide = images[:, ch : ch + 1, :Ht, :Wt]

        dense_by_method = {"learned": out["dense_probs"]}
        if include_baselines:
            for mode in BASELINE_MODES:
                dense_by_method[mode] = upsample_patch_probs(patch_probs, block_size, mode)

        labels_all.append(labels[valid])

        do_structural = (
            max_structural_batches is None or batch_idx < max_structural_batches
        )

        for name, dense in dense_by_method.items():
            scores[name].append(aggregate_dense(dense, block_size)[valid])

            if not do_structural:
                continue

            _accumulate(
                structural[name],
                grid_artifact_ratio(dense, block_size, valid_pixels),
            )
            _accumulate(
                structural[name],
                image_structure_alignment(dense, guide, valid_pixels),
            )

        if do_structural and run_equivariance:
            try:
                _accumulate(
                    equivariance,
                    equivariance_error(
                        model, images, block_mask=block_mask,
                        flip_dims=flip_dims, valid_pixels=valid_pixels,
                    ),
                )
            except ValueError:
                # Padded batches can break the multiple-of-block_size requirement.
                pass

        if do_structural and run_perturbation:
            _accumulate(
                perturbation,
                localized_perturbation_ratio(
                    model, images, block_mask=block_mask,
                    seed=perturbation_seed + batch_idx,
                ),
            )

        n_batches += 1

    if not labels_all:
        return {"methods": {}, "n_batches": 0, "n_valid_blocks": 0}

    y_true = torch.cat(labels_all, dim=0).float()

    methods: Dict[str, Dict[str, float]] = {}

    for name in method_names:
        entry = detection_fidelity_metrics(
            scores=torch.cat(scores[name], dim=0),
            labels=y_true,
            threshold=threshold,
        )
        entry.update({k: float(np.mean(v)) for k, v in structural[name].items()})
        methods[name] = entry

    # Equivariance and perturbation probe the model itself, so they only apply
    # to the learned head; the fixed baselines have no model to probe.
    if equivariance:
        methods["learned"].update({k: float(np.mean(v)) for k, v in equivariance.items()})
    if perturbation:
        methods["learned"].update({k: float(np.mean(v)) for k, v in perturbation.items()})

    return {
        "methods": methods,
        "n_batches": n_batches,
        "n_valid_blocks": int(y_true.numel()),
    }


def format_ablation_table(
    results_by_variant: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str] = (
        "f1",
        "pr_auc",
        "roc_auc",
        "grid_artifact_fraction",
        "alignment_ratio",
        "alignment_corr",
        "equivariance_mae",
        "perturbation_ratio",
    ),
) -> str:
    """
    Render a markdown ablation table from {variant_name: metrics_dict}.

    Pass e.g. {"Nearest": res["methods"]["nearest"], "Learned": res["methods"]["learned"]}.
    """
    header = "| Variant | " + " | ".join(keys) + " |"
    sep = "|---|" + "|".join(["---:"] * len(keys)) + "|"

    rows = [header, sep]

    for variant, metrics in results_by_variant.items():
        cells = []
        for key in keys:
            value = metrics.get(key, None)
            cells.append("" if value is None else f"{float(value):.4f}")
        rows.append(f"| {variant} | " + " | ".join(cells) + " |")

    return "\n".join(rows)
