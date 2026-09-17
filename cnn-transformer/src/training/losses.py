# src/training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_weighted_bce(alpha: float, device: torch.device) -> nn.Module:
    """
    Weighted BCE: multiply positives by pos_weight = alpha / (1 - alpha).
    """
    pos_weight = alpha / max(1e-6, (1.0 - alpha))
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

class FocalLoss(nn.Module):
    """
    Binary focal loss on logits: FL = -(alpha_t) * (1 - p_t)^gamma * log(p_t)
    alpha: weight for positives; gamma: focusing parameter (>=0).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        pt   = prob * targets + (1 - prob) * (1 - targets)
        at   = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -at * ((1 - pt) ** self.gamma) * torch.log(pt.clamp_min(1e-8))
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()


# ----------------------------------------------------------------------
# Dense pixel-resolution refinement
# ----------------------------------------------------------------------

def aggregate_dense_to_patch(dense_probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Mean predicted occupancy per original block.

    dense_probs: [B, 1, Ht, Wt] in [0, 1]
    returns m:   [B, Bh, Bw], m_ij = (1/block_size^2) * sum of pixel probs in block ij
    """
    if dense_probs.ndim != 4 or dense_probs.shape[1] != 1:
        raise ValueError(f"dense_probs must have shape [B, 1, Ht, Wt]; got {tuple(dense_probs.shape)}")

    return F.avg_pool2d(dense_probs, kernel_size=block_size, stride=block_size).squeeze(1)


def soft_majority_logit(mean_occupancy: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    Differentiable stand-in for the majority vote that produced the block labels.

    q = sigmoid((m - 0.5) / tau); this returns the logit (m - 0.5) / tau so it can
    be fed straight to BCEWithLogitsLoss, which is numerically safer than taking
    the log of an already-saturated sigmoid.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive; got {tau}")

    return (mean_occupancy - 0.5) / tau


def _guide_gradient_weights(guide: torch.Tensor, eps: float = 1e-6):
    """
    exp(-|grad guide|) along each axis, normalized by the per-sample mean
    gradient so the weighting does not depend on the guidance channel's units.

    Used by edge_aware_smoothness to down-weight the prediction-gradient penalty
    wherever the guidance image itself changes.
    """
    dIx = (guide[..., :, 1:] - guide[..., :, :-1]).abs()
    dIy = (guide[..., 1:, :] - guide[..., :-1, :]).abs()

    dIx = dIx / (dIx.mean(dim=(1, 2, 3), keepdim=True) + eps)
    dIy = dIy / (dIy.mean(dim=(1, 2, 3), keepdim=True) + eps)

    return torch.exp(-dIx), torch.exp(-dIy)


def edge_aware_smoothness(
    dense_probs: torch.Tensor,
    guide: torch.Tensor,
    valid_pixels: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize prediction gradients except where the guidance image also has one.

    dense_probs:  [B, 1, Ht, Wt]
    guide:        [B, 1, Ht, Wt] scientifically meaningful image channel,
                  e.g. rolling-mean high-pass SSHA
    valid_pixels: [B, 1, Ht, Wt] bool, True where the pixel is real (not padding)

    Image gradients are normalized by their per-sample mean so the exp(-|grad I|)
    weighting does not depend on the physical units of the guidance channel.
    """
    dPx = (dense_probs[..., :, 1:] - dense_probs[..., :, :-1]).abs()
    dPy = (dense_probs[..., 1:, :] - dense_probs[..., :-1, :]).abs()

    ex, ey = _guide_gradient_weights(guide, eps)

    wx = dPx * ex
    wy = dPy * ey

    if valid_pixels is None:
        return 0.5 * (wx.mean() + wy.mean())

    # A difference is only meaningful when both of its pixels are real.
    vx = valid_pixels[..., :, 1:] & valid_pixels[..., :, :-1]
    vy = valid_pixels[..., 1:, :] & valid_pixels[..., :-1, :]

    sum_x = (wx * vx).sum()
    sum_y = (wy * vy).sum()
    count = vx.sum() + vy.sum()

    if count == 0:
        return dense_probs.sum() * 0.0

    return (sum_x + sum_y) / count.clamp_min(1)


class DenseRefinementLoss(nn.Module):
    """
    Loss for the dense refinement head:

        L = L_patch + lambda_teacher * L_teacher + lambda_edge * L_edge

    L_patch   constrains the dense map to reproduce the existing block labels
              once aggregated back with the same majority rule that created them.
    L_teacher keeps the aggregated map close to the frozen detector's own
              confidences, so validated detection behavior is preserved.
    L_edge    lets the map stay smooth inside homogeneous regions while still
              changing across real image structure.
    """

    def __init__(
        self,
        alpha: float = 0.7,
        tau: float = 0.1,
        lambda_teacher: float = 1.0,
        lambda_edge: float = 1e-3,
    ):
        super().__init__()
        self.tau = tau
        self.lambda_teacher = lambda_teacher
        self.lambda_edge = lambda_edge

        pos_weight = alpha / max(1e-6, (1.0 - alpha))
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(
        self,
        dense_probs: torch.Tensor,
        block_size: int,
        labels: torch.Tensor,
        valid: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
        guide: torch.Tensor | None = None,
        valid_pixels: torch.Tensor | None = None,
    ) -> dict:
        """
        dense_probs:   [B, 1, Ht, Wt]
        labels:        [B, Bh, Bw] block labels in {0, 1}
        valid:         [B, Bh, Bw] bool, True for real (non-padded) blocks
        teacher_probs: [B, Bh, Bw] frozen detector confidences, or None
        guide:         [B, 1, Ht, Wt] guidance channel for the edge term, or None
        valid_pixels:  [B, 1, Ht, Wt] bool pixel validity, or None

        returns a dict with "loss" plus each component and the aggregated
        patch-level quantities m and q for metric computation.
        """
        m = aggregate_dense_to_patch(dense_probs, block_size)   # [B, Bh, Bw]
        q_logit = soft_majority_logit(m, self.tau)              # [B, Bh, Bw]

        valid = valid.bool()
        if valid.shape != m.shape:
            raise ValueError(
                f"valid shape {tuple(valid.shape)} does not match the aggregated "
                f"patch grid {tuple(m.shape)}; check block_size and collator padding."
            )

        q_logit_v = q_logit[valid]
        labels_v = labels.float()[valid]

        loss_patch = F.binary_cross_entropy_with_logits(
            q_logit_v,
            labels_v,
            pos_weight=self.pos_weight.to(q_logit_v.dtype),
        )

        loss = loss_patch
        components = {"loss_patch": loss_patch}

        if teacher_probs is not None and self.lambda_teacher > 0:
            teacher_v = teacher_probs.detach().float()[valid]
            loss_teacher = F.binary_cross_entropy_with_logits(q_logit_v, teacher_v)
            loss = loss + self.lambda_teacher * loss_teacher
            components["loss_teacher"] = loss_teacher

        if guide is not None and self.lambda_edge > 0:
            loss_edge = edge_aware_smoothness(dense_probs, guide, valid_pixels)
            loss = loss + self.lambda_edge * loss_edge
            components["loss_edge"] = loss_edge

        components["loss"] = loss
        components["m"] = m
        components["q"] = torch.sigmoid(q_logit)

        return components
