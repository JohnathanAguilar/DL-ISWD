# src/training/engine.py
import os, time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import build_weighted_bce, DenseRefinementLoss
from .metrics import compute_metrics

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        alpha: float = 0.7,
        grad_clip: float = 1.0,
        threshold: float = 0.5,
        save_dir: str = "experiments/exp_001",
        use_amp: bool = False,         # enable for CUDA; keep False on MPS
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_fn = build_weighted_bce(alpha, device)
        self.grad_clip = grad_clip
        self.threshold = threshold
        self.save_dir = save_dir
        self.use_amp = use_amp and (device.type == "cuda")
        os.makedirs(self.save_dir, exist_ok=True)

    def _step(self, batch, optimizer=None) -> Tuple[float, Dict[str, torch.Tensor]]:
        images = batch["images"].to(self.device)      # [B, 2, H_max, W_max]
        labels = batch["labels"].to(self.device)      # [B, Bh_max, Bw_max]
        valid  = batch["valid"].to(self.device)       # [B, Bh_max, Bw_max]
        block_mask = batch.get("block_attn_mask", valid).to(self.device)

        # forward
        if self.use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                out = self.model(images, block_mask=block_mask)
                logits = out["logits"]                 # [B, N]
                token_mask = out["token_mask"]         # [B, N]
                labels_flat = labels.reshape(labels.shape[0], -1).float()
                loss = self.loss_fn(logits[token_mask], labels_flat[token_mask])
        else:
            out = self.model(images, block_mask=block_mask)
            logits = out["logits"]
            token_mask = out["token_mask"]
            labels_flat = labels.reshape(labels.shape[0], -1).float()
            loss = self.loss_fn(logits[token_mask], labels_flat[token_mask])

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

        probs = torch.sigmoid(logits)
        return float(loss.item()), {"logits": logits, "probs": probs, "token_mask": token_mask, "labels_flat": labels_flat}

    def train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        total, steps = 0.0, 0
        for batch in loader:
            loss, _ = self._step(batch, optimizer=optimizer)
            total += loss; steps += 1
        return total / max(steps, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
    
        total = 0.0
        steps = 0
    
        all_probs = []
        all_labels = []
    
        for batch in loader:
            loss, out = self._step(
                batch,
                optimizer=None,
            )
    
            total += loss
            steps += 1
    
            # Select VALID tokens only.
            probs_v = out["probs"][out["token_mask"]]
            labels_v = out["labels_flat"][out["token_mask"]]
    
            all_probs.append(probs_v)
            all_labels.append(labels_v)
    
        if len(all_probs) == 0:
            return total, {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "specificity": float("nan"),
                "f1": float("nan"),
                "f0_5": float("nan"),
                "iou": float("nan"),
                "balanced_accuracy": float("nan"),
                "roc_auc": float("nan"),
                "pr_auc": float("nan"),
                "label_positive_rate": float("nan"),
                "pred_positive_rate": float("nan"),
            }
    
        y_prob = torch.cat(
            all_probs,
            dim=0,
        )
    
        y_true = torch.cat(
            all_labels,
            dim=0,
        )
    
        metrics = compute_metrics(
            labels_v=y_true,
            probs_v=y_prob,
            threshold=self.threshold,
        )
    
        return total / max(steps, 1), metrics

    def save_ckpt(self, epoch: int, optimizer, scheduler, extra_metrics: Dict[str, float]):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": extra_metrics,
        }
        # Store the model config so inference/eval can rebuild the exact model
        # without the caller having to restate every field.
        cfg = getattr(self.model, "cfg", None)
        if cfg is not None and hasattr(cfg, "__dict__"):
            ckpt["config"] = dict(vars(cfg))

        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch:02d}.pt")
        torch.save(ckpt, path)
        return path


class DenseTrainer:
    """
    Train the dense pixel-resolution refinement head.

    The original detector is frozen by default, so the patch-level predictions
    that the paper already reports stay bit-for-bit identical and the only thing
    being learned is the within-block refinement. Metrics are reported on the
    dense map aggregated back to the block grid, which is the quantity that has
    to match the original detector.

    Freezing also puts the CNN stem's BatchNorm layers in eval mode; otherwise
    their running statistics would keep updating and quietly change the frozen
    detector's outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        alpha: float = 0.7,
        tau: float = 0.1,
        lambda_teacher: float = 1.0,
        lambda_edge: float = 1e-3,
        edge_channel: int = -1,
        grad_clip: float = 1.0,
        threshold: float = 0.5,
        save_dir: str = "experiments/dense_001",
        freeze_backbone: bool = True,
        teacher_model: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.model.freeze_backbone(True)

        self.loss_fn = DenseRefinementLoss(
            alpha=alpha,
            tau=tau,
            lambda_teacher=lambda_teacher,
            lambda_edge=lambda_edge,
        ).to(device)

        # When the backbone is unfrozen, the live patch head is no longer a fixed
        # teacher, so a separate frozen copy is required for the distillation term.
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(device).eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)
        elif not freeze_backbone and lambda_teacher > 0:
            raise ValueError(
                "lambda_teacher > 0 with an unfrozen backbone requires an explicit "
                "teacher_model; otherwise the teacher drifts along with the student."
            )

        self.edge_channel = edge_channel
        self.lambda_edge = lambda_edge
        self.grad_clip = grad_clip
        self.threshold = threshold
        self.save_dir = save_dir
        self.block_size = int(model.cfg.block_size)
        os.makedirs(self.save_dir, exist_ok=True)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _prepare_batch(self, batch):
        images = batch["images"].to(self.device)      # [B, C, H_max, W_max]
        labels = batch["labels"].to(self.device)      # [B, Bh_max, Bw_max]
        valid = batch["valid"].to(self.device).bool() # [B, Bh_max, Bw_max]
        block_mask = batch.get("block_attn_mask", valid).to(self.device).bool()
        return images, labels, valid, block_mask

    def _step(self, batch, optimizer=None) -> Tuple[float, Dict[str, torch.Tensor]]:
        images, labels, valid, block_mask = self._prepare_batch(batch)

        out = self.model(images, block_mask=block_mask, return_dense=True)

        Bh, Bw = int(out["Bh"]), int(out["Bw"])
        B = images.shape[0]
        bs = self.block_size
        Ht, Wt = Bh * bs, Bw * bs

        dense_probs = out["dense_probs"]                      # [B, 1, Ht, Wt]
        detector_probs = out["probs"].reshape(B, Bh, Bw)      # [B, Bh, Bw]

        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_out = self.teacher_model(images, block_mask=block_mask)
                teacher_probs = teacher_out["probs"].reshape(B, Bh, Bw)
        else:
            teacher_probs = detector_probs.detach()

        # Guidance channel and pixel validity, both cropped to the tokenized region.
        guide = None
        if self.lambda_edge > 0:
            # Normalize the index so negative values (e.g. -1 for the high-pass
            # channel) select a single channel rather than an empty slice.
            ch = self.edge_channel % images.shape[1]
            guide = images[:, ch : ch + 1, :Ht, :Wt]

        valid_pixels = F.interpolate(
            valid.unsqueeze(1).float(), scale_factor=bs, mode="nearest"
        ).bool()

        components = self.loss_fn(
            dense_probs=dense_probs,
            block_size=bs,
            labels=labels,
            valid=valid,
            teacher_probs=teacher_probs,
            guide=guide,
            valid_pixels=valid_pixels,
        )
        loss = components["loss"]

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.trainable_parameters(), self.grad_clip)
            optimizer.step()

        stats = {
            "q": components["q"].detach(),
            "m": components["m"].detach(),
            "detector_probs": detector_probs.detach(),
            "labels": labels,
            "valid": valid,
        }
        for key in ("loss_patch", "loss_teacher", "loss_edge"):
            if key in components:
                stats[key] = float(components[key].detach().item())

        return float(loss.item()), stats

    def train_epoch(self, loader, optimizer) -> Dict[str, float]:
        self.model.set_train_mode(True)

        totals: Dict[str, float] = {}
        steps = 0

        for batch in loader:
            loss, stats = self._step(batch, optimizer=optimizer)
            totals["loss"] = totals.get("loss", 0.0) + loss
            for key in ("loss_patch", "loss_teacher", "loss_edge"):
                if key in stats:
                    totals[key] = totals.get(key, 0.0) + stats[key]
            steps += 1

        return {k: v / max(steps, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, Dict[str, float]]:
        """
        Detection-fidelity evaluation: score the aggregated dense map against the
        held-out block labels, alongside the original detector on the same batches.
        """
        self.model.eval()

        total = 0.0
        steps = 0

        dense_probs_all, detector_probs_all, labels_all = [], [], []

        for batch in loader:
            loss, stats = self._step(batch, optimizer=None)
            total += loss
            steps += 1

            valid = stats["valid"]
            dense_probs_all.append(stats["q"][valid])
            detector_probs_all.append(stats["detector_probs"][valid])
            labels_all.append(stats["labels"].float()[valid])

        if steps == 0 or not dense_probs_all:
            return float("nan"), {}

        y_true = torch.cat(labels_all, dim=0)

        metrics = compute_metrics(
            labels_v=y_true,
            probs_v=torch.cat(dense_probs_all, dim=0),
            threshold=self.threshold,
        )

        detector_metrics = compute_metrics(
            labels_v=y_true,
            probs_v=torch.cat(detector_probs_all, dim=0),
            threshold=self.threshold,
        )
        metrics.update({f"detector_{k}": v for k, v in detector_metrics.items()})

        return total / steps, metrics

    def save_ckpt(self, epoch: int, optimizer, scheduler, extra_metrics: Dict[str, float]):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "metrics": extra_metrics,
            "config": dict(vars(self.model.cfg)),
        }
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch:02d}.pt")
        torch.save(ckpt, path)
        return path
