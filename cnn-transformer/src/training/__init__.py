# src/training/__init__.py

from .engine import Trainer, DenseTrainer
from .losses import (
    build_weighted_bce,
    FocalLoss,
    DenseRefinementLoss,
    aggregate_dense_to_patch,
    soft_majority_logit,
    edge_aware_smoothness,
)
from .metrics import compute_metrics
from .optim import build_optim_and_sched

__all__ = [
    "Trainer",
    "DenseTrainer",
    "build_weighted_bce",
    "FocalLoss",
    "DenseRefinementLoss",
    "aggregate_dense_to_patch",
    "soft_majority_logit",
    "edge_aware_smoothness",
    "compute_metrics",
    "build_optim_and_sched",
]
