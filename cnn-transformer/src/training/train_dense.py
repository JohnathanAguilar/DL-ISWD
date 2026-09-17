# src/training/train_dense.py
"""
Stage 1 of dense pixel-resolution refinement.

Loads a trained patch-level detector, freezes it, attaches the dense refinement
head, and trains only that head against the existing block labels plus the
frozen detector's own confidences. Because the backbone never changes, the
patch-level numbers already reported in the paper are preserved exactly and the
only thing being measured is the contribution of the refinement module.

Pass --unfreeze_backbone for the Stage 2 variant that fine-tunes end to end; a
frozen copy of the detector is kept as the distillation teacher so the teacher
does not drift along with the student.
"""

import copy
import os
import argparse
import json
import random

import numpy as np
import torch

from dataio import (
    DataCollatorBlocks,
    RandomFlip,
    create_dataloaders_train_val,
    make_nc_block_dataset,
)
from models import ModelConfig, BlocksTokenModel
from inference.inference import load_backbone_weights

from .optim import build_optim_and_sched
from .engine import DenseTrainer


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train the dense pixel-resolution refinement head on a frozen detector"
    )

    # --- Data ---
    # Pass the three split directories to reuse the pass-level grouped split
    # from the paper, or --nc_dir to split a single folder on the fly.
    ap.add_argument("--nc_dir", default=None, help="Single folder, split randomly")
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--valid_dir", default=None)
    ap.add_argument("--test_dir", default=None)
    ap.add_argument("--valid_ratio", type=float, default=0.10)
    ap.add_argument(
        "--channel_vars",
        nargs="+",
        default=["ssha_unfiltered", "ssha_unfiltered_rolling_highpass"],
        help="Variable names to read as channels (e.g., SSH + highpass)",
    )
    ap.add_argument("--mask_var", default="block_mask", help="Variable name for block labels")
    ap.add_argument("--block", type=int, default=10, help="Square block size (pixels)")
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for dense-head init and batch order. Vary this across runs "
             "to report mean +/- std.",
    )
    ap.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed for the train/val split. Hold this FIXED while --seed varies, "
             "otherwise each run reshards the data and the reported spread mixes "
             "model-init variance with split variance. Defaults to --seed.",
    )

    # --- Dataloaders ---
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--val_bs", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true", default=False)
    ap.add_argument("--bucket_by_grid", action="store_true", default=True)

    # --- Augmentation ---
    ap.add_argument(
        "--hflip_prob", type=float, default=0.0,
        help="Probability of a horizontal flip applied to training items only (0 = disabled)",
    )
    ap.add_argument(
        "--vflip_prob", type=float, default=0.0,
        help="Probability of a vertical flip applied to training items only (0 = disabled)",
    )

    # --- Detector checkpoint to refine ---
    ap.add_argument(
        "--detector_ckpt",
        required=True,
        help="Trained patch-level checkpoint, e.g. models/rolling_mean_highpass/best.pt",
    )
    ap.add_argument("--unfreeze_backbone", action="store_true", default=False)

    # --- Dense head ---
    ap.add_argument("--dense_hidden", type=int, default=64)
    ap.add_argument(
        "--no_cnn_features",
        action="store_true",
        default=False,
        help="Transformer-only ablation: drop the full-resolution CNN features",
    )
    ap.add_argument(
        "--dense_token_upsample",
        choices=["nearest", "bilinear", "bicubic"],
        default="bilinear",
        help="How the dense head broadcasts each token over its block footprint before "
             "the conv stack. 'nearest' reproduces the pre-fix step-function ablations.",
    )

    # --- Loss ---
    ap.add_argument("--alpha", type=float, default=0.7, help="Positive class weighting for BCE")
    ap.add_argument("--tau", type=float, default=0.1, help="Soft-majority temperature")
    ap.add_argument("--lambda_teacher", type=float, default=1.0)
    ap.add_argument("--lambda_edge", type=float, default=1e-3)
    ap.add_argument(
        "--edge_channel",
        type=int,
        default=-1,
        help="Index of the guidance channel for the edge-aware term (default: last)",
    )

    # --- Training ---
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_dir", default="experiments/dense_001")

    # --- Model shape, must match the detector checkpoint ---
    ap.add_argument("--in_channels", type=int, default=None, help="Defaults to len(channel_vars)")
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp_ratio", type=float, default=2.0)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--attn_drop", type=float, default=0.1)
    ap.add_argument("--posenc", choices=["learned2d", "sincos2d"], default="learned2d")
    ap.add_argument("--sra", action="store_true", default=False)
    ap.add_argument("--sra_ratio", type=int, default=2)

    return ap.parse_args()


def build_model(args) -> BlocksTokenModel:
    in_channels = args.in_channels if args.in_channels is not None else len(args.channel_vars)

    cfg = ModelConfig(
        in_channels=in_channels,
        block_size=args.block,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop,
        attn_drop=args.attn_drop,
        sra=args.sra,
        sra_ratio=args.sra_ratio,
        positional=args.posenc,
        dense_head=True,
        dense_hidden=args.dense_hidden,
        dense_use_cnn_features=not args.no_cnn_features,
        dense_token_upsample=args.dense_token_upsample,
    )

    model = BlocksTokenModel(cfg)
    load_backbone_weights(model, args.detector_ckpt)

    return model


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seeds(args.seed)

    # The split must not move when --seed is varied for the multi-seed protocol,
    # so it gets its own seed that stays fixed across runs.
    split_seed = args.split_seed if args.split_seed is not None else args.seed

    with open(os.path.join(args.save_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump({**vars(args), "resolved_split_seed": split_seed}, f, indent=2)

    use_dirs = all(d is not None for d in (args.train_dir, args.valid_dir, args.test_dir))

    if not use_dirs and args.nc_dir is None:
        raise ValueError(
            "Provide either --nc_dir or all three of --train_dir/--valid_dir/--test_dir."
        )

    if use_dirs:
        train_items, valid_items, test_items = make_nc_block_dataset(
            train_dir=args.train_dir,
            valid_dir=args.valid_dir,
            test_dir=args.test_dir,
            channel_var_names=tuple(args.channel_vars),
            mask_var_name=args.mask_var,
            block_size=(args.block, args.block),
        )
    else:
        train_items, valid_items, test_items = make_nc_block_dataset(
            nc_dir=args.nc_dir,
            channel_var_names=tuple(args.channel_vars),
            mask_var_name=args.mask_var,
            block_size=(args.block, args.block),
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
            seed=split_seed,
        )

    print(
        f"Train files: {len(train_items)} | Valid: {len(valid_items)} | Test: {len(test_items)}"
    )

    collator = DataCollatorBlocks(
        pad_images=True,
        labels_as_float=True,
        make_block_attention_mask=True,
        flatten_block_attention_mask=False,
        enforce_same_block_size=True,
    )

    # --- Augmentation: flip probabilities of 0 (the default) leave items untouched.
    train_transform = None
    if args.hflip_prob > 0 or args.vflip_prob > 0:
        train_transform = RandomFlip(hflip_prob=args.hflip_prob, vflip_prob=args.vflip_prob)

    train_loader, val_loader, test_loader = create_dataloaders_train_val(
        train_items=train_items,
        val_items=valid_items,
        test_items=test_items,
        collate_fn=collator,
        train_batch_size=args.train_bs,
        val_batch_size=args.val_bs,
        test_batch_size=args.val_bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        bucket_by_grid=args.bucket_by_grid,
        seed=args.seed,
        train_transform=train_transform,
    )

    model = build_model(args)
    device = get_device()

    freeze = not args.unfreeze_backbone

    # Stage 2 fine-tuning still distills against the *original* detector, so
    # keep an untouched copy before anything is optimized.
    teacher_model = None
    if not freeze and args.lambda_teacher > 0:
        teacher_model = copy.deepcopy(model)
        teacher_model.dense_head = None

    trainer = DenseTrainer(
        model=model,
        device=device,
        alpha=args.alpha,
        tau=args.tau,
        lambda_teacher=args.lambda_teacher,
        lambda_edge=args.lambda_edge,
        edge_channel=args.edge_channel,
        grad_clip=args.grad_clip,
        threshold=args.threshold,
        save_dir=args.save_dir,
        freeze_backbone=freeze,
        teacher_model=teacher_model,
    )

    trainable = trainer.trainable_parameters()
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | trainable params: {n_trainable:,} / {n_total:,}")

    optimizer, scheduler = build_optim_and_sched(
        model=model, lr=args.lr, weight_decay=args.wd, epochs=args.epochs
    )

    best_pr_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_stats = trainer.train_epoch(train_loader, optimizer)
        val_loss, metrics = trainer.evaluate(val_loader)
        scheduler.step()

        parts = " | ".join(f"{k}={v:.4f}" for k, v in train_stats.items())
        print(
            f"Epoch {epoch:02d} | {parts} | val_loss={val_loss:.4f} | "
            f"dense f1={metrics.get('f1', float('nan')):.4f} "
            f"pr_auc={metrics.get('pr_auc', float('nan')):.4f} || "
            f"detector f1={metrics.get('detector_f1', float('nan')):.4f} "
            f"pr_auc={metrics.get('detector_pr_auc', float('nan')):.4f}"
        )

        ckpt_path = trainer.save_ckpt(
            epoch, optimizer, scheduler, {"val_loss": val_loss, **metrics}
        )

        pr_auc = metrics.get("pr_auc", float("nan"))
        if np.isfinite(pr_auc) and pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            torch.save(torch.load(ckpt_path), os.path.join(args.save_dir, "best.pt"))

    print(f"Done. Best dense PR-AUC: {best_pr_auc:.4f}")


if __name__ == "__main__":
    main()
