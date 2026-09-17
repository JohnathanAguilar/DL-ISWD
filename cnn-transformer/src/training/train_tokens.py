# src/training/train_tokens.py
import argparse
import json
import os
import random

import numpy as np
import torch
from dataio import (
    DataCollatorBlocks,
    RandomFlip,
    create_dataloaders_train_val,
    make_nc_block_dataset,
)
from models import BlocksTokenModel, ModelConfig

from .engine import Trainer
from .optim import build_optim_and_sched


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
        description="Train token-wise CNN→Transformer on block labels"
    )

    # --- Data ---
    ap.add_argument(
        "--nc_dir",
        default=None,
        help="Folder of processed NetCDF files to auto-split (mutually "
             "exclusive with --train_dir/--valid_dir/--test_dir)",
    )
    ap.add_argument(
        "--train_dir", default=None, help="Existing training split directory"
    )
    ap.add_argument(
        "--valid_dir", default=None, help="Existing validation split directory"
    )
    ap.add_argument(
        "--test_dir", default=None, help="Existing test split directory"
    )
    ap.add_argument(
        "--channel_vars",
        nargs="+",
        default=["ssha_unfiltered"],
        help="Variable names to read as channels (e.g., SSH + highpass)",
    )
    ap.add_argument(
        "--mask_var", default="block_mask", help="Variable name for block labels"
    )
    ap.add_argument("--block", type=int, default=20, help="Square block size (pixels)")
    ap.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction to hold out for validation/test",
    )
    ap.add_argument("--seed", type=int, default=123)

    # --- Dataloaders ---
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--val_bs", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)  # macOS/Jupyter: start at 0
    ap.add_argument("--pin_memory", action="store_true", default=False)
    ap.add_argument("--bucket_by_grid", action="store_true", default=True)
    ap.add_argument("--drop_last", action="store_true", default=False)

    # --- Augmentation ---
    ap.add_argument(
        "--hflip_prob",
        type=float,
        default=1.0,
        help="Probability of a horizontal flip applied to training items only (0 = disabled)",
    )
    ap.add_argument(
        "--vflip_prob",
        type=float,
        default=1.0,
        help="Probability of a vertical flip applied to training items only (0 = disabled)",
    )

    # --- Training ---
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Positive class weighting factor for BCE",
    )
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5, help="F1 / PR threshold")
    ap.add_argument("--save_dir", default="experiments/exp_001")
    ap.add_argument(
        "--save_top_k",
        type=int,
        default=None,
        help="Keep only the N best per-epoch checkpoints on disk, ranked by "
             "validation PR-AUC (best.pt is always kept regardless). Default: keep all.",
    )

    # --- Model ---
    ap.add_argument("--in_channels", type=int, default=1)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--mlp_ratio", type=float, default=3.0)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--attn_drop", type=float, default=0.1)
    ap.add_argument("--posenc", choices=["learned2d", "sincos2d"], default="sincos2d")
    ap.add_argument("--sra", action="store_true", default=True)
    ap.add_argument("--sra_ratio", type=int, default=2)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    set_seeds(args.seed)

    # --- Build dataset splits
    # Dataset helper returns three lists of item dicts (train, valid, test).
    train_items, valid_items, test_items = make_nc_block_dataset(
        nc_dir=args.nc_dir,
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        test_dir=args.test_dir,
        channel_var_names=tuple(args.channel_vars),
        mask_var_name=args.mask_var,
        block_size=(args.block, args.block),
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # --- Collator: pads & builds block-level attention mask that excludes padded tiles in loss/attention. [1](https://jpl365prod-my.sharepoint.com/personal/johnathan_g_aguilar_jpl_nasa_gov/Documents/Microsoft%20Copilot%20Chat%20Files/loader.py)
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
        train_transform = RandomFlip(
            hflip_prob=args.hflip_prob, vflip_prob=args.vflip_prob
        )

    # --- DataLoaders: bucket_by_grid groups by identical (Bh,Bw) to reduce padding. [2](https://jpl365prod-my.sharepoint.com/personal/johnathan_g_aguilar_jpl_nasa_gov/Documents/Microsoft%20Copilot%20Chat%20Files/collator.py)
    train_loader, val_loader, _test_loader = create_dataloaders_train_val(
        train_items=train_items,
        val_items=valid_items,
        test_items=test_items,
        collate_fn=collator,
        train_batch_size=args.train_bs,
        val_batch_size=args.val_bs,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        bucket_by_grid=args.bucket_by_grid,
        seed=args.seed,
        train_transform=train_transform,
    )

    # --- Model (CNN→blocks→posenc→Transformer→token head), sized per your baseline. [3](https://jpl365prod-my.sharepoint.com/personal/johnathan_g_aguilar_jpl_nasa_gov/Documents/Microsoft%20Copilot%20Chat%20Files/dataset.py)
    cfg = ModelConfig(
        in_channels=args.in_channels,
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
    )
    model = BlocksTokenModel(cfg)
    device = get_device()

    # --- Optimizer & scheduler (AdamW + cosine) per your plan. [3](https://jpl365prod-my.sharepoint.com/personal/johnathan_g_aguilar_jpl_nasa_gov/Documents/Microsoft%20Copilot%20Chat%20Files/dataset.py)
    optimizer, scheduler = build_optim_and_sched(
        model=model, lr=args.lr, weight_decay=args.wd, epochs=args.epochs
    )

    # --- Trainer handles masking, loss, metrics, checkpoints
    trainer = Trainer(
        model=model,
        device=device,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        threshold=args.threshold,
        save_dir=args.save_dir,
    )

    # --- Training loop
    best_pr_auc = -1.0
    kept_ckpts: list[tuple[float, str]] = []  # (pr_auc, path) for --save_top_k pruning
    for epoch in range(1, args.epochs + 1):
        tr_loss = trainer.train_epoch(train_loader, optimizer)
        val_loss, metrics = trainer.evaluate(val_loader)
        scheduler.step()

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
            f"pr_auc={metrics['pr_auc']:.4f} | f1@{args.threshold}={metrics['f1']:.4f}"
        )

        ckpt_path = trainer.save_ckpt(
            epoch, optimizer, scheduler, {"val_loss": val_loss, **metrics}
        )
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
            # Save a copy as best.pt
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(torch.load(ckpt_path), best_path)

        if args.save_top_k is not None:
            kept_ckpts.append((metrics["pr_auc"], ckpt_path))
            if len(kept_ckpts) > args.save_top_k:
                kept_ckpts.sort(key=lambda item: item[0])
                _, worst_path = kept_ckpts.pop(0)
                if os.path.exists(worst_path):
                    os.remove(worst_path)

    print(f"Done. Best PR-AUC: {best_pr_auc:.4f}")


if __name__ == "__main__":
    main()

