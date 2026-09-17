# src/training/finetune_tokens.py
"""
Fine-tune an already-trained token/block CNN->Transformer detector.

This is train_tokens.py with the "start from scratch" step replaced by "load an
existing checkpoint and keep training". Point it at a folder that holds a train
split and a val split; the train split drives the gradient updates and the val
split selects the best checkpoint (validation PR-AUC), exactly as in
train_tokens.py.

By default every parameter is fine-tuned at a low learning rate. Use
--freeze_backbone to adapt only the transformer encoder + token head, or
--head_only to adapt just the final token head.

The architecture is read from the checkpoint itself (BlocksTokenModel embeds its
ModelConfig under "config"), so none of the model-shape flags from train_tokens.py
are needed here. Output checkpoints use the same format, so evaluation/eval.py and
inference/infer.py consume them unchanged.

Example:

    python -m src.training.finetune_tokens \
        --init_ckpt results/cnntrans_gfh_42/best.pt \
        --data_dir /workspace/data/south_china_sea/netcdf \
        --channel_vars ssha_unfiltered ssha_unfiltered_rolling_highpass \
        --epochs 20 --lr 1e-5 \
        --save_dir experiments/finetune_scs_gfh_42
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataio import (
    DataCollatorBlocks,
    RandomFlip,
    create_dataloaders_train_val,
    make_nc_block_dataset,
)
from inference.inference import load_model

from .engine import Trainer


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _FineTuneTrainer(Trainer):
    """Trainer that keeps explicitly frozen modules in eval mode while training.

    The stock ``Trainer.train_epoch`` calls ``model.train()``, which re-enables
    the CNN stem's BatchNorm running-stat updates even when its weights are
    frozen. Any module passed in ``frozen_modules`` is forced back to ``eval()``
    each epoch so its parameters *and* buffers stay put. With nothing frozen this
    behaves exactly like the base class.
    """

    def __init__(self, *args, frozen_modules=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen_modules = list(frozen_modules or [])

    def train_epoch(self, loader, optimizer) -> float:
        self.model.train()
        for module in self._frozen_modules:
            module.eval()
        total, steps = 0.0, 0
        for batch in loader:
            loss, _ = self._step(batch, optimizer=optimizer)
            total += loss
            steps += 1
        return total / max(steps, 1)


def _first_existing_subdir(root: str, names: list[str]) -> str | None:
    for name in names:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path
    return None


def resolve_split_dirs(args) -> tuple[str, str, str]:
    """Return (train_dir, valid_dir, test_dir).

    Explicit --train_dir/--valid_dir win. Otherwise --data_dir is expected to
    contain train/ and val/ (or valid/) subdirectories. make_nc_block_dataset
    requires a test dir too; when there is no test/ subdirectory the val dir is
    reused (the test loader is built but never used here).
    """
    if args.train_dir or args.valid_dir:
        if not (args.train_dir and args.valid_dir):
            raise SystemExit("--train_dir and --valid_dir must be provided together")
        test_dir = args.test_dir or args.valid_dir
        return args.train_dir, args.valid_dir, test_dir

    if not args.data_dir:
        raise SystemExit(
            "Provide --data_dir (a folder with train/ and val/ subdirectories) "
            "or an explicit --train_dir and --valid_dir."
        )

    root = args.data_dir
    if not os.path.isdir(root):
        raise SystemExit(f"--data_dir is not a directory: {root}")

    train_dir = _first_existing_subdir(root, ["train", "training"])
    valid_dir = _first_existing_subdir(root, ["val", "valid", "validation"])
    test_dir = _first_existing_subdir(root, ["test", "testing"]) or valid_dir

    if train_dir is None or valid_dir is None:
        raise SystemExit(
            f"Expected train/ and val/ (or valid/) subdirectories under {root}. "
            f"Found: {sorted(os.listdir(root))}"
        )
    return train_dir, valid_dir, test_dir


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune a trained token-wise CNN->Transformer block detector"
    )

    # --- Starting point ---
    ap.add_argument(
        "--init_ckpt",
        required=True,
        help="Path to the trained checkpoint to fine-tune (e.g. runs/.../best.pt)",
    )

    # --- Data ---
    ap.add_argument(
        "--data_dir",
        default=None,
        help="Folder containing train/ and val/ (or valid/) subdirectories of NetCDF files",
    )
    ap.add_argument("--train_dir", default=None, help="Explicit training split directory")
    ap.add_argument("--valid_dir", default=None, help="Explicit validation split directory")
    ap.add_argument(
        "--test_dir",
        default=None,
        help="Explicit test split directory (optional; defaults to the val directory)",
    )
    ap.add_argument(
        "--channel_vars",
        nargs="+",
        default=["ssha_unfiltered"],
        help="NetCDF variable names to read as channels; must match what the checkpoint was trained on",
    )
    ap.add_argument(
        "--mask_var", default="block_mask", help="NetCDF variable name for block labels"
    )
    ap.add_argument(
        "--block",
        type=int,
        default=None,
        help="Square block size in pixels (default: read from the checkpoint's config)",
    )
    ap.add_argument("--seed", type=int, default=123)

    # --- Dataloaders ---
    ap.add_argument("--train_bs", type=int, default=16)
    ap.add_argument("--val_bs", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true", default=False)
    ap.add_argument("--bucket_by_grid", action="store_true", default=True)

    # --- Augmentation (train split only) ---
    ap.add_argument("--hflip_prob", type=float, default=1.0)
    ap.add_argument("--vflip_prob", type=float, default=1.0)

    # --- Fine-tuning ---
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5, help="Low by design for fine-tuning")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument(
        "--alpha", type=float, default=0.6, help="Positive-class weighting factor for BCE"
    )
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5, help="F1 / PR threshold")
    ap.add_argument("--save_dir", default="experiments/finetune_001")
    ap.add_argument(
        "--save_top_k",
        type=int,
        default=None,
        help="Keep only the N best per-epoch checkpoints on disk, ranked by "
        "validation PR-AUC (best.pt is always kept). Default: keep all.",
    )

    # --- Fine-tune scope (mutually exclusive; default = full model) ---
    ap.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze the CNN stem (and learned positional encoding); adapt only the transformer encoder and token head",
    )
    ap.add_argument(
        "--head_only",
        action="store_true",
        help="Freeze the CNN stem, positional encoding, and transformer encoder; adapt only the final token head",
    )

    ap.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Override the auto device (MPS -> CUDA -> CPU)",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    if args.freeze_backbone and args.head_only:
        raise SystemExit("--freeze_backbone and --head_only are mutually exclusive")

    os.makedirs(args.save_dir, exist_ok=True)
    set_seeds(args.seed)

    train_dir, valid_dir, test_dir = resolve_split_dirs(args)
    device = get_device(args.device)

    # --- Load the checkpoint; architecture comes from its embedded config. ---
    model, cfg = load_model(args.init_ckpt, device)

    n_channels = int(getattr(cfg, "in_channels", len(args.channel_vars)))
    if len(args.channel_vars) != n_channels:
        raise SystemExit(
            f"--channel_vars lists {len(args.channel_vars)} channel(s) "
            f"({args.channel_vars}) but the checkpoint expects in_channels={n_channels}. "
            "Pass exactly the channels the model was trained on, in the same order."
        )
    block = args.block if args.block is not None else int(cfg.block_size)

    # --- Apply the fine-tune scope by freezing whole modules. ---
    # BlocksTokenModel.freeze_backbone() also freezes the encoder and head (it is
    # built for dense-head training), so freeze the pieces directly instead.
    frozen_modules = []
    if args.freeze_backbone:
        frozen_modules = [model.stem]
    elif args.head_only:
        frozen_modules = [model.stem, model.encoder]
    if frozen_modules and getattr(model, "posenc", None) is not None:
        frozen_modules.append(model.posenc)

    for module in frozen_modules:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    if not trainable:
        raise SystemExit("No trainable parameters remain after freezing; loosen --freeze_backbone / --head_only")
    print(
        f"Fine-tuning {n_trainable:,} / {n_total:,} parameters "
        f"({'full model' if n_trainable == n_total else 'partial'}) on {device}"
    )

    # --- Freeze the resolved run config next to the checkpoints. ---
    meta = dict(vars(args))
    meta.update(
        {
            "resolved_train_dir": train_dir,
            "resolved_valid_dir": valid_dir,
            "resolved_test_dir": test_dir,
            "resolved_block": block,
            "resolved_device": str(device),
            "trainable_param_count": n_trainable,
            "total_param_count": n_total,
        }
    )
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- Build dataset splits (explicit-directory mode). ---
    train_items, valid_items, test_items = make_nc_block_dataset(
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        channel_var_names=tuple(args.channel_vars),
        mask_var_name=args.mask_var,
        block_size=(block, block),
    )

    collator = DataCollatorBlocks(
        pad_images=True,
        labels_as_float=True,
        make_block_attention_mask=True,
        flatten_block_attention_mask=False,
        enforce_same_block_size=True,
    )

    train_transform = None
    if args.hflip_prob > 0 or args.vflip_prob > 0:
        train_transform = RandomFlip(
            hflip_prob=args.hflip_prob, vflip_prob=args.vflip_prob
        )

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

    # --- Optimizer & scheduler: only the params still requiring grad. ---
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = _FineTuneTrainer(
        model=model,
        device=device,
        alpha=args.alpha,
        grad_clip=args.grad_clip,
        threshold=args.threshold,
        save_dir=args.save_dir,
        frozen_modules=frozen_modules,
    )

    # --- Fine-tuning loop (selection on validation PR-AUC). ---
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
            epoch,
            optimizer,
            scheduler,
            {"val_loss": val_loss, "finetuned_from": args.init_ckpt, **metrics},
        )
        if metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["pr_auc"]
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
