"""Train a Faster R-CNN internal-wave detector on the YOLO-format SWOT chips.

Example
-------
    python src/training/train.py --name frcnn_frgb

Outputs land in ``runs/<name>/``:
    weights/best.pt   checkpoint with the highest validation mAP50
    weights/last.pt   checkpoint from the final epoch
    results.csv       per-epoch losses and validation metrics
    results.png       loss / metric curves
    args.json         the exact configuration used
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC_DIR = Path(__file__).resolve().parent.parent
_MODEL_DIR = _SRC_DIR.parent

for _p in (_SRC_DIR, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from configs import config
except ImportError:
    import config

try:
    from dataio.dataset import build_dataset, collate_fn
except ImportError:
    from dataset import build_dataset, collate_fn

try:
    from training.engine import (
        gradient_backend_is_sane,
        predict_dataset,
        train_one_epoch,
        warmup_lr_scheduler,
    )
except ImportError:
    from engine import (
        gradient_backend_is_sane,
        predict_dataset,
        train_one_epoch,
        warmup_lr_scheduler,
    )

try:
    from training.metrics import ap_metrics, fixed_threshold_metrics
except ImportError:
    from metrics import ap_metrics, fixed_threshold_metrics

try:
    from models.model import build_model, resolve_device
except ImportError:
    from model import build_model, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data", default=str(config.DEFAULT_DATA_YAML))
    parser.add_argument("--name", default="frcnn_frgb")
    parser.add_argument("--project", default=str(config.DEFAULT_RUNS_DIR))

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default="auto",
        help="auto | mps | cuda | cpu | a bare GPU index like 0 or 1 (-> cuda:0 / cuda:1)",
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--backbone",
        default="resnet50_fpn",
        choices=["resnet50_fpn_v2", "resnet50_fpn", "mobilenet_v3_large_fpn"],
        help="resnet50_fpn is the canonical Faster R-CNN baseline and ~3x "
             "faster to train here than the v2 variant.",
    )
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument(
        "--default-anchors",
        action="store_true",
        help="Use torchvision's square anchors instead of the tall SWOT anchors.",
    )

    parser.add_argument("--imgsz", type=int, default=config.MIN_SIZE,
                        help="Long-side resize target, matching YOLO's imgsz.")
    parser.add_argument("--nms-iou", type=float, default=config.NMS_IOU)

    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=float, default=1.0,
                        help="Linear LR warmup, capped at one epoch.")
    parser.add_argument("--lr-schedule", default="cosine", choices=["cosine", "step", "none"])
    parser.add_argument("--lr-final-factor", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=10.0)

    # Augmentation -- all disabled by default. The training set was already
    # augmented offline, so no online augmentation is applied here.
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.0)

    parser.add_argument("--val-conf", type=float, default=0.05,
                        help="Confidence used for the per-epoch P/R/F1 readout.")
    parser.add_argument("--eval-iou", type=float, default=config.EVAL_IOU)
    parser.add_argument("--patience", type=int, default=0,
                        help="Early-stop after N epochs without mAP50 improvement (0 = off).")
    parser.add_argument("--skip-device-check", action="store_true",
                        help="Skip the pre-training gradient sanity check.")

    return parser.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]

    if args.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )

    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.lr_schedule == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_final_factor
        )

    if args.lr_schedule == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)],
            gamma=0.1,
        )

    return None


def plot_results(history, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    loss_keys = [
        k for k in history[0] if k.startswith("loss_") or k == "train_loss"
    ]
    for key in loss_keys:
        axes[0].plot(epochs, [row.get(key, np.nan) for row in history], label=key)
    axes[0].set_title("training losses")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    for key in ("val_mAP50", "val_mAP50-95"):
        axes[1].plot(epochs, [row.get(key, np.nan) for row in history], label=key)
    axes[1].set_title("validation AP")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    for key in ("val_precision", "val_recall", "val_f1"):
        axes[2].plot(epochs, [row.get(key, np.nan) for row in history], label=key)
    axes[2].set_title("validation P / R / F1")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_csv(history, out_path):
    import csv

    fieldnames = list(history[0].keys())

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"device: {device}")

    run_dir = Path(args.project) / args.name
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    aug = dict(
        fliplr=args.fliplr,
        flipud=args.flipud,
        scale=args.scale,
        translate=args.translate,
    )

    train_set, class_names = build_dataset(args.data, "train", train=True, **aug)
    val_set, _ = build_dataset(args.data, "val", train=False)

    print(
        f"train: {len(train_set)} images | val: {len(val_set)} images | "
        f"classes: {class_names}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )

    model = build_model(
        num_classes=len(class_names) + 1,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        min_size=args.imgsz,
        max_size=args.imgsz,
        nms_iou=args.nms_iou,
        box_score_thresh=config.CACHE_CONF,
        trainable_backbone_layers=args.trainable_backbone_layers,
        custom_anchors=not args.default_anchors,
    ).to(device)

    if not args.skip_device_check:
        sane, norms = gradient_backend_is_sane(
            model, next(iter(val_loader)), device
        )

        if not sane:
            message = (
                f"Gradient sanity check failed on {device}: gradient norms "
                + ", ".join(f"{n:.3g}" for n in norms)
                + ". This backend is not computing usable Faster R-CNN "
                "gradients (see README, 'Apple MPS')."
            )

            if args.device == "auto" and device.type == "mps":
                print(message + "\nFalling back to CPU.")
                device = torch.device("cpu")
                model.to(device)
            else:
                raise RuntimeError(message + " Rerun with --device cpu.")
        else:
            print(
                "gradient check ok (norms "
                + ", ".join(f"{n:.3g}" for n in norms)
                + ")"
            )

    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    warmup_iters = min(int(args.warmup_epochs * len(train_loader)), len(train_loader))
    warmup_scheduler = (
        warmup_lr_scheduler(optimizer, warmup_iters, 1e-3) if warmup_iters > 0 else None
    )

    with open(run_dir / "args.json", "w") as f:
        json.dump(
            {**vars(args), "device": str(device), "class_names": class_names},
            f,
            indent=2,
        )

    history = []
    best_map50 = -1.0
    best_epoch = -1
    class_ids = tuple(range(1, len(class_names) + 1))

    for epoch in range(args.epochs):
        losses = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch=epoch,
            epochs=args.epochs,
            warmup_scheduler=warmup_scheduler if epoch == 0 else None,
            clip_grad_norm=args.clip_grad_norm,
        )

        if scheduler is not None:
            scheduler.step()

        predictions, targets = predict_dataset(
            model, val_loader, device, desc=f"val {epoch + 1}/{args.epochs}"
        )

        ap = ap_metrics(predictions, targets, class_ids=class_ids)
        fixed = fixed_threshold_metrics(
            predictions, targets, conf=args.val_conf, iou_thr=args.eval_iou, beta=1.0
        )

        row = {
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": losses["total"],
            **{k: v for k, v in losses.items() if k.startswith("loss_")},
            "epoch_seconds": losses["epoch_seconds"],
            "skipped_steps": losses["skipped_steps"],
            "val_mAP50": ap["mAP50"],
            "val_mAP50-95": ap["mAP50-95"],
            "val_precision": fixed["precision"],
            "val_recall": fixed["recall"],
            "val_f1": fixed["f1"],
            "val_conf": args.val_conf,
        }
        history.append(row)

        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"loss {row['train_loss']:.4f}  "
            f"mAP50 {ap['mAP50']:.4f}  mAP50-95 {ap['mAP50-95']:.4f}  "
            f"P {fixed['precision']:.3f}  R {fixed['recall']:.3f}  "
            f"F1 {fixed['f1']:.3f}  ({row['epoch_seconds']:.0f}s)"
        )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "args": vars(args),
            "class_names": class_names,
            "metrics": row,
        }

        torch.save(checkpoint, weights_dir / "last.pt")

        if ap["mAP50"] > best_map50:
            best_map50 = ap["mAP50"]
            best_epoch = epoch + 1
            torch.save(checkpoint, weights_dir / "best.pt")

        write_csv(history, run_dir / "results.csv")
        plot_results(history, run_dir / "results.png")

        if args.patience and (epoch + 1) - best_epoch >= args.patience:
            print(f"early stop: no mAP50 improvement for {args.patience} epochs")
            break

    print(f"\nbest mAP50 {best_map50:.4f} at epoch {best_epoch}")
    print(f"weights: {weights_dir / 'best.pt'}")

    return run_dir


if __name__ == "__main__":
    main()
