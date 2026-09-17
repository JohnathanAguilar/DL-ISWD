#!/usr/bin/env python3
# src/training/train_dense_2.0.py
"""
CLI port of ``training_dense_head.ipynb``.

Trains the dense pixel-resolution refinement head on top of a frozen patch-level
detector, then runs the full WACV evaluation from the notebook and, optionally,
the 5-variant x N-seed ablation sweep:

  * train the dense head (frozen backbone by default; ``--unfreeze-backbone`` for
    the Stage 2 end-to-end variant, which keeps a frozen distillation teacher);
  * verify the detector is reproduced bit-for-bit before training;
  * select a reporting threshold on the validation split;
  * score the learned dense map against the frozen detector and the
    nearest / bilinear / bicubic / gaussian interpolation baselines;
  * structural metrics (grid-artifact, image-structure alignment, equivariance,
    localized perturbation), a qualitative figure, per-sample structural CSV, and
    a pass-level paired bootstrap CI;
  * ``--run-ablations`` sweeps the notebook's five dense-head variants over
    ``--ablation-seeds`` and writes a mean +/- std summary.

Every artifact the notebook produces -- including the figures it only shows
inline -- is written under a single ``--out-dir``::

    <out-dir>/
      run.log            full stdout/stderr transcript
      args.json          resolved arguments + device
      summary.json       roll-up: artifact paths + headline numbers
      checkpoints/       checkpoint_epoch_XX.pt, best.pt
      metrics/           *.json  (preservation, history, threshold, eval, bootstrap, ...)
      tables/            ablation_table.md, per_sample_structural.csv, seed_summary.md
      figures/           loss_curves.png, detection_fidelity.png, threshold_sweep.png,
                         qualitative_comparison.png, per_sample_structural_hist.png
      ablations/<v>_s<seed>/   per-variant checkpoints + eval_results.json

The ``2.0`` in the filename makes ``python -m src.training.train_dense_2.0``
impossible, so run it directly (from ``cnn-transformer/``)::

    python src/training/train_dense_2.0.py --help
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback

import matplotlib

matplotlib.use("Agg")  # headless: every figure is saved to disk, never shown

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# --- make the sibling packages importable without the editable install --------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dataio import (  # noqa: E402
    DataCollatorBlocks,
    create_dataloaders_train_val,
    make_nc_block_dataset,
)
from models import ModelConfig, BlocksTokenModel  # noqa: E402
from inference.inference import (  # noqa: E402
    infer_model_config_from_checkpoint,
    load_backbone_weights,
)
from training import DenseTrainer  # noqa: E402
from training.optim import build_optim_and_sched  # noqa: E402
from training.metrics import compute_metrics  # noqa: E402
from evaluation.dense_eval import (  # noqa: E402
    evaluate_dense_loader,
    upsample_patch_probs,
    grid_artifact_ratio,
    image_structure_alignment,
    format_ablation_table,
    collect_aggregated_scores,
    select_best_threshold,
    THRESHOLD_SELECTABLE_METRICS,
)
from evaluation.viz import plot_dense_comparison  # noqa: E402


ABLATION_VARIANTS = (
    "patch_only",
    "cnn_feats",
    "cnn_edge",
    "no_teacher",
    "bilinear_broadcast",
)

DETECTION_KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
STRUCT_KEYS = [
    "grid_artifact_fraction",
    "grid_artifact_ratio",
    "alignment_ratio",
    "alignment_corr",
    "equivariance_mae",
    "perturbation_ratio",
]


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

class _Tee:
    """Write to several streams at once (stdout/stderr + the run log)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref=None) -> torch.device:
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def dump_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def args_to_jsonable(args) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}


def _checkpoint_state_dict(ckpt):
    for key in ("model_state", "model_state_dict", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt


def _safe(stage: str, fn, *a, **kw):
    """Run an optional evaluation stage; log and continue if it raises."""
    try:
        return fn(*a, **kw)
    except Exception:
        print(f"\n[WARNING] stage {stage!r} failed; continuing.\n{traceback.format_exc()}")
        return None


def _list_artifacts(out: str) -> dict:
    found = {}
    for root, _, files in os.walk(out):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, out)
            if rel == "summary.json":
                continue
            found[rel] = os.path.getsize(p)
    return found


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Train + fully evaluate the dense refinement head "
                    "(CLI port of training_dense_head.ipynb).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data (same pass-level grouped split as the patch-level paper) --------
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--valid-dir", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--channel-vars", nargs="+", default=["rolling_mean_highpass"])
    ap.add_argument("--mask-var", default="block_mask")
    ap.add_argument("--block-h", type=int, default=20)
    ap.add_argument("--block-w", type=int, default=20)

    # --- Detector to refine -------------------------------------------------
    ap.add_argument("--detector-ckpt", required=True,
                    help="Trained patch-level checkpoint, e.g. models/rolling_mean_highpass/best.pt")
    ap.add_argument("--unfreeze-backbone", action="store_true",
                    help="Stage 2: fine-tune the backbone end to end (a frozen copy is "
                         "kept as the distillation teacher).")

    # --- Model fields that CANNOT be inferred from checkpoint weights -------
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--sra", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sra-ratio", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--attn-drop", type=float, default=0.1)

    # --- Dense head -------------------------------------------------------
    ap.add_argument("--dense-hidden", type=int, default=64)
    ap.add_argument("--cnn-features", action=argparse.BooleanOptionalAction, default=True,
                    help="Full-resolution CNN features in the dense head "
                         "(--no-cnn-features = transformer-only ablation).")
    ap.add_argument("--dense-token-upsample", choices=["nearest", "bilinear", "bicubic"],
                    default="bilinear")

    # --- Loss ----------------------------------------------------------------
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--lambda-teacher", type=float, default=1.0)
    ap.add_argument("--lambda-edge", type=float, default=1e-3)
    ap.add_argument("--edge-channel", type=int, default=-1)

    # --- Training ----------------------------------------------------------
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--val-bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--bucket-by-grid", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--best-metric", default="f1")
    ap.add_argument("--seed", type=int, default=123)

    # --- Evaluation ------------------------------------------------------
    ap.add_argument("--test-metric", default="f1",
                    help=f"Threshold-selectable metric; one of {THRESHOLD_SELECTABLE_METRICS}")
    ap.add_argument("--eval-baselines", nargs="+",
                    default=["nearest", "bilinear", "bicubic", "gaussian"],
                    help="Interpolation baselines drawn in the qualitative figure.")
    ap.add_argument("--max-structural-batches", type=int, default=None,
                    help="Cap the batches used for structural metrics (None = all).")
    ap.add_argument("--flip-dims", nargs="+", type=int, default=[-1])
    ap.add_argument("--bootstrap-n", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--skip-preservation-check", action="store_true",
                    help="Downgrade the bit-exact detector-preservation assert to a warning.")

    # --- Ablations -----------------------------------------------------
    ap.add_argument("--run-ablations", action="store_true",
                    help="After the main run, sweep the dense-head variants over --ablation-seeds.")
    ap.add_argument("--ablation-seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--ablation-variants", nargs="+", choices=list(ABLATION_VARIANTS),
                    default=None, help="Subset of variants to run (default: all five).")
    ap.add_argument("--ablation-epochs", type=int, default=None,
                    help="Epochs per ablation run (default: --epochs).")

    # --- Output ---------------------------------------------------------
    ap.add_argument("--out-dir", required=True,
                    help="Root directory for every checkpoint, metric, table and figure.")
    ap.add_argument("--device", default=None, help="Force a torch device (default: MPS>CUDA>CPU).")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Build data / model  (notebook cells 6, 8, 10)
# ---------------------------------------------------------------------------

def build_dataloaders(args):
    train_items, valid_items, test_items = make_nc_block_dataset(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        test_dir=args.test_dir,
        channel_var_names=tuple(args.channel_vars),
        mask_var_name=args.mask_var,
        block_size=(args.block_h, args.block_w),
    )
    print(f"Train files: {len(train_items)}")
    print(f"Valid files: {len(valid_items)}")
    print(f"Test files:  {len(test_items)}")

    collator = DataCollatorBlocks(
        pad_images=True,
        labels_as_float=True,
        make_block_attention_mask=True,
        flatten_block_attention_mask=False,
        enforce_same_block_size=True,
    )

    train_loader, valid_loader, test_loader = create_dataloaders_train_val(
        train_items=train_items,
        val_items=valid_items,
        test_items=test_items,
        collate_fn=collator,
        train_batch_size=args.train_bs,
        val_batch_size=args.val_bs,
        test_batch_size=args.val_bs,
        num_workers=0,
        pin_memory=False,
        bucket_by_grid=args.bucket_by_grid,
        seed=args.seed,
    )
    return (
        train_loader, valid_loader, test_loader,
        len(train_items), len(valid_items), len(test_items),
    )


def build_model(args, device):
    # Architecture is read straight off the checkpoint tensors. in_channels,
    # embed_dim, depth, mlp_ratio and positional are recoverable; heads,
    # block_size, sra, sra_ratio, drop and attn_drop are not, so they come from
    # args and must match the detector's training run.
    inferred = infer_model_config_from_checkpoint(
        args.detector_ckpt,
        block_size=args.block_h,
        num_heads=args.heads,
        drop=args.drop,
        attn_drop=args.attn_drop,
        sra=args.sra,
        sra_ratio=args.sra_ratio,
    )
    print("Inferred from checkpoint:")
    for k, v in inferred.items():
        print(f"  {k:14s} {v}")

    cfg = ModelConfig(
        **inferred,
        dense_head=True,
        dense_hidden=args.dense_hidden,
        dense_use_cnn_features=args.cnn_features,
        dense_token_upsample=args.dense_token_upsample,
    )

    model = BlocksTokenModel(cfg)

    # Loads every backbone tensor; the dense head stays at its random init.
    _, report = load_backbone_weights(model, args.detector_ckpt)
    missing_non_dense = [k for k in report.missing_keys if not k.startswith("dense_head.")]
    assert not missing_non_dense, missing_non_dense

    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_dense = sum(p.numel() for p in model.dense_head.parameters())
    print(f"\nLoaded detector. Total params {n_total:,} | dense head {n_dense:,} "
          f"({100 * n_dense / n_total:.1f}%)")
    return model, inferred, cfg


def verify_detector_preserved(model, inferred, args, valid_loader, device, out):
    """Notebook cell 12: the dense-head model must reproduce the detector's patch
    probabilities before anything is trained, otherwise the "did not degrade
    detection" claim is unsupported."""
    baseline_model = BlocksTokenModel(ModelConfig(**inferred))
    _ck = torch.load(args.detector_ckpt, map_location="cpu", weights_only=False)
    baseline_model.load_state_dict(_checkpoint_state_dict(_ck))
    baseline_model = baseline_model.to(device).eval()
    model.eval()

    _b = next(iter(valid_loader))
    _img = _b["images"].to(device)
    _bm = _b.get("block_attn_mask", _b["valid"]).to(device).bool()

    with torch.no_grad():
        _p_ref = baseline_model(_img, block_mask=_bm)["probs"]
        _p_new = model(_img, block_mask=_bm)["probs"]

    delta = float((_p_ref - _p_new).abs().max())
    preserved = delta < 1e-6
    print(f"max |original detector - dense model| patch probability: {delta:.3e}")

    dump_json(
        os.path.join(out, "metrics", "detector_preservation.json"),
        {"max_abs_delta": delta, "tolerance": 1e-6, "preserved": preserved},
    )

    del baseline_model, _ck

    if not preserved:
        msg = (f"Dense model does not reproduce the detector (delta {delta:.3e} >= 1e-6); "
               "check the --heads / --block-h / --sra / --sra-ratio overrides.")
        if args.skip_preservation_check:
            print("WARNING: " + msg)
        else:
            raise AssertionError(msg)
    else:
        print("Detector preserved exactly.")


# ---------------------------------------------------------------------------
# Training loop  (notebook cell 16)
# ---------------------------------------------------------------------------

def train(model, trainer, optimizer, scheduler, loaders, args, out):
    train_loader, valid_loader, _ = loaders
    ckpt_dir = os.path.join(out, "checkpoints")
    best_path = os.path.join(ckpt_dir, "best.pt")

    best_metric_value = float("-inf")
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_stats = trainer.train_epoch(train_loader, optimizer)
        scheduler.step()

        should_evaluate = (epoch % args.eval_every == 0 or epoch == args.epochs)
        if not should_evaluate:
            parts = " | ".join(f"{k}={v:.4f}" for k, v in train_stats.items())
            print(f"Epoch {epoch:03d} | {parts} | validation=skipped | "
                  f"time={time.time() - t0:.1f}s")
            continue

        val_loss, metrics = trainer.evaluate(valid_loader)

        if args.best_metric not in metrics:
            raise KeyError(
                f"Metric {args.best_metric!r} was not returned by DenseTrainer.evaluate(). "
                f"Available: {sorted(metrics.keys())}"
            )

        current = float(metrics[args.best_metric])
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:03d} | "
            f"loss={train_stats['loss']:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"DENSE roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f} "
            f"p={metrics['precision']:.4f} r={metrics['recall']:.4f} f1={metrics['f1']:.4f} | "
            f"DETECTOR pr_auc={metrics['detector_pr_auc']:.4f} f1={metrics['detector_f1']:.4f} | "
            f"time={elapsed:.1f}s"
        )

        history.append({"epoch": epoch, "val_loss": val_loss, **train_stats, **metrics})

        ckpt_path = trainer.save_ckpt(
            epoch=epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            extra_metrics={
                "val_loss": val_loss,
                "best_metric_name": args.best_metric,
                "best_metric_value": current,
                **train_stats,
                **metrics,
            },
        )

        if np.isfinite(current) and current > best_metric_value:
            best_metric_value = current
            best_epoch = epoch

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            checkpoint["best_metric_name"] = args.best_metric
            checkpoint["best_metric_value"] = best_metric_value
            checkpoint["best_epoch"] = best_epoch
            torch.save(checkpoint, best_path)

            print(f"  -> Saved new best model: {best_path} | epoch={epoch} | "
                  f"{args.best_metric}={best_metric_value:.4f}")

    dump_json(os.path.join(out, "metrics", "training_history.json"), history)
    dump_json(
        os.path.join(out, "metrics", "best_checkpoint_info.json"),
        {
            "best_epoch": best_epoch,
            "best_metric_name": args.best_metric,
            "best_metric_value": (None if best_epoch is None else best_metric_value),
            "best_path": (best_path if best_epoch is not None else None),
            "epochs_run": args.epochs,
        },
    )

    if best_epoch is None:
        print(f"Done. No valid best model was selected because {args.best_metric} "
              f"was NaN or infinite on every evaluation epoch.")
    else:
        print(f"Done. Best {args.best_metric}={best_metric_value:.4f} at epoch {best_epoch}. "
              f"Saved to {best_path}")

    return history, (best_path if best_epoch is not None else None), best_epoch, best_metric_value


def plot_training_curves(history, out):
    """Notebook cell 17, saved instead of shown."""
    if not history:
        print("No evaluation epochs recorded; skipping training curves.")
        return

    fig_dir = os.path.join(out, "figures")
    ep = [h["epoch"] for h in history]

    plt.figure(figsize=(7, 4))
    for key in ("loss", "loss_patch", "loss_teacher", "loss_edge"):
        if key in history[0]:
            plt.plot(ep, [h[key] for h in history], marker="o", label=key)
    plt.plot(ep, [h["val_loss"] for h in history], marker="s", ls="--", label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Dense refinement loss components")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(ep, [h["pr_auc"] for h in history], marker="o", label="dense pr_auc")
    plt.plot(ep, [h["f1"] for h in history], marker="o", label="dense f1")
    plt.axhline(history[-1]["detector_pr_auc"], ls="--", c="k", label="detector pr_auc")
    plt.axhline(history[-1]["detector_f1"], ls=":", c="k", label="detector f1")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Detection fidelity vs. frozen detector")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "detection_fidelity.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Wrote loss_curves.png and detection_fidelity.png to {fig_dir}")


def load_best(model, best_path, device):
    """Notebook cell 19: never evaluate the last epoch by accident."""
    _best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(_best["model_state"])
    model = model.to(device).eval()
    print(f"Loaded {best_path} (epoch {_best.get('best_epoch')}, "
          f"{_best.get('best_metric_name')}={_best.get('best_metric_value')})")
    return model


# ---------------------------------------------------------------------------
# Threshold selection  (notebook cell 21)
# ---------------------------------------------------------------------------

def select_threshold(model, valid_loader, device, args, out):
    if args.test_metric not in THRESHOLD_SELECTABLE_METRICS:
        raise ValueError(
            f"--test-metric {args.test_metric!r} is not threshold-selectable. "
            f"Choose one of: {THRESHOLD_SELECTABLE_METRICS}"
        )

    val_labels, val_scores = collect_aggregated_scores(model, valid_loader, device)
    sweep = select_best_threshold(val_labels, val_scores, args.test_metric)

    n_pos = int(val_labels.sum())
    print(f"Selected threshold={sweep['threshold']:.3f} maximizing validation "
          f"{args.test_metric}={sweep['value']:.4f} "
          f"({n_pos} positive / {val_labels.numel()} total valid blocks)")

    _t = [s["threshold"] for s in sweep["sweep"]]
    _v = [s[args.test_metric] for s in sweep["sweep"]]
    plt.figure(figsize=(6, 3.5))
    plt.plot(_t, _v)
    plt.axvline(sweep["threshold"], color="r", ls="--",
                label=f"selected = {sweep['threshold']:.3f}")
    plt.xlabel("threshold")
    plt.ylabel(args.test_metric)
    plt.title(f"Validation {args.test_metric} vs. threshold")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "threshold_sweep.png"), dpi=160, bbox_inches="tight")
    plt.close()

    dump_json(
        os.path.join(out, "metrics", "threshold_selection.json"),
        {
            "selected_threshold": float(sweep["threshold"]),
            "selected_value": float(sweep["value"]),
            "metric": args.test_metric,
            "n_positive_blocks": n_pos,
            "n_total_valid_blocks": int(val_labels.numel()),
            "sweep": sweep["sweep"],
        },
    )
    return float(sweep["threshold"])


# ---------------------------------------------------------------------------
# Evaluation  (notebook cells 23, 24, 26, 27)
# ---------------------------------------------------------------------------

def run_evaluation(model, test_loader, device, args, out):
    results = evaluate_dense_loader(
        model=model,
        loader=test_loader,
        device=device,
        threshold=args.threshold,
        edge_channel=args.edge_channel,
        include_baselines=True,
        run_equivariance=True,
        run_perturbation=True,
        flip_dims=tuple(args.flip_dims),
        max_structural_batches=args.max_structural_batches,
    )

    print(f"Evaluated {results['n_batches']} batches, "
          f"{results['n_valid_blocks']:,} valid blocks\n")

    print("=" * 78)
    print("1. DETECTION FIDELITY  (dense map aggregated back to blocks)")
    print("=" * 78)
    print(f"{'method':<12}" + "".join(f"{k:>12}" for k in DETECTION_KEYS))
    for name, m in results["methods"].items():
        print(f"{name:<12}" + "".join(f"{m.get(k, float('nan')):>12.4f}" for k in DETECTION_KEYS))

    print("\n" + "=" * 96)
    print("3-6. STRUCTURAL METRICS")
    print("=" * 96)
    print(f"{'method':<12}" + "".join(f"{k:>22}" for k in STRUCT_KEYS))
    for name, m in results["methods"].items():
        row = f"{name:<12}"
        for k in STRUCT_KEYS:
            v = m.get(k)
            row += f"{'':>22}" if v is None else f"{v:>22.4f}"
        print(row)
    print("\nLower is better: grid_artifact_*, equivariance_mae")
    print("Higher is better: alignment_*, perturbation_ratio")

    # --- Reference row: the detector's own patch predictions, unaggregated ----
    _probs, _labels = [], []
    with torch.no_grad():
        for b in test_loader:
            _img = b["images"].to(device)
            _v = b["valid"].to(device).bool()
            _bm = b.get("block_attn_mask", b["valid"]).to(device).bool()
            _o = model(_img, block_mask=_bm)
            _Bh, _Bw = int(_o["Bh"]), int(_o["Bw"])
            _probs.append(_o["probs"].reshape(_img.shape[0], _Bh, _Bw)[_v])
            _labels.append(b["labels"].to(device)[_v])

    detector_row = compute_metrics(
        labels_v=torch.cat(_labels), probs_v=torch.cat(_probs), threshold=args.threshold
    )
    n_pos = int(torch.cat(_labels).sum())
    n_all = int(torch.cat(_labels).numel())

    print("\nOriginal detector (patch head, no aggregation):")
    print("".join(f"{k:>12}" for k in DETECTION_KEYS))
    print("".join(f"{detector_row[k]:>12.4f}" for k in DETECTION_KEYS))
    print(f"\nPositive blocks in evaluation set: {n_pos:,} of {n_all:,}")

    learned_f1 = results["methods"]["learned"].get("f1", float("nan"))
    f1_delta = None
    if n_pos == 0:
        verdict = "no positive blocks; no fidelity conclusion can be drawn"
        print("WARNING: " + verdict)
    elif not np.isfinite(detector_row["f1"]) or not np.isfinite(learned_f1):
        verdict = "F1 is not finite; cannot compare"
        print("WARNING: " + verdict)
    else:
        f1_delta = float(learned_f1 - detector_row["f1"])
        verdict = ("preserved" if abs(f1_delta) < 0.01
                   else "detection quality moved by more than 0.01 F1")
        print(f"F1 delta (learned dense - detector): {f1_delta:+.4f}")
        print("Dense refinement preserved detection quality."
              if abs(f1_delta) < 0.01 else
              "NOTE: detection quality moved by more than 0.01 F1; investigate before reporting.")

    dump_json(
        os.path.join(out, "metrics", "detection_fidelity.json"),
        {
            "detector_patch_head": detector_row,
            "learned_dense_f1": (None if not np.isfinite(learned_f1) else float(learned_f1)),
            "f1_delta_learned_minus_detector": f1_delta,
            "verdict": verdict,
            "n_positive_blocks": n_pos,
            "n_valid_blocks": n_all,
            "threshold": args.threshold,
        },
    )

    # --- Markdown ablation table + full results dump -------------------------
    table = format_ablation_table({
        "Nearest": results["methods"]["nearest"],
        "Bilinear": results["methods"]["bilinear"],
        "Bicubic": results["methods"]["bicubic"],
        "Gaussian": results["methods"]["gaussian"],
        "Learned (CNN + Transformer)": results["methods"]["learned"],
    })
    print("\n" + table)
    with open(os.path.join(out, "tables", "ablation_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")

    dump_json(
        os.path.join(out, "metrics", "eval_results.json"),
        {
            "methods": results["methods"],
            "detector_patch_head": detector_row,
            "n_valid_blocks": results["n_valid_blocks"],
            "threshold": args.threshold,
            "args": args_to_jsonable(args),
        },
    )
    print(f"\nWrote ablation_table.md and eval_results.json under {out}")
    return results, detector_row, f1_delta, verdict


# ---------------------------------------------------------------------------
# Qualitative figure  (notebook cell 29)
# ---------------------------------------------------------------------------

def qualitative_figure(model, test_loader, device, args, out):
    def dense_panels_for_batch(batch, sample_i=0):
        imgs = batch["images"].to(device)
        bm = batch.get("block_attn_mask", batch["valid"]).to(device).bool()

        with torch.no_grad():
            o = model(imgs, block_mask=bm, return_dense=True)

        Bh, Bw = int(o["Bh"]), int(o["Bw"])
        bs = args.block_h
        Ht, Wt = Bh * bs, Bw * bs

        patch = o["probs"].reshape(imgs.shape[0], Bh, Bw)

        panels = {}
        for mode in args.eval_baselines:
            panels[mode.title()] = upsample_patch_probs(patch, bs, mode)[sample_i, 0].cpu().numpy()
        panels["Learned"] = o["dense_probs"][sample_i, 0].cpu().numpy()

        labels = batch["labels"][sample_i, :Bh, :Bw].cpu().numpy()
        valid = batch["valid"][sample_i, :Bh, :Bw].cpu().numpy().astype(bool)
        labels = np.where(valid, labels, np.nan)

        image = imgs[sample_i, :, :Ht, :Wt].cpu().numpy()
        ids = batch.get("ids", [f"sample_{sample_i}"])
        return image, labels, panels, str(ids[sample_i])

    best_batch, best_i, best_n = None, 0, -1
    for b in test_loader:
        for i in range(b["labels"].shape[0]):
            n = int(np.nansum(b["labels"][i].cpu().numpy()))
            if n > best_n:
                best_batch, best_i, best_n = b, i, n
        if best_n > 40:
            break

    if best_batch is None:
        print("No test batches; skipping qualitative figure.")
        return

    image, labels, panels, sid = dense_panels_for_batch(best_batch, best_i)
    print(f"Sample {sid} | {best_n} positive blocks | image {image.shape}")
    if best_n == 0:
        print("WARNING: no positive blocks in the chosen sample; the confidence panels "
              "will be uniformly low.")

    Ht, Wt = image.shape[-2:]

    lab = np.nan_to_num(labels)
    if lab.sum() > 0:
        by_row = lab.sum(axis=1)
        cy = int(np.argmax(np.convolve(by_row, np.ones(3), mode="same"))) * args.block_h
        cx = int(lab.sum(axis=0).argmax()) * args.block_w
    else:
        cy, cx = Ht // 2, Wt // 2

    win = min(6 * args.block_h, Ht // 2, Wt // 2)
    y0 = int(np.clip(cy - win, 0, max(0, Ht - 2 * win)))
    x0 = int(np.clip(cx - win, 0, max(0, Wt - 2 * win)))
    zoom = (y0, min(Ht, y0 + 2 * win), x0, min(Wt, x0 + 2 * win))
    print(f"zoom window (y0,y1,x0,x1) = {zoom}")

    save_path = os.path.join(out, "figures", "qualitative_comparison.png")
    plot_dense_comparison(
        image=image,
        patch_labels=lab,
        dense_by_method=panels,
        block_size=args.block_h,
        zoom=zoom,
        layout="auto",
        title=f"Dense refinement vs. interpolation baselines - {sid}",
        save_path=save_path,
        show=False,
    )
    print(f"Wrote {save_path}")


# ---------------------------------------------------------------------------
# Per-sample structural metrics  (notebook cell 31)
# ---------------------------------------------------------------------------

def per_sample_structural(model, test_loader, device, args, out):
    import pandas as pd

    rows = []
    with torch.no_grad():
        for b in test_loader:
            imgs = b["images"].to(device)
            valid = b["valid"].to(device).bool()
            bm = b.get("block_attn_mask", b["valid"]).to(device).bool()
            o = model(imgs, block_mask=bm, return_dense=True)

            Bh, Bw = int(o["Bh"]), int(o["Bw"])
            bs = args.block_h
            Ht, Wt = Bh * bs, Bw * bs

            vpix = F.interpolate(
                valid.unsqueeze(1).float(), scale_factor=bs, mode="nearest").bool()
            ch = args.edge_channel % imgs.shape[1]
            guide = imgs[:, ch:ch + 1, :Ht, :Wt]
            ids = b.get("ids", [f"s{i}" for i in range(imgs.shape[0])])

            for i in range(imgs.shape[0]):
                d = o["dense_probs"][i:i + 1]
                g = grid_artifact_ratio(d, bs, vpix[i:i + 1])
                a = image_structure_alignment(d, guide[i:i + 1], vpix[i:i + 1])
                rows.append({
                    "id": str(ids[i]),
                    "n_pos": int(b["labels"][i].sum()),
                    "grid_artifact_fraction": g["grid_artifact_fraction"],
                    **a,
                })

    if not rows:
        print("No test samples; skipping per-sample structural metrics.")
        return

    df = pd.DataFrame(rows)
    print(df[["grid_artifact_fraction", "alignment_ratio", "alignment_corr"]].describe())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["grid_artifact_fraction", "alignment_ratio", "alignment_corr"]):
        ax.hist(df[col].dropna(), bins=40)
        ax.set_title(f"per-sample {col}")
        ax.grid(alpha=.3)
    axes[0].axvline(1.0, c="r", ls="--", label="nearest (=1.0)")
    axes[0].axvline(1.0 / args.block_h, c="g", ls="--", label=f"seam share (1/{args.block_h})")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "figures", "per_sample_structural_hist.png"),
                dpi=160, bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(out, "tables", "per_sample_structural.csv")
    df.to_csv(csv_path, index=False)

    print("\nBest-aligned samples:")
    print(df.nlargest(5, "alignment_corr")[["id", "n_pos", "alignment_corr"]])
    print("\nWorst-aligned samples:")
    print(df.nsmallest(5, "alignment_corr")[["id", "n_pos", "alignment_corr"]])
    print(f"\nWrote {csv_path} and per_sample_structural_hist.png")


# ---------------------------------------------------------------------------
# Pass-level paired bootstrap  (notebook cell 36)
# ---------------------------------------------------------------------------

def paired_bootstrap_pass_level(per_sample_a, per_sample_b, n_boot=10000, seed=0):
    """Paired bootstrap CI for a metric difference, resampling whole passes.

    Resampling passes rather than pixels is what keeps the interval honest:
    pixels inside one pass are not independent samples.
    """
    a = np.asarray(per_sample_a, dtype=float)
    b = np.asarray(per_sample_b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(f"Inputs must align by pass; got {a.shape} and {b.shape}")

    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    n = a.size

    if n == 0:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_passes": 0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = (a[idx] - b[idx]).mean(axis=1)

    return {
        "mean_diff": float((a - b).mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_passes": int(n),
    }


def bootstrap_ci(model, test_loader, device, args, out):
    per_pass = {"learned": [], "bilinear": []}

    with torch.no_grad():
        for b in test_loader:
            imgs = b["images"].to(device)
            valid = b["valid"].to(device).bool()
            bm = b.get("block_attn_mask", b["valid"]).to(device).bool()
            o = model(imgs, block_mask=bm, return_dense=True)

            Bh, Bw = int(o["Bh"]), int(o["Bw"])
            bs = args.block_h
            vpix = F.interpolate(
                valid.unsqueeze(1).float(), scale_factor=bs, mode="nearest").bool()
            patch = o["probs"].reshape(imgs.shape[0], Bh, Bw)
            bl = upsample_patch_probs(patch, bs, "bilinear")

            for i in range(imgs.shape[0]):
                per_pass["learned"].append(
                    grid_artifact_ratio(o["dense_probs"][i:i + 1], bs, vpix[i:i + 1])
                    ["grid_artifact_fraction"])
                per_pass["bilinear"].append(
                    grid_artifact_ratio(bl[i:i + 1], bs, vpix[i:i + 1])
                    ["grid_artifact_fraction"])

    ci = paired_bootstrap_pass_level(
        per_pass["learned"], per_pass["bilinear"],
        n_boot=args.bootstrap_n, seed=args.bootstrap_seed,
    )
    significant = bool(ci["ci_high"] < 0 or ci["ci_low"] > 0) if ci["n_passes"] else False

    print("Grid-artifact fraction, learned - bilinear (negative favours learned):")
    print(f"  mean difference {ci['mean_diff']:+.4f}")
    print(f"  95% CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] over {ci['n_passes']} passes")
    print("  Significant." if significant
          else "  CI spans zero; difference is not significant at this sample size.")

    dump_json(
        os.path.join(out, "metrics", "bootstrap_ci.json"),
        {
            "comparison": "grid_artifact_fraction: learned - bilinear",
            "n_boot": args.bootstrap_n,
            "seed": args.bootstrap_seed,
            "significant": significant,
            **ci,
            "per_pass_learned": per_pass["learned"],
            "per_pass_bilinear": per_pass["bilinear"],
        },
    )
    return ci


# ---------------------------------------------------------------------------
# Ablation sweep  (notebook cells 33, 35)
# ---------------------------------------------------------------------------

def summarize_seeds(ablation_results, keys=("f1", "pr_auc", "roc_auc",
                                            "grid_artifact_fraction",
                                            "alignment_corr", "equivariance_mae")):
    """Mean +/- std across seeds for each variant (notebook cell 35)."""
    by_variant = {}
    for (name, seed), res in ablation_results.items():
        by_variant.setdefault(name, []).append(res["methods"]["learned"])

    lines = ["| Variant | n seeds | " + " | ".join(keys) + " |",
             "|---|---:|" + "|".join(["---:"] * len(keys)) + "|"]

    for name, runs in by_variant.items():
        cells = []
        for k in keys:
            vals = [r[k] for r in runs if k in r and np.isfinite(r[k])]
            cells.append(f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if vals else "")
        lines.append(f"| {name} | {len(runs)} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def run_ablations(args, inferred, loaders, device, out):
    train_loader, valid_loader, test_loader = loaders

    variants = {
        "patch_only": dict(dense_use_cnn_features=False, lambda_teacher=1.0, lambda_edge=0.0,
                           dense_token_upsample="nearest"),
        "cnn_feats": dict(dense_use_cnn_features=True, lambda_teacher=1.0, lambda_edge=0.0,
                          dense_token_upsample="nearest"),
        "cnn_edge": dict(dense_use_cnn_features=True, lambda_teacher=1.0, lambda_edge=1e-3,
                         dense_token_upsample="nearest"),
        "no_teacher": dict(dense_use_cnn_features=True, lambda_teacher=0.0, lambda_edge=1e-3,
                           dense_token_upsample="nearest"),
        "bilinear_broadcast": dict(dense_use_cnn_features=True, lambda_teacher=1.0,
                                   lambda_edge=1e-3, dense_token_upsample="bilinear"),
    }

    selected = args.ablation_variants or list(variants.keys())
    variants = {k: variants[k] for k in selected}
    seeds = args.ablation_seeds
    epochs = args.ablation_epochs or args.epochs

    def run_variant(name, spec, seed):
        set_seeds(seed)

        vcfg = ModelConfig(
            **inferred,
            dense_head=True,
            dense_hidden=args.dense_hidden,
            dense_use_cnn_features=spec["dense_use_cnn_features"],
            dense_token_upsample=spec["dense_token_upsample"],
        )
        m = BlocksTokenModel(vcfg)
        load_backbone_weights(m, args.detector_ckpt)
        m = m.to(device)

        save_dir = os.path.join(out, "ablations", f"{name}_s{seed}")
        os.makedirs(save_dir, exist_ok=True)

        tr = DenseTrainer(
            model=m, device=device, alpha=args.alpha, tau=args.tau,
            lambda_teacher=spec["lambda_teacher"], lambda_edge=spec["lambda_edge"],
            edge_channel=args.edge_channel,
            grad_clip=args.grad_clip, threshold=args.threshold,
            save_dir=save_dir, freeze_backbone=True,
        )
        opt, sch = build_optim_and_sched(
            model=m, lr=args.lr, weight_decay=args.weight_decay, epochs=epochs)

        best_v, best_state = float("-inf"), None
        for ep in range(1, epochs + 1):
            tr.train_epoch(train_loader, opt)
            sch.step()
            if ep % args.eval_every and ep != epochs:
                continue
            _, mt = tr.evaluate(valid_loader)
            v = float(mt[args.best_metric])
            if np.isfinite(v) and v > best_v:
                best_v = v
                best_state = {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}

        if best_state is not None:
            m.load_state_dict(best_state)

        res = evaluate_dense_loader(
            model=m, loader=test_loader, device=device, threshold=args.threshold,
            edge_channel=args.edge_channel, include_baselines=True,
            max_structural_batches=args.max_structural_batches,
        )
        print(f"  {name} seed={seed}: val {args.best_metric}={best_v:.4f} | "
              f"test f1={res['methods']['learned']['f1']:.4f} "
              f"grid={res['methods']['learned']['grid_artifact_fraction']:.4f}")

        dump_json(
            os.path.join(save_dir, "eval_results.json"),
            {
                "methods": res["methods"],
                "val_best_metric": args.best_metric,
                "val_best_value": best_v,
                "n_valid_blocks": res["n_valid_blocks"],
                "spec": spec,
                "seed": seed,
                "epochs": epochs,
            },
        )
        return res

    ablation_results = {}
    for name, spec in variants.items():
        for seed in seeds:
            print(f"\n=== {name} | seed {seed} ===")
            ablation_results[(name, seed)] = run_variant(name, spec, seed)

    dump_json(
        os.path.join(out, "metrics", "ablation_results.json"),
        {f"{n}_s{s}": r["methods"] for (n, s), r in ablation_results.items()},
    )

    summary = summarize_seeds(ablation_results)
    print("\n" + summary)
    with open(os.path.join(out, "tables", "seed_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"Wrote ablation_results.json and seed_summary.md under {out}")

    return ablation_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out = args.out_dir

    for sub in ("checkpoints", "metrics", "tables", "figures"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    if args.run_ablations:
        os.makedirs(os.path.join(out, "ablations"), exist_ok=True)

    log_fh = open(os.path.join(out, "run.log"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    try:
        assert args.block_h == args.block_w, \
            "Tokenizer assumes square blocks (--block-h must equal --block-w)"

        device = get_device(args.device)
        print(f"device: {device}")

        dump_json(
            os.path.join(out, "args.json"),
            {**args_to_jsonable(args), "resolved_device": str(device)},
        )

        (train_loader, valid_loader, test_loader,
         n_train, n_valid, n_test) = build_dataloaders(args)
        loaders = (train_loader, valid_loader, test_loader)

        model, inferred, _cfg = build_model(args, device)
        verify_detector_preserved(model, inferred, args, valid_loader, device, out)

        freeze = not args.unfreeze_backbone

        # Stage 2 fine-tuning still distills against the *original* detector, so
        # keep an untouched copy before anything is optimized.
        teacher_model = None
        if not freeze and args.lambda_teacher > 0:
            teacher_model = copy.deepcopy(model)
            teacher_model.dense_head = None

        set_seeds(args.seed)  # notebook cell 14: seed right before training

        trainer = DenseTrainer(
            model=model, device=device,
            alpha=args.alpha, tau=args.tau,
            lambda_teacher=args.lambda_teacher, lambda_edge=args.lambda_edge,
            edge_channel=args.edge_channel,
            grad_clip=args.grad_clip, threshold=args.threshold,
            save_dir=os.path.join(out, "checkpoints"),
            freeze_backbone=freeze, teacher_model=teacher_model,
        )

        trainable = trainer.trainable_parameters()
        print(f"Trainable: {sum(p.numel() for p in trainable):,} of "
              f"{sum(p.numel() for p in model.parameters()):,}")

        optimizer, scheduler = build_optim_and_sched(
            model=model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs)

        history, best_path, best_epoch, best_value = train(
            model, trainer, optimizer, scheduler, loaders, args, out)

        plot_training_curves(history, out)

        eval_summary = {}
        if best_path is None:
            print("No best checkpoint was written; skipping the evaluation stages.")
        else:
            model = load_best(model, best_path, device)

            selected_threshold = select_threshold(model, valid_loader, device, args, out)
            # Every evaluation stage below reads args.threshold.
            args.threshold = selected_threshold

            _results, _detector_row, f1_delta, verdict = run_evaluation(
                model, test_loader, device, args, out)
            eval_summary = {
                "selected_threshold": selected_threshold,
                "learned_vs_detector_f1_delta": f1_delta,
                "fidelity_verdict": verdict,
            }

            _safe("qualitative_figure", qualitative_figure,
                  model, test_loader, device, args, out)
            _safe("per_sample_structural", per_sample_structural,
                  model, test_loader, device, args, out)
            ci = _safe("bootstrap_ci", bootstrap_ci, model, test_loader, device, args, out)
            if ci:
                eval_summary["bootstrap_ci"] = ci

        ablation_summary = None
        if args.run_ablations:
            abl = _safe("run_ablations", run_ablations, args, inferred, loaders, device, out)
            if abl is not None:
                ablation_summary = {
                    "n_variants": len({n for (n, _) in abl}),
                    "n_seeds": len({s for (_, s) in abl}),
                    "runs": len(abl),
                }

        summary = {
            "out_dir": os.path.abspath(out),
            "device": str(device),
            "data": {"train_files": n_train, "valid_files": n_valid, "test_files": n_test},
            "training": {
                "epochs": args.epochs,
                "best_metric": args.best_metric,
                "best_epoch": best_epoch,
                "best_value": (None if best_epoch is None else best_value),
                "best_checkpoint": best_path,
            },
            "evaluation": eval_summary,
            "ablations": ablation_summary,
            "artifacts": _list_artifacts(out),
        }
        dump_json(os.path.join(out, "summary.json"), summary)
        print(f"\nAll outputs written under {os.path.abspath(out)}")

    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fh.close()


if __name__ == "__main__":
    main()
