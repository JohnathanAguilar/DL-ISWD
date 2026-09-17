"""Train YOLO11 on the YOLO-format SWOT chips via Ultralytics' built-in trainer.

Example
-------
    python train.py --model-size n --epochs 100 --name frgb
    python train.py --model-size s --epochs 100 --name frgb_small

Outputs land in ``runs/<name>/`` (Ultralytics' own run layout):
    weights/best.pt   checkpoint with the highest validation mAP50
    weights/last.pt   checkpoint from the final epoch
    results.csv       per-epoch losses and validation metrics
    results.png       loss / metric curves
    args.yaml         the exact configuration used (written by Ultralytics)
"""

import argparse
import os
from pathlib import Path

# Allow duplicate OpenMP runtimes (torch + numpy) to coexist instead of crashing.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from ultralytics import YOLO

_MODEL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODEL_DIR.parent

DEFAULT_DATA_YAML = (
    _REPO_ROOT
    / "data"
    / "Labeled_SWOT_L3_BOB_dataset"
    / "composite_grouped_yolo_splits"
    / "scientific_rgb_composite"
    / "data.yaml"
)
DEFAULT_RUNS_DIR = _MODEL_DIR / "runs"

WEIGHTS_BY_SIZE = {"n": "yolo11n.pt", "s": "yolo11s.pt"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--name", default="frgb")
    parser.add_argument("--project", default=str(DEFAULT_RUNS_DIR))

    parser.add_argument(
        "--model-size", default="n", choices=["n", "s"],
        help="YOLO11 variant to fine-tune: n = nano, s = small.",
    )
    parser.add_argument(
        "--weights", default=None,
        help="Explicit checkpoint to start from, overriding --model-size "
             "(e.g. to resume or fine-tune from a specific .pt).",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device", default="mps",
        help="Compute backend to train on: 'mps', 'cpu', 'auto' (picks mps > cuda > cpu), "
             "or a CUDA device spec like '0' or '0,1'.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM for faster epochs.")

    parser.add_argument("--lr0", type=float, default=0.002,
                        help="Initial learning rate (Ultralytics default is 0.01).")

    # Augmentation -- all disabled by default. The training set was already
    # augmented offline, so no online augmentation is applied here, matching
    # the faster-rcnn / rt-detr training recipes used on this dataset.
    parser.add_argument("--hsv-h", type=float, default=0.0)
    parser.add_argument("--hsv-s", type=float, default=0.0)
    parser.add_argument("--hsv-v", type=float, default=0.0)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--multi-scale", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.0)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--translate", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--degrees", type=float, default=0.0)

    return parser.parse_args(argv)


def resolve_device(device):
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
        return "cpu"
    return device


def main(argv=None):
    args = parse_args(argv)

    print("MPS available:", torch.backends.mps.is_available())
    device = resolve_device(args.device)

    weights = args.weights or WEIGHTS_BY_SIZE[args.model_size]
    model = YOLO(weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        cache=args.cache,
        lr0=args.lr0,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        multi_scale=args.multi_scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        translate=args.translate,
        scale=args.scale,
        degrees=args.degrees,
    )


if __name__ == "__main__":
    main()
