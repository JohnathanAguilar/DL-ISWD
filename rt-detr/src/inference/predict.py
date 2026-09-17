"""Draw RT-DETR predictions (and ground truth) on dataset chips.

Example
-------
    python src/inference/predict.py --weights runs/rtdetr_frgb/weights/best.pt \
        --split test --conf 0.15 --num 12
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch

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
    from dataio.dataset import build_dataset
except ImportError:
    from dataset import build_dataset

try:
    from evaluation.evaluate import load_checkpoint
except ImportError:
    from evaluate import load_checkpoint

try:
    from models.model import resolve_device
except ImportError:
    from model import resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--only-with-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)

    device = resolve_device(args.device)
    weights = Path(args.weights).resolve()

    model, processor, _, train_args, _ = load_checkpoint(weights, device, cache_conf=args.conf)

    data_yaml = args.data or train_args.get("data", str(config.DEFAULT_DATA_YAML))
    dataset, _ = build_dataset(data_yaml, args.split, train=False)

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(dataset))

    chosen = []
    for idx in indices:
        idx = int(idx)
        image, annotation = dataset[idx]

        if args.only_with_labels and not annotation["annotations"]:
            continue

        chosen.append((idx, image, annotation))

        if len(chosen) >= args.num:
            break

    if not chosen:
        raise RuntimeError("No images matched the selection criteria.")

    batch = processor(
        images=[c[1] for c in chosen],
        annotations=[c[2] for c in chosen],
        return_tensors="pt",
    )
    pixel_values = batch["pixel_values"].to(device)
    pixel_mask = batch.get("pixel_mask")
    pixel_mask = pixel_mask.to(device) if pixel_mask is not None else None

    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    target_sizes = [tuple(int(v) for v in label["orig_size"].tolist()) for label in batch["labels"]]
    results = processor.post_process_object_detection(
        outputs, threshold=args.conf, target_sizes=target_sizes, use_focal_loss=True
    )

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 7.5))
    axes = np.atleast_1d(axes)

    for ax, (idx, image, _), result in zip(axes, chosen, results):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(dataset.image_name(idx)[:18] + "…", fontsize=5)

        for box in dataset.ground_truth_boxes(idx):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(
                patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="lime", facecolor="none")
            )

        for box, score in zip(result["boxes"].cpu(), result["scores"].cpu()):
            x1, y1, x2, y2 = box.tolist()
            ax.add_patch(
                patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="red", facecolor="none")
            )
            ax.text(x1, max(0, y1 - 4), f"{float(score):.2f}", color="red", fontsize=5)

    fig.suptitle(
        f"RT-DETR {args.split} predictions (red) vs ground truth (green), conf ≥ {args.conf}", fontsize=9
    )
    fig.tight_layout()

    out_path = Path(args.out) if args.out else weights.parent.parent / f"predictions_{args.split}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {out_path}")

    return out_path


if __name__ == "__main__":
    main()
