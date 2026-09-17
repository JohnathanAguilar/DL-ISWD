"""Evaluate a trained Faster R-CNN using the same protocol as the YOLO runs.

Protocol (mirrors the eval cells in
``training_notebooks/composite/yolo_training_highpass_frgb.ipynb``):

1. Cache predictions once per split at a very low confidence (0.001).
2. Sweep confidence 0.05 -> 0.95 in 0.05 steps on the **validation** split and
   pick the threshold that maximizes fixed-threshold F-beta at IoU 0.50.
3. Report the **test** split at that operating point.
4. Separately report curve-based mAP50 / mAP50-95, which need no threshold.

Example
-------
    python src/evaluation/evaluate.py --weights runs/frcnn_frgb/weights/best.pt --beta 1.0
"""

import argparse
import json
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
    from training.engine import predict_dataset
except ImportError:
    from engine import predict_dataset

try:
    from training.metrics import ap_metrics, fbeta_key, fixed_threshold_metrics
except ImportError:
    from metrics import ap_metrics, fbeta_key, fixed_threshold_metrics

try:
    from models.model import build_model, resolve_device
except ImportError:
    from model import build_model, resolve_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default=None,
                        help="Defaults to the data.yaml recorded in the checkpoint.")
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to <run_dir>/eval.")
    parser.add_argument("--tag", default="faster_rcnn",
                        help="Label used in the summary CSV's eval_type column.")

    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--beta", type=float, default=1.0,
                        help="1.0 for F1, 0.5 to favor precision.")
    parser.add_argument("--eval-iou", type=float, default=config.EVAL_IOU)
    parser.add_argument("--nms-iou", type=float, default=None,
                        help="Defaults to the value used during training.")
    parser.add_argument("--cache-conf", type=float, default=config.CACHE_CONF)
    parser.add_argument("--max-det", type=int, default=100)

    return parser.parse_args(argv)


def load_checkpoint(weights, device, nms_iou=None, cache_conf=None, max_det=100):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)

    train_args = checkpoint.get("args", {})
    class_names = checkpoint.get("class_names", ["object"])

    model = build_model(
        num_classes=len(class_names) + 1,
        backbone=train_args.get("backbone", "resnet50_fpn"),
        pretrained=False,
        min_size=train_args.get("imgsz", config.MIN_SIZE),
        max_size=train_args.get("imgsz", config.MIN_SIZE),
        nms_iou=nms_iou if nms_iou is not None else train_args.get("nms_iou", config.NMS_IOU),
        box_score_thresh=cache_conf if cache_conf is not None else config.CACHE_CONF,
        box_detections_per_img=max_det,
        trainable_backbone_layers=train_args.get("trainable_backbone_layers", 3),
        custom_anchors=not train_args.get("default_anchors", False),
    )

    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    return model, checkpoint, train_args, class_names


def cache_predictions(model, data_yaml, split, device, batch, workers):
    dataset, _ = build_dataset(data_yaml, split, train=False)

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
    )

    predictions, targets = predict_dataset(model, loader, device, desc=f"predict {split}")

    return predictions, targets, dataset


def confidence_sweep(predictions, targets, conf_values, eval_iou, beta):
    return [
        fixed_threshold_metrics(predictions, targets, conf=conf, iou_thr=eval_iou, beta=beta)
        for conf in conf_values
    ]


def main(argv=None):
    args = parse_args(argv)

    device = resolve_device(args.device)
    print(f"device: {device}")

    weights = Path(args.weights).resolve()

    model, checkpoint, train_args, class_names = load_checkpoint(
        weights,
        device,
        nms_iou=args.nms_iou,
        cache_conf=args.cache_conf,
        max_det=args.max_det,
    )

    data_yaml = args.data or train_args.get("data", str(config.DEFAULT_DATA_YAML))
    nms_iou = args.nms_iou if args.nms_iou is not None else train_args.get(
        "nms_iou", config.NMS_IOU
    )

    run_dir = weights.parent.parent
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_ids = tuple(range(1, len(class_names) + 1))
    key = fbeta_key(args.beta)

    cached = {}
    for split in ("val", "test"):
        predictions, targets, _ = cache_predictions(
            model, data_yaml, split, device, args.batch, args.workers
        )
        cached[split] = (predictions, targets)

    conf_values = np.round(
        np.arange(
            config.CONF_SWEEP_START,
            config.CONF_SWEEP_STOP + 1e-9,
            config.CONF_SWEEP_STEP,
        ),
        2,
    )

    val_sweep = confidence_sweep(
        *cached["val"], conf_values=conf_values, eval_iou=args.eval_iou, beta=args.beta
    )

    best = max(val_sweep, key=lambda row: (row[key], row["conf"] * -1))
    selected_conf = best["conf"]

    test_at_op = fixed_threshold_metrics(
        *cached["test"], conf=selected_conf, iou_thr=args.eval_iou, beta=args.beta
    )

    val_ap = ap_metrics(*cached["val"], class_ids=class_ids)
    test_ap = ap_metrics(*cached["test"], class_ids=class_ids)

    summary = {
        "model": f"faster_rcnn_{train_args.get('backbone', 'resnet50_fpn')}",
        "weights": str(weights),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "data": str(data_yaml),
        "class_names": class_names,
        "selection_rule": (
            f"Maximize fixed-threshold F{args.beta:g} on validation split"
        ),
        "confidence_range_tested": {
            "start": config.CONF_SWEEP_START,
            "stop": config.CONF_SWEEP_STOP,
            "step": config.CONF_SWEEP_STEP,
            "num_thresholds": len(conf_values),
        },
        "beta": args.beta,
        "eval_iou_for_tp_fp_fn": args.eval_iou,
        "nms_iou": nms_iou,
        "prediction_cache_conf": args.cache_conf,
        "max_detections_per_image": args.max_det,
        "selected_confidence": selected_conf,
        "validation_at_selected_conf": best,
        "test_operating_point_metrics": test_at_op,
        "standard_ap_metrics": {
            "validation": {"conf_used_for_ap_eval": args.cache_conf, **val_ap},
            "test": {"conf_used_for_ap_eval": args.cache_conf, **test_ap},
        },
        "notes": [
            "Precision, recall, and F-beta are computed manually from "
            "fixed-threshold TP/FP/FN counts.",
            "TP/FP/FN matching is class-aware, greedy by descending score, and "
            "uses eval_iou_for_tp_fp_fn.",
            "mAP50 and mAP50-95 use 101-point interpolated AP (COCO definition) "
            "over predictions cached at prediction_cache_conf.",
            "No confidence threshold is selected by mAP50 because AP/mAP is "
            "already confidence-curve based.",
        ],
        "sweep_results": val_sweep,
    }

    beta_slug = key.replace(".", "")
    json_path = out_dir / f"{args.tag}_best_conf_fixed_{beta_slug}_summary.json"

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    csv_path = out_dir / f"{args.tag}_{beta_slug}_operating_points_summary.csv"
    with open(csv_path, "w") as f:
        f.write(f"eval_type,split,confidence,tp,fp,fn,precision,recall,{key}\n")
        for split_name, row in (("validation", best), ("test", test_at_op)):
            f.write(
                f"{args.tag},{split_name},{row['conf']},{row['tp']},{row['fp']},"
                f"{row['fn']},{row['precision']},{row['recall']},{row[key]}\n"
            )

    print("\n" + "=" * 62)
    print(f"selected confidence (max val F{args.beta:g}): {selected_conf}")
    print(
        f"  val   TP {best['tp']:4d}  FP {best['fp']:4d}  FN {best['fn']:4d}  "
        f"P {best['precision']:.4f}  R {best['recall']:.4f}  {key} {best[key]:.4f}"
    )
    print(
        f"  test  TP {test_at_op['tp']:4d}  FP {test_at_op['fp']:4d}  "
        f"FN {test_at_op['fn']:4d}  P {test_at_op['precision']:.4f}  "
        f"R {test_at_op['recall']:.4f}  {key} {test_at_op[key]:.4f}"
    )
    print(
        f"  val   mAP50 {val_ap['mAP50']:.4f}  mAP50-95 {val_ap['mAP50-95']:.4f}"
    )
    print(
        f"  test  mAP50 {test_ap['mAP50']:.4f}  mAP50-95 {test_ap['mAP50-95']:.4f}"
    )
    print("=" * 62)
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")

    return summary


if __name__ == "__main__":
    main()
