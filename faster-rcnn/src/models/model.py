"""Faster R-CNN model builders tuned for SWOT internal-wave chips."""

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

try:
    from configs.config import (
        ANCHOR_ASPECT_RATIOS,
        ANCHOR_REFERENCE_IMGSZ,
        ANCHOR_SIZES,
        MAX_SIZE,
        MIN_SIZE,
        NMS_IOU,
    )
except ImportError:
    from config import (
        ANCHOR_ASPECT_RATIOS,
        ANCHOR_REFERENCE_IMGSZ,
        ANCHOR_SIZES,
        MAX_SIZE,
        MIN_SIZE,
        NMS_IOU,
    )

BACKBONES = {
    "resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
}


def _anchor_sizes(num_levels, imgsz):
    """Adapt ``ANCHOR_SIZES`` to the backbone's FPN depth and the resize target.

    ``resnet50_fpn`` exposes five levels and uses the ladder verbatim; the
    mobilenet backbone exposes fewer, so the ladder is resampled geometrically
    over the same range. Sizes are quoted at ``ANCHOR_REFERENCE_IMGSZ`` and
    scaled linearly with ``imgsz`` so the anchors keep covering the same
    fraction of the chip.
    """
    scale = imgsz / ANCHOR_REFERENCE_IMGSZ

    if num_levels == len(ANCHOR_SIZES):
        ladder = [s[0] for s in ANCHOR_SIZES]
    else:
        low, high = ANCHOR_SIZES[0][0], ANCHOR_SIZES[-1][0]
        ratio = (high / low) ** (1.0 / max(num_levels - 1, 1))
        ladder = [low * ratio**i for i in range(num_levels)]

    return tuple((max(8, int(round(size * scale))),) for size in ladder)


def build_model(
    num_classes,
    backbone="resnet50_fpn_v2",
    pretrained=True,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    nms_iou=NMS_IOU,
    box_score_thresh=0.001,
    box_detections_per_img=100,
    trainable_backbone_layers=3,
    custom_anchors=True,
):
    """Build a Faster R-CNN detector.

    Parameters
    ----------
    num_classes : int
        Number of foreground classes **plus one** for background.
    backbone : str
        Key of ``BACKBONES``.
    custom_anchors : bool
        Replace the default square-ish anchors with the tall anchors that match
        internal-wave box statistics (see ``config.py``).
    """
    if backbone not in BACKBONES:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from {sorted(BACKBONES)}."
        )

    builder = BACKBONES[backbone]

    kwargs = dict(
        weights="DEFAULT" if pretrained else None,
        weights_backbone="DEFAULT" if pretrained else None,
        min_size=min_size,
        max_size=max_size,
        box_nms_thresh=nms_iou,
        box_score_thresh=box_score_thresh,
        box_detections_per_img=box_detections_per_img,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    model = builder(**kwargs)

    if custom_anchors:
        # The stock builders install their own anchor generator, so replace it
        # (and the RPN head, whose output width depends on the anchor count)
        # after construction.
        sizes = _anchor_sizes(len(model.rpn.anchor_generator.sizes), min_size)

        anchor_generator = AnchorGenerator(
            sizes=sizes,
            aspect_ratios=(ANCHOR_ASPECT_RATIOS,) * len(sizes),
        )

        model.rpn.anchor_generator = anchor_generator
        model.rpn.head = RPNHead(
            model.backbone.out_channels,
            anchor_generator.num_anchors_per_location()[0],
            conv_depth=2 if backbone.endswith("_v2") else 1,
        )

    # Swap the COCO 91-class head for this dataset's head.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def resolve_device(requested="auto"):
    """Pick a torch device, preferring MPS on Apple silicon."""
    if requested not in ("auto", None):
        requested = str(requested)
        if requested.isdigit():
            # Bare GPU index, e.g. "0" or "1" -> cuda:0 / cuda:1.
            requested = f"cuda:{requested}"
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
