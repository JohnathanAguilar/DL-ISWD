"""Shared configuration for the RT-DETR internal-wave detector.

Mirrors ``faster-rcnn/config.py`` so the two baselines are held to the same
protocol; see ``README.md`` for what is/isn't matched to the YOLO run.
"""

import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = MODEL_DIR.parent

DEFAULT_DATA_YAML = (
    REPO_ROOT
    / "data"
    / "Labeled_SWOT_L3_BOB_dataset"
    / "composite_grouped_yolo_splits"
    / "scientific_rgb_composite"
    / "data.yaml"
)

DEFAULT_RUNS_DIR = MODEL_DIR / "runs"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------
# RTDetrImageProcessor's {"shortest_edge": S, "longest_edge": L} resize uses
# the same rule as torchvision's GeneralizedRCNNTransform(min_size, max_size):
# scale so the long side hits L, unless that would need the short side to
# exceed S. Our chips are ~69 x 709 px (width << height), so the long-side
# cap always binds; setting both to 1280 reproduces the exact resize used for
# Faster R-CNN (verified: 709x69 -> 1280x125 under both implementations).
SHORTEST_EDGE = 1280
LONGEST_EDGE = 1280

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# r50vd matches the ResNet-50 backbone used for the Faster R-CNN baseline, so
# the two baselines differ in detection head (two-stage R-CNN vs. DETR-style
# transformer decoder) rather than backbone capacity.
DEFAULT_MODEL_NAME = "PekingU/rtdetr_r50vd"

# ---------------------------------------------------------------------------
# Evaluation protocol (matches the YOLO and Faster R-CNN eval cells)
# ---------------------------------------------------------------------------
# RT-DETR has no NMS step (query-based one-to-one matching suppresses
# duplicates during training), so there is no NMS_IOU knob here.
EVAL_IOU = 0.50
CACHE_CONF = 0.001
CONF_SWEEP_START = 0.05
CONF_SWEEP_STOP = 0.95
CONF_SWEEP_STEP = 0.05

# Maximum predictions kept per image before confidence filtering. RT-DETR
# decodes a fixed number of queries (300 by default); this just bounds how
# many are cached during evaluation.
MAX_DETECTIONS = 300
