"""Shared configuration for the Faster R-CNN internal-wave detector.

The defaults mirror the YOLO11n run in
``training_notebooks/composite/yolo_training_highpass_frgb.ipynb`` so the two
models can be compared on equal footing.
"""

import os
from pathlib import Path

# Allow ops without an MPS kernel to fall back to CPU instead of crashing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
# The composite chips are ~69 x 709 px (W x H). torchvision's detection
# transform rescales so that min(MIN_SIZE / short_side, MAX_SIZE / long_side)
# is used, so setting both to 1280 puts the long side at 1280 px -- the same
# effective resolution as the YOLO run's ``imgsz=1280``.
MIN_SIZE = 1280
MAX_SIZE = 1280

# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
# Measured on the training labels (in 1280-long-side pixel space):
#   sqrt(area) p5/median/p95 = 128 / 299 / 398 px
#   aspect ratio h/w p5/median/p95 = 1.5 / 5.8 / 10.2
# Internal-wave boxes span nearly the full swath width and vary in along-track
# height, so the default torchvision ratios (0.5, 1.0, 2.0) are far too square.
# The sizes are quoted at ANCHOR_REFERENCE_IMGSZ and rescaled if a different
# --imgsz is used.
ANCHOR_REFERENCE_IMGSZ = 1280
ANCHOR_SIZES = ((64,), (128,), (256,), (384,), (512,))
ANCHOR_ASPECT_RATIOS = (1.0, 2.0, 4.0, 8.0)

# ---------------------------------------------------------------------------
# Evaluation protocol (matches the YOLO eval cells)
# ---------------------------------------------------------------------------
EVAL_IOU = 0.50
NMS_IOU = 0.7
CACHE_CONF = 0.001
CONF_SWEEP_START = 0.05
CONF_SWEEP_STOP = 0.95
CONF_SWEEP_STEP = 0.05
