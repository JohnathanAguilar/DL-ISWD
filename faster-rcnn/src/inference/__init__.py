# inference/__init__.py
from .gaussian_highpass_frcnn_detector import GaussianHighpassFasterRCNNDetector
from .rgb_composite_frcnn_detector import RGBCompositeFasterRCNNDetector

__all__ = [
    "GaussianHighpassFasterRCNNDetector",
    "RGBCompositeFasterRCNNDetector",
]
