"""YOLO11 internal-wave detector for the single-channel "highpass" magma
variants (gauss, fft, gradmag, ssha, rollmean, stepconv).

Each variant renders one precomputed NetCDF variable through the magma
colormap (percentile clip [2.5, 97.5], NaN -> white) -- the only rendering
recipe confirmed anywhere in this repo (gaussian_highpass_frcnn_detector.py),
reused uniformly across all 6 variants per user direction.
"""

from PIL import Image
from ultralytics import YOLO

from . import common


class HighpassYOLODetector:
    def __init__(self, weights_path, device="auto", source_var=None, p_low=2.5, p_high=97.5):
        if not source_var:
            raise ValueError("source_var is required")

        self.weights_path = weights_path
        self.device = device
        self.source_var = source_var
        self.p_low = p_low
        self.p_high = p_high

        self.model = YOLO(str(weights_path))

    def detect_on_l3_file(
        self,
        l3_file,
        conf,
        imgsz,
        mask_var_name="internal_wave_bbox_mask",
        conf_mask_var_name="internal_wave_bbox_confidence",
    ):
        ds = common.load_l3_dataset(l3_file)

        arr = common.read_2d_var(ds, self.source_var)
        transform_info = common.get_transform_info(arr.shape)

        rgb = common.array_to_magma_rgb(arr, self.p_low, self.p_high)
        rgb_oriented = common.orient_rgb_for_model(rgb, transform_info["was_transposed"])

        pil_image = Image.fromarray(rgb_oriented, mode="RGB")
        detections = common.run_yolo_detection(self.model, pil_image, conf, imgsz, self.device)

        mask = common.detections_to_original_mask(detections, transform_info)
        confidence_mask = common.detections_to_original_confidence_mask(detections, transform_info)

        return ds, detections, transform_info, mask, confidence_mask
