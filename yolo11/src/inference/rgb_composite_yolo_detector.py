"""YOLO11 internal-wave detector for the scientific_rgb_composite family
(frgb + its 3 channel ablations: nograd, nohp, nossha).

Channel recipe (confirmed by the user, not previously documented anywhere in
this repo): R = ssha_quality_masked, G = gaussian_filter_highpass,
B = gradient_magnitude -- all three already precomputed in the source
NetCDF. Each ablation zero-fills exactly the channel it ablates; the other
two channels are unchanged.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

from . import common

CHANNEL_VARS = {
    "R": "ssha_quality_masked",
    "G": "gaussian_filter_highpass",
    "B": "gradient_magnitude",
}


class RGBCompositeYOLODetector:
    def __init__(self, weights_path, device="auto", zero_channel=None, p_low=2.0, p_high=98.0):
        if zero_channel not in (None, "R", "G", "B"):
            raise ValueError(f"zero_channel must be one of None/R/G/B, got {zero_channel!r}")

        self.weights_path = weights_path
        self.device = device
        self.zero_channel = zero_channel
        self.p_low = p_low
        self.p_high = p_high

        self.model = YOLO(str(weights_path))

    def build_composite(self, ds):
        base_ssha = common.read_2d_var(ds, CHANNEL_VARS["R"])
        no_data_mask = ~np.isfinite(base_ssha)

        channels = []
        for key in ("R", "G", "B"):
            if key == self.zero_channel:
                channels.append(np.zeros(base_ssha.shape, dtype=np.uint8))
            else:
                arr = base_ssha if key == "R" else common.read_2d_var(ds, CHANNEL_VARS[key])
                channels.append(common.robust_normalize(arr, self.p_low, self.p_high))

        rgb = np.stack(channels, axis=-1).astype(np.uint8)
        rgb[no_data_mask] = np.array([255, 255, 255], dtype=np.uint8)

        return rgb

    def detect_on_l3_file(
        self,
        l3_file,
        conf,
        imgsz,
        mask_var_name="internal_wave_bbox_mask",
        conf_mask_var_name="internal_wave_bbox_confidence",
    ):
        ds = common.load_l3_dataset(l3_file)

        rgb = self.build_composite(ds)
        transform_info = common.get_transform_info(rgb.shape[:2])
        rgb_oriented = common.orient_rgb_for_model(rgb, transform_info["was_transposed"])

        pil_image = Image.fromarray(rgb_oriented, mode="RGB")
        detections = common.run_yolo_detection(self.model, pil_image, conf, imgsz, self.device)

        mask = common.detections_to_original_mask(detections, transform_info)
        confidence_mask = common.detections_to_original_confidence_mask(detections, transform_info)

        return ds, detections, transform_info, mask, confidence_mask
