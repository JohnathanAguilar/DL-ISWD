"""RT-DETR internal-wave detector for SWOT L3 files, gaussian-highpass baseline.

Reproduces the single-channel Gaussian high-pass + magma-colormap image built
by ``SWOTInternalWaveDetector_gaussian.py`` (mode="nearest", sigma=8,
percentile [2.5, 97.5]) -- verified to be the exact algorithm behind the
``gaussian_filter_highpass_magma`` dataset ``rt-detr/runs/rtdetr_gh`` was
trained on (same mode="nearest"/sigma=8 as the precomputed
``gaussian_filter_highpass`` variable already present in
``data_w_preprocessing`` files). Structured the same way as
``rgb_composite_rtdetr_detector.py`` so the two RT-DETR baselines share
identical evaluation plumbing.
"""

import sys
from pathlib import Path

_INFERENCE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _INFERENCE_DIR.parent
_MODEL_DIR = _SRC_DIR.parent

for _p in (_SRC_DIR, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from transformers import RTDetrImageProcessor

try:
    from configs import config
except ImportError:
    import config

try:
    from models.model import build_model, resolve_device
except ImportError:
    from model import build_model, resolve_device


class GaussianHighpassRTDetrDetector:
    """Detect internal waves in SWOT L3 files with an RT-DETR checkpoint
    trained on the ``gaussian_filter_highpass_magma`` dataset."""

    def __init__(
        self,
        model_path,
        device="auto",
        base_var="ssha_unfiltered",
        quality_var="quality_flag",
        gaussian_sigma=8,
        p_low=2.5,
        p_high=97.5,
        max_valid_ssha=10.0,
    ):
        self.base_var = base_var
        self.quality_var = quality_var
        self.gaussian_sigma = gaussian_sigma
        self.p_low = p_low
        self.p_high = p_high
        self.max_valid_ssha = max_valid_ssha

        self.device = resolve_device(device)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        train_args = checkpoint.get("args", {})
        self.class_names = checkpoint.get("class_names", ["object"])

        self.model = build_model(
            train_args.get("model_name", config.DEFAULT_MODEL_NAME),
            num_labels=len(self.class_names),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        self.model.eval()

        self.processor = RTDetrImageProcessor(
            size={
                "shortest_edge": train_args.get("shortest_edge", config.SHORTEST_EDGE),
                "longest_edge": train_args.get("longest_edge", config.LONGEST_EDGE),
            },
            do_pad=True,
        )

        self.model_path = model_path

    # ============================================================
    # Dataset loading
    # ============================================================

    def load_l3_dataset(self, l3_file):
        """Load the complete L3 NetCDF file without latitude clipping."""
        with xr.open_dataset(l3_file) as ds:
            ds_full = ds.load()

        ds_full.attrs["region"] = "full_swath"
        ds_full.attrs["clip_type"] = "none"
        ds_full.attrs["source_l3_file"] = str(l3_file)

        return ds_full

    # ============================================================
    # Array helpers
    # ============================================================

    def get_2d_array(self, da):
        arr = da.values
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array after squeeze, got shape {arr.shape}")

        return arr

    def apply_quality_flag_mask(self, ssha, quality_flag):
        """Keep only quality_flag values 0, 5, 30, and 50."""
        ssha = ssha.astype(np.float32).copy()

        good_quality = (
            (quality_flag == 0)
            | (quality_flag == 5)
            | (quality_flag == 30)
            | (quality_flag == 50)
        )

        ssha[~good_quality] = np.nan

        return ssha

    def prepare_ssha(self, ds):
        """Extract SSHA, apply quality masking and the max-valid clip."""
        ds_plot = ds.isel(file=0) if "file" in ds.dims else ds

        if self.base_var not in ds_plot:
            raise KeyError(f"Missing base variable: {self.base_var}")

        if self.quality_var not in ds_plot:
            raise KeyError(f"Missing quality variable: {self.quality_var}")

        ssha = self.get_2d_array(ds_plot[self.base_var]).astype(np.float32)
        quality_flag = self.get_2d_array(ds_plot[self.quality_var])

        if ssha.shape != quality_flag.shape:
            raise ValueError(
                f"Shape mismatch: {self.base_var} shape={ssha.shape}, "
                f"{self.quality_var} shape={quality_flag.shape}"
            )

        ssha = self.apply_quality_flag_mask(ssha, quality_flag)
        ssha = np.where(ssha <= self.max_valid_ssha, ssha, np.nan).astype(np.float32)

        return ssha

    # ============================================================
    # Gaussian high-pass (mode="nearest", matches SWOTInternalWaveDetector_gaussian.py)
    # ============================================================

    def nan_gaussian_filter(self, arr, sigma):
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        arr_filled = np.where(valid, arr, 0.0).astype(np.float32)
        weights = valid.astype(np.float32)

        smooth_sum = gaussian_filter(arr_filled, sigma=sigma, mode="nearest")
        weight_sum = gaussian_filter(weights, sigma=sigma, mode="nearest")

        smooth = smooth_sum / np.maximum(weight_sum, 1e-6)
        smooth[weight_sum <= 1e-6] = np.nan

        return smooth.astype(np.float32)

    def gaussian_highpass(self, arr):
        """High-pass = original - Gaussian low-pass."""
        arr = arr.astype(np.float32)

        valid = np.isfinite(arr)
        smooth = self.nan_gaussian_filter(arr, sigma=self.gaussian_sigma)

        highpass = arr - smooth
        highpass[~valid] = np.nan

        return highpass.astype(np.float32)

    # ============================================================
    # Image creation
    # ============================================================

    def array_to_rgb(self, arr):
        """Convert the Gaussian high-pass array into an unoriented uint8 RGB
        array via the magma colormap. NaN/masked pixels are white."""
        arr = arr.astype(np.float32).copy()
        arr = np.where(arr <= self.max_valid_ssha, arr, np.nan)

        if not np.isfinite(arr).any():
            raise ValueError("Cannot convert array to image because all values are NaN.")

        vmin, vmax = np.nanpercentile(arr, [self.p_low, self.p_high])

        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            raise ValueError(f"Bad image scaling limits: vmin={vmin}, vmax={vmax}")

        cmap_obj = plt.get_cmap("magma").copy()
        cmap_obj.set_bad(color="white")

        arr_masked = np.ma.masked_invalid(arr)
        normalized = (arr_masked - vmin) / (vmax - vmin)
        normalized = np.ma.clip(normalized, 0, 1)

        rgba = cmap_obj(normalized)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        return rgb

    # ============================================================
    # Orientation (matches the training image pipeline)
    # ============================================================

    def get_transform_info(self, arr2d):
        original_shape = arr2d.shape
        was_transposed = arr2d.shape[1] > arr2d.shape[0]
        transformed_shape = arr2d.T.shape if was_transposed else arr2d.shape

        return {
            "original_shape": original_shape,
            "transformed_shape": transformed_shape,
            "was_transposed": was_transposed,
        }

    @staticmethod
    def transform_info_from_shape(original_shape):
        was_transposed = original_shape[1] > original_shape[0]
        transformed_shape = (
            (original_shape[1], original_shape[0]) if was_transposed else tuple(original_shape)
        )

        return {
            "original_shape": tuple(original_shape),
            "transformed_shape": transformed_shape,
            "was_transposed": was_transposed,
        }

    @staticmethod
    def yolo_label_to_original_mask(label_path, original_shape):
        """Rasterize a normalized YOLO-format label file (``cls cx cy w h``
        per line, in the oriented model-input image space) into a binary
        mask on the original SWOT grid. Identical geometry to
        ``rgb_composite_rtdetr_detector.py``'s method of the same name --
        the gaussian-highpass and rgb-composite datasets share the exact
        same test-split labels and image dimensions.
        """
        transform_info = GaussianHighpassRTDetrDetector.transform_info_from_shape(
            original_shape
        )
        height, width = transform_info["transformed_shape"]

        mask_transformed = np.zeros(transform_info["transformed_shape"], dtype=np.uint8)

        label_path = Path(label_path)
        if label_path.is_file():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue

                cx, cy, w, h = (float(v) for v in parts[1:5])

                x1 = (cx - w / 2.0) * width
                x2 = (cx + w / 2.0) * width
                y1 = (cy - h / 2.0) * height
                y2 = (cy + h / 2.0) * height

                x1 = max(0, min(int(np.floor(x1)), width - 1))
                x2 = max(0, min(int(np.ceil(x2)), width - 1))
                y1 = max(0, min(int(np.floor(y1)), height - 1))
                y2 = max(0, min(int(np.ceil(y2)), height - 1))

                if x2 >= x1 and y2 >= y1:
                    mask_transformed[y1 : y2 + 1, x1 : x2 + 1] = 1

        mask_unflipped = np.fliplr(mask_transformed)
        mask_original = (
            mask_unflipped.T if transform_info["was_transposed"] else mask_unflipped
        )

        if mask_original.shape != transform_info["original_shape"]:
            raise ValueError(
                f"Mask shape mismatch. Expected {transform_info['original_shape']}, "
                f"got {mask_original.shape}"
            )

        return mask_original.astype(np.uint8)

    def orient_rgb_for_model(self, rgb, was_transposed):
        if was_transposed:
            rgb = np.transpose(rgb, (1, 0, 2))

        return np.fliplr(rgb)

    # ============================================================
    # RT-DETR detection
    # ============================================================

    def run_detection(self, pil_image, conf):
        rgb_image = pil_image.convert("RGB")

        # do_pad=True is a no-op for a batch of one (it pads to the batch's
        # own max height/width, i.e. itself), so the resized width here can
        # land on any value depending on the chip's aspect ratio -- e.g. 110
        # or 125 px for our ~60-70px-wide chips. The hybrid encoder's FPN
        # upsamples its coarsest (stride-32) feature map by exactly 2x and
        # concats it with the stride-16 feature map; that only lines up when
        # every spatial dim is exactly divisible by 32 through all of the
        # backbone's halving stages; otherwise the two branches round to
        # different sizes and torch.concat throws a tensor-size mismatch.
        # Explicitly padding both dims up to the next multiple of 32 (the
        # standard size-divisibility convention for ResNet-FPN detectors)
        # fixes this for every chip aspect ratio, not just the ones tested.
        inputs = self.processor(
            images=[np.array(rgb_image)], do_pad=False, return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"][0]

        stride = 32
        resized_height, resized_width = pixel_values.shape[-2:]
        padded_height = -(-resized_height // stride) * stride
        padded_width = -(-resized_width // stride) * stride

        pixel_values, pixel_mask, _ = self.processor.pad(
            pixel_values, padded_size=(padded_height, padded_width)
        )
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        pixel_mask = pixel_mask.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        target_sizes = [(rgb_image.height, rgb_image.width)]
        result = self.processor.post_process_object_detection(
            outputs, threshold=conf, target_sizes=target_sizes, use_focal_loss=True
        )[0]

        boxes_xyxy = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()

        detections = [
            {
                "class_id": int(label),
                "score": float(score),
                "xyxy": [float(v) for v in box],
            }
            for box, score, label in zip(boxes_xyxy, scores, labels)
        ]

        return detections

    def detections_to_text(self, detections):
        if not detections:
            return "No detections."

        lines = []
        for i, det in enumerate(detections):
            label = self.class_names[det["class_id"]] if det["class_id"] < len(
                self.class_names
            ) else str(det["class_id"])
            x1, y1, x2, y2 = det["xyxy"]
            lines.append(
                f"{i}: {label} conf={det['score']:.3f} "
                f"bbox_xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
            )

        return "\n".join(lines)

    def draw_detections(self, pil_image, detections):
        boxed_image = pil_image.copy()
        draw = ImageDraw.Draw(boxed_image)

        for det in detections:
            label = self.class_names[det["class_id"]] if det["class_id"] < len(
                self.class_names
            ) else str(det["class_id"])
            x1, y1, x2, y2 = det["xyxy"]

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, max(0, y1 - 14)), f"{label} {det['score']:.2f}", fill="red")

        return boxed_image

    # ============================================================
    # Detections back to the original SWOT grid
    # ============================================================

    def detections_to_original_mask(self, detections, transform_info):
        transformed_shape = transform_info["transformed_shape"]
        original_shape = transform_info["original_shape"]
        was_transposed = transform_info["was_transposed"]

        mask_transformed = np.zeros(transformed_shape, dtype=np.uint8)
        height, width = transformed_shape

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]

            x1 = max(0, min(int(np.floor(x1)), width - 1))
            x2 = max(0, min(int(np.ceil(x2)), width - 1))
            y1 = max(0, min(int(np.floor(y1)), height - 1))
            y2 = max(0, min(int(np.ceil(y2)), height - 1))

            if x2 >= x1 and y2 >= y1:
                mask_transformed[y1 : y2 + 1, x1 : x2 + 1] = 1

        mask_unflipped = np.fliplr(mask_transformed)
        mask_original = mask_unflipped.T if was_transposed else mask_unflipped

        if mask_original.shape != original_shape:
            raise ValueError(
                f"Mask shape mismatch. Expected {original_shape}, got {mask_original.shape}"
            )

        return mask_original.astype(np.uint8)

    def detections_to_original_confidence_mask(self, detections, transform_info):
        transformed_shape = transform_info["transformed_shape"]
        original_shape = transform_info["original_shape"]
        was_transposed = transform_info["was_transposed"]

        confidence_transformed = np.zeros(transformed_shape, dtype=np.float32)
        height, width = transformed_shape

        for det in detections:
            confidence = float(np.clip(det["score"], 0.0, 1.0))
            x1, y1, x2, y2 = det["xyxy"]

            x1 = max(0, min(int(np.floor(x1)), width - 1))
            x2 = max(0, min(int(np.ceil(x2)), width - 1))
            y1 = max(0, min(int(np.floor(y1)), height - 1))
            y2 = max(0, min(int(np.ceil(y2)), height - 1))

            if x2 >= x1 and y2 >= y1:
                region = confidence_transformed[y1 : y2 + 1, x1 : x2 + 1]
                np.maximum(region, confidence, out=region)

        confidence_unflipped = np.fliplr(confidence_transformed)
        confidence_original = (
            confidence_unflipped.T if was_transposed else confidence_unflipped
        )

        if confidence_original.shape != original_shape:
            raise ValueError(
                f"Confidence mask shape mismatch. Expected {original_shape}, "
                f"got {confidence_original.shape}"
            )

        return confidence_original.astype(np.float32)

    # ============================================================
    # Save results back to xarray
    # ============================================================

    def add_array_to_dataset(self, ds, arr, var_name, dims, coords, attrs):
        ds_out = ds.copy()

        ds_out[var_name] = xr.DataArray(arr, dims=dims, coords=coords, name=var_name, attrs=attrs)

        return ds_out

    def add_outputs_to_dataset(
        self,
        clipped_ds,
        mask,
        confidence_mask,
        filtered_arr,
        mask_var_name,
        conf_mask_var_name,
        save_filtered,
    ):
        ds_out = clipped_ds.copy()
        ds_plot = ds_out.isel(file=0) if "file" in ds_out.dims else ds_out
        base_da = ds_plot[self.base_var].squeeze()
        dims = base_da.dims
        coords = {dim: base_da.coords[dim] for dim in dims if dim in base_da.coords}

        ds_out = self.add_array_to_dataset(
            ds_out,
            mask.astype(np.uint8),
            mask_var_name,
            dims,
            coords,
            {
                "description": "Pixels inside RT-DETR internal-wave bounding boxes.",
                "mask_values": "1 = inside a detected bounding box, 0 = outside",
            },
        )

        ds_out = self.add_array_to_dataset(
            ds_out,
            confidence_mask.astype(np.float32),
            conf_mask_var_name,
            dims,
            coords,
            {
                "description": (
                    "RT-DETR detection confidence assigned to pixels inside "
                    "internal-wave bounding boxes. Overlapping boxes use the highest "
                    "confidence."
                ),
                "valid_range": np.array([0.0, 1.0], dtype=np.float32),
                "mask_values": (
                    "0 = outside all detected boxes; values in (0, 1] = maximum "
                    "detection confidence covering the pixel"
                ),
            },
        )

        if save_filtered:
            ds_out = self.add_array_to_dataset(
                ds_out,
                filtered_arr.astype(np.float32),
                "rtdetr_gh_gaussian_highpass",
                dims,
                coords,
                {
                    "description": (
                        "Gaussian high-pass SSHA (mode='nearest', sigma=8) fed to the "
                        "magma colormap image used for RT-DETR detection. Aligned to "
                        "the original grid, not the oriented model-input image."
                    ),
                    "source_variable": self.base_var,
                },
            )

        return ds_out

    # ============================================================
    # Full detection wrapper
    # ============================================================

    def detect_on_l3_file(
        self,
        l3_file,
        conf=0.10,
        mask_var_name="isw_detection_mask",
        conf_mask_var_name="isw_detection_confidence",
        save_filtered=True,
    ):
        """Load an L3 file, run the RT-DETR gaussian-highpass detector on it,
        and return the dataset with:
        - a binary bounding-box mask
        - a confidence mask
        - the Gaussian high-pass array used for detection, if save_filtered=True
        """
        ds = self.load_l3_dataset(l3_file)

        ssha = self.prepare_ssha(ds)
        transform_info = self.get_transform_info(ssha)

        filtered_arr = self.gaussian_highpass(ssha)
        rgb = self.array_to_rgb(filtered_arr)
        rgb_oriented = self.orient_rgb_for_model(rgb, transform_info["was_transposed"])

        pil_image = Image.fromarray(rgb_oriented, mode="RGB")

        detections = self.run_detection(pil_image, conf=conf)
        detection_text = self.detections_to_text(detections)
        boxed_image = self.draw_detections(pil_image, detections)

        mask = self.detections_to_original_mask(detections, transform_info)
        confidence_mask = self.detections_to_original_confidence_mask(
            detections, transform_info
        )

        ds_with_outputs = self.add_outputs_to_dataset(
            ds,
            mask=mask,
            confidence_mask=confidence_mask,
            filtered_arr=filtered_arr,
            mask_var_name=mask_var_name,
            conf_mask_var_name=conf_mask_var_name,
            save_filtered=save_filtered,
        )

        ds_with_outputs.attrs["rtdetr_model_path"] = str(self.model_path)
        ds_with_outputs.attrs["rtdetr_confidence_threshold"] = float(conf)
        ds_with_outputs.attrs["rtdetr_mask_variable"] = str(mask_var_name)
        ds_with_outputs.attrs["rtdetr_confidence_mask_variable"] = str(conf_mask_var_name)

        return ds_with_outputs, detection_text, boxed_image, detections
