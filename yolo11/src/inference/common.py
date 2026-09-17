"""Shared, family-agnostic helpers for running trained YOLO11 models on SWOT L3
NetCDF files.

Geometry (orientation, detection -> original-grid mask projection) is ported
verbatim from ``faster-rcnn/src/inference/rgb_composite_frcnn_detector.py`` --
that logic is preprocessing-family-agnostic and must stay pixel-identical
across every YOLO11 variant, so it lives here once instead of being copied
into each detector file the way the two Faster R-CNN detectors do.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


# ============================================================
# NetCDF loading
# ============================================================

def load_l3_dataset(l3_file):
    """Load a complete SWOT L3 NetCDF file into memory."""
    with xr.open_dataset(l3_file) as ds:
        ds_full = ds.load()

    ds_full.attrs["source_l3_file"] = str(l3_file)
    return ds_full


def read_2d_var(ds, name):
    if name not in ds:
        raise KeyError(f"Missing variable: {name}")

    arr = np.squeeze(np.asarray(ds[name].values)).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected {name!r} to be 2D after squeeze, got shape {arr.shape}")

    return arr


# ============================================================
# Normalization / colormap rendering
# ============================================================

def robust_normalize(arr, p_low, p_high):
    """Normalize a 2D array to uint8 [0, 255] using robust percentiles.

    Ported verbatim from rgb_composite_frcnn_detector.py:227-255.
    """
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)

    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)

    work = arr.copy()
    median = np.nanmedian(work)
    if not np.isfinite(median):
        median = 0.0
    work[~finite] = median

    lo = np.percentile(work, p_low)
    hi = np.percentile(work, p_high)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(work)
        hi = np.nanmax(work)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    work = (work - lo) / (hi - lo)
    work = np.clip(work, 0.0, 1.0)

    return (work * 255.0).astype(np.uint8)


def array_to_magma_rgb(arr, p_low=2.5, p_high=97.5):
    """Convert a 2D array into an unoriented uint8 RGB array via the magma
    colormap. NaN/masked pixels are white.

    Ported verbatim from gaussian_highpass_frcnn_detector.py:186-210 (the
    only magma-rendering recipe confirmed anywhere in this repo; reused for
    every single-channel "highpass" variant per user direction).
    """
    arr = np.asarray(arr, dtype=np.float32)

    if not np.isfinite(arr).any():
        raise ValueError("Cannot convert array to image because all values are NaN.")

    vmin, vmax = np.nanpercentile(arr, [p_low, p_high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        raise ValueError(f"Bad image scaling limits: vmin={vmin}, vmax={vmax}")

    cmap_obj = plt.get_cmap("magma").copy()
    cmap_obj.set_bad(color="white")

    arr_masked = np.ma.masked_invalid(arr)
    normalized = (arr_masked - vmin) / (vmax - vmin)
    normalized = np.ma.clip(normalized, 0, 1)

    rgba = cmap_obj(normalized)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# ============================================================
# Orientation (matches the training image pipeline)
# ============================================================

def get_transform_info(shape):
    original_shape = tuple(shape)
    was_transposed = original_shape[1] > original_shape[0]
    transformed_shape = (
        (original_shape[1], original_shape[0]) if was_transposed else original_shape
    )

    return {
        "original_shape": original_shape,
        "transformed_shape": transformed_shape,
        "was_transposed": was_transposed,
    }


def orient_rgb_for_model(rgb, was_transposed):
    if was_transposed:
        rgb = np.transpose(rgb, (1, 0, 2))
    return np.fliplr(rgb)


# ============================================================
# YOLO detection
# ============================================================

def run_yolo_detection(model, pil_image, conf, imgsz, device):
    """Run an Ultralytics YOLO model on an oriented RGB image.

    Returns a list of {"score": float, "xyxy": [x1, y1, x2, y2]} dicts in the
    oriented model-input image's pixel space.
    """
    results = model.predict(
        source=pil_image, conf=conf, imgsz=imgsz, device=device, verbose=False
    )
    boxes = results[0].boxes

    xyxy = boxes.xyxy.cpu().numpy().tolist()
    scores = boxes.conf.cpu().numpy().tolist()

    return [
        {"score": float(score), "xyxy": [float(v) for v in box]}
        for box, score in zip(xyxy, scores)
    ]


# ============================================================
# Detections back to the original SWOT grid
# ============================================================

def detections_to_original_mask(detections, transform_info):
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


def detections_to_original_confidence_mask(detections, transform_info):
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
    confidence_original = confidence_unflipped.T if was_transposed else confidence_unflipped

    if confidence_original.shape != original_shape:
        raise ValueError(
            f"Confidence mask shape mismatch. Expected {original_shape}, "
            f"got {confidence_original.shape}"
        )

    return confidence_original.astype(np.float32)


# ============================================================
# Outputs: YOLO-format label txt + output NetCDF
# ============================================================

def write_yolo_label_txt(detections, transform_info, out_path):
    """Write standard YOLO-format labels (cls cx cy w h conf), normalized to
    the oriented model-input image size. Writes nothing if there are no
    detections, matching Ultralytics' own save_txt behavior."""
    if not detections:
        return

    height, width = transform_info["transformed_shape"]
    lines = []

    for det in detections:
        x1, y1, x2, y2 = det["xyxy"]
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {det['score']:.6f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


# Kept in every output file regardless of variant, so the generic evaluator
# (eval/evaluate_block_metrics.py, default --ground-truth-var block_mask /
# --background-var gaussian_filter_highpass) runs against any of these files
# with zero extra flags. Everything else from the ~30-variable source file is
# deliberately dropped -- it's still available in the original NetCDF by
# filename, and duplicating it per-model was what exhausted the volume quota
# (~26GB across 30 models for one split pair).
CARRIED_OVER_VARS = ["block_mask", "iw_bbox_mask", "gaussian_filter_highpass"]

# ~2-6x smaller on top of dropping the other ~28 variables; masks (mostly
# zero) and smooth SSHA-derived fields both compress well.
COMPRESSION_ENCODING = {"zlib": True, "complevel": 4}


def write_output_netcdf(
    ds,
    mask,
    confidence_mask,
    model_path,
    conf,
    variant_name,
    mask_var_name,
    conf_mask_var_name,
    out_path,
    base_var="ssha_quality_masked",
):
    """Write a slim NetCDF containing lat/lon, the ground-truth/background
    variables needed by the shared evaluator, and the new detection mask +
    confidence mask -- not a full duplicate of the ~30-variable source
    dataset."""
    base_da = ds[base_var].squeeze()
    dims = base_da.dims
    coords = {dim: base_da.coords[dim] for dim in dims if dim in base_da.coords}

    data_vars = {}
    for var_name in CARRIED_OVER_VARS:
        if var_name in ds:
            data_vars[var_name] = ds[var_name].squeeze()

    data_vars[mask_var_name] = xr.DataArray(
        mask.astype(np.uint8),
        dims=dims,
        coords=coords,
        name=mask_var_name,
        attrs={
            "description": "Pixels inside a detected YOLO11 internal-wave bounding box.",
            "mask_values": "1 = inside a detected bounding box, 0 = outside",
        },
    )

    data_vars[conf_mask_var_name] = xr.DataArray(
        confidence_mask.astype(np.float32),
        dims=dims,
        coords=coords,
        name=conf_mask_var_name,
        attrs={
            "description": (
                "YOLO11 detection confidence assigned to pixels inside internal-wave "
                "bounding boxes. Overlapping boxes use the highest confidence."
            ),
            "valid_range": np.array([0.0, 1.0], dtype=np.float32),
            "mask_values": (
                "0 = outside all detected boxes; values in (0, 1] = maximum detection "
                "confidence covering the pixel"
            ),
        },
    )

    ds_out = xr.Dataset(data_vars, coords={"latitude": ds["latitude"], "longitude": ds["longitude"]})

    ds_out.attrs["yolo11_model_path"] = str(model_path)
    ds_out.attrs["yolo11_confidence_threshold"] = float(conf)
    ds_out.attrs["yolo11_variant"] = str(variant_name)
    ds_out.attrs["yolo11_mask_variable"] = str(mask_var_name)
    ds_out.attrs["yolo11_confidence_mask_variable"] = str(conf_mask_var_name)
    ds_out.attrs["yolo11_source_l3_file"] = ds.attrs.get("source_l3_file", "")
    ds_out.attrs["yolo11_note"] = (
        "Slim output: only lat/lon, ground-truth (block_mask, iw_bbox_mask), the "
        "eval-pipeline background variable (gaussian_filter_highpass), and this "
        "model's detection outputs are included. The full set of source variables "
        "is available in the original file at yolo11_source_l3_file."
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: COMPRESSION_ENCODING for name in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
