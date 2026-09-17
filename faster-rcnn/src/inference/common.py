"""Shared output-writing helpers for the Faster R-CNN NetCDF inference CLI
(``infer.py``).

Both detector classes (``GaussianHighpassFasterRCNNDetector``,
``RGBCompositeFasterRCNNDetector``) already know how to run detection and
merge a mask + confidence variable into a *full* copy of the source dataset
(``add_outputs_to_dataset`` does ``ds_out = clipped_ds.copy()``). This module
adds the missing piece: a slim writer that keeps only what
``eval/evaluate_block_metrics.py`` needs, mirroring
``yolo11/src/inference/common.py::write_output_netcdf`` /
``CARRIED_OVER_VARS`` -- dropping the other ~28 source variables is what
keeps output NetCDFs small instead of duplicating the full source file per
model per split.
"""

from pathlib import Path

import xarray as xr


def write_yolo_label_txt(detections, transform_info, out_path):
    """Write standard YOLO-format labels (cls cx cy w h conf), normalized to
    the oriented model-input image size. Writes nothing if there are no
    detections, matching Ultralytics' own save_txt behavior.

    Identical to yolo11's ``src/inference/common.py::write_yolo_label_txt`` --
    only reads ``det["score"]``/``det["xyxy"]`` and
    ``transform_info["transformed_shape"]``, which every detector family's
    ``detections`` list and ``transform_info_from_shape()`` both provide.
    """
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
# with zero extra flags. Everything else from the source file is dropped --
# it's still available by filename via the source_l3_file attribute, and
# duplicating it per model is what exhausted the disk quota for yolo11's
# equivalent pipeline earlier this session.
CARRIED_OVER_VARS = ["block_mask", "iw_bbox_mask", "gaussian_filter_highpass"]

COMPRESSION_ENCODING = {"zlib": True, "complevel": 4}


def write_slim_netcdf(
    ds_with_outputs,
    mask_var_name,
    conf_mask_var_name,
    model_path,
    conf,
    variant_name,
    out_path,
    family_tag="frcnn",
):
    """Write a slim NetCDF from the full dataset ``detect_on_l3_file``
    returns: lat/lon, the ground-truth/background variables needed by the
    shared evaluator, and this run's mask + confidence variables -- not a
    full duplicate of the source dataset.

    ``ds_with_outputs`` is whatever ``detect_on_l3_file`` returned (its mask
    and confidence variables are already merged in under ``mask_var_name``/
    ``conf_mask_var_name``); this function only re-packages it, it doesn't
    need to know how those arrays were computed.
    """
    data_vars = {}
    for var_name in CARRIED_OVER_VARS:
        if var_name in ds_with_outputs:
            data_vars[var_name] = ds_with_outputs[var_name].squeeze()

    data_vars[mask_var_name] = ds_with_outputs[mask_var_name].squeeze()
    data_vars[conf_mask_var_name] = ds_with_outputs[conf_mask_var_name].squeeze()

    ds_out = xr.Dataset(
        data_vars,
        coords={
            "latitude": ds_with_outputs["latitude"],
            "longitude": ds_with_outputs["longitude"],
        },
    )

    ds_out.attrs[f"{family_tag}_model_path"] = str(model_path)
    ds_out.attrs[f"{family_tag}_confidence_threshold"] = float(conf)
    ds_out.attrs[f"{family_tag}_variant"] = str(variant_name)
    ds_out.attrs[f"{family_tag}_mask_variable"] = str(mask_var_name)
    ds_out.attrs[f"{family_tag}_confidence_mask_variable"] = str(conf_mask_var_name)
    ds_out.attrs[f"{family_tag}_source_l3_file"] = ds_with_outputs.attrs.get(
        "source_l3_file", ""
    )
    ds_out.attrs[f"{family_tag}_note"] = (
        "Slim output: only lat/lon, ground-truth (block_mask, iw_bbox_mask), the "
        "eval-pipeline background variable (gaussian_filter_highpass), and this "
        "model's detection outputs are included. The full set of source variables "
        f"is available in the original file at {family_tag}_source_l3_file."
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: COMPRESSION_ENCODING for name in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
