"""Shared output-writing helpers for the CNN-Transformer NetCDF inference CLI
(``infer.py``).

``inference()`` in this package (``inference/inference.py``) never touches the
source NetCDF at all -- it returns an in-memory ``[Bh, Bw]`` block-probability
grid and nothing is written to disk. This module is the missing piece: given
the raw source dataset plus a pixel-resolution mask/confidence array already
broadcast from that block grid, write a slim NetCDF that keeps only what
``eval/evaluate_block_metrics.py`` needs, mirroring
``yolo11/src/inference/common.py::write_output_netcdf`` /
``CARRIED_OVER_VARS`` -- dropping the other source variables is what keeps
output NetCDFs small instead of duplicating the full source file per model
per split.
"""

from pathlib import Path

import numpy as np
import xarray as xr

# Kept in every output file regardless of variant, so the generic evaluator
# (eval/evaluate_block_metrics.py, default --ground-truth-var block_mask /
# --background-var gaussian_filter_highpass) runs against any of these files
# with zero extra flags. quality_flag is carried too so downstream analysis can
# screen predictions by SWOT data quality without reopening the full source.
# Everything else from the source file is dropped -- it's still available by
# filename via the source_l3_file attribute, and duplicating it per model is
# what exhausted the disk quota for yolo11's equivalent pipeline earlier this
# session.
CARRIED_OVER_VARS = [
    "block_mask",
    "iw_bbox_mask",
    "gaussian_filter_highpass",
    "quality_flag",
]

COMPRESSION_ENCODING = {"zlib": True, "complevel": 4}


def write_slim_netcdf(
    ds,
    mask,
    confidence,
    mask_var_name,
    conf_mask_var_name,
    model_path,
    conf_threshold,
    variant_name,
    out_path,
    base_var="block_mask",
    family_tag="cnntrans",
):
    """Write a slim NetCDF: lat/lon, the ground-truth/background variables
    needed by the shared evaluator, and this run's mask + confidence
    variables -- not a full duplicate of the source dataset.

    ``ds`` is the raw dataset opened directly from the source ``.nc`` file
    (there is no ``detect_on_l3_file``-style merged copy to slim down here).
    ``mask``/``confidence`` are pixel-resolution numpy arrays already
    broadcast from the model's ``[Bh, Bw]`` block grid, matching ``ds[base_var]``'s
    shape.
    """
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
            "description": "Pixels inside a 20x20 block the CNN-Transformer classified as internal-wave-positive.",
            "mask_values": "1 = predicted-positive block, 0 = predicted-negative or outside the tiled region",
        },
    )

    data_vars[conf_mask_var_name] = xr.DataArray(
        confidence.astype(np.float32),
        dims=dims,
        coords=coords,
        name=conf_mask_var_name,
        attrs={
            "description": (
                "CNN-Transformer per-block internal-wave probability, broadcast to pixel "
                "resolution (every pixel in a block shares that block's score)."
            ),
            "valid_range": np.array([0.0, 1.0], dtype=np.float32),
            "mask_values": "NaN outside the floor-tiled region the model was run over",
        },
    )

    ds_out = xr.Dataset(
        data_vars,
        coords={"latitude": ds["latitude"], "longitude": ds["longitude"]},
    )

    ds_out.attrs[f"{family_tag}_model_path"] = str(model_path)
    ds_out.attrs[f"{family_tag}_confidence_threshold"] = float(conf_threshold)
    ds_out.attrs[f"{family_tag}_variant"] = str(variant_name)
    ds_out.attrs[f"{family_tag}_mask_variable"] = str(mask_var_name)
    ds_out.attrs[f"{family_tag}_confidence_mask_variable"] = str(conf_mask_var_name)
    ds_out.attrs[f"{family_tag}_source_l3_file"] = ds.attrs.get("source_l3_file", "")
    ds_out.attrs[f"{family_tag}_note"] = (
        "Slim output: only lat/lon, ground-truth (block_mask, iw_bbox_mask), the "
        "eval-pipeline background variable (gaussian_filter_highpass), quality_flag, "
        "and this model's detection outputs are included. The full set of source "
        f"variables is available in the original file at {family_tag}_source_l3_file."
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {name: COMPRESSION_ENCODING for name in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
