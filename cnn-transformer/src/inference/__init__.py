# __init__.py

"""
High-level inference API for block-token SWOT model.
"""

from .inference import (
    inference,
    dense_inference,
    load_model,
    load_backbone_weights,
    infer_model_config_from_checkpoint,
    infer_model_config_from_state_dict,
    load_nc_channels,
    make_block_inputs,
)

# Backward-compatible alias for older notebooks/imports.
load_nc_arrays = load_nc_channels

__all__ = [
    "inference",
    "dense_inference",
    "load_model",
    "load_backbone_weights",
    "infer_model_config_from_checkpoint",
    "infer_model_config_from_state_dict",
    "load_nc_channels",
    "make_block_inputs",
    "load_nc_arrays",
]