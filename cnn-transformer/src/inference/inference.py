from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import xarray as xr

try:
    from models import ModelConfig, BlocksTokenModel
except ImportError:  # Useful when running this file beside the model files during debugging.
    from model_config import ModelConfig  # type: ignore
    from token_model import BlocksTokenModel  # type: ignore

try:
    from dataio.dataset import _ensure_block_size
except ImportError:  # Useful when running this file beside dataset.py during debugging.
    from dataset import _ensure_block_size  # type: ignore


def _checkpoint_state_dict(ckpt: Any) -> Mapping[str, torch.Tensor]:
    """
    Return a model state_dict from common checkpoint formats.
    """
    if isinstance(ckpt, Mapping):
        for key in ("model_state", "model_state_dict", "state_dict"):
            state = ckpt.get(key)
            if isinstance(state, Mapping):
                return state

        # Some checkpoints are saved as the state_dict itself.
        if ckpt and all(isinstance(k, str) for k in ckpt.keys()):
            return ckpt  # type: ignore[return-value]

    raise KeyError(
        "Could not find a model state dict. Expected one of: "
        "'model_state', 'model_state_dict', 'state_dict', or a raw state_dict checkpoint."
    )


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Handle checkpoints saved from nn.DataParallel / DistributedDataParallel.
    """
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return dict(state_dict)

    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _checkpoint_config(
    ckpt: Any,
    block_size,
    model_cfg_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build ModelConfig kwargs from the checkpoint plus explicit overrides.
    """
    cfg_dict: dict[str, Any] = {}

    if isinstance(ckpt, Mapping):
        raw_cfg = ckpt.get("config", ckpt.get("model_config", {}))
        if isinstance(raw_cfg, Mapping):
            cfg_dict.update(dict(raw_cfg))
        elif hasattr(raw_cfg, "__dict__"):
            cfg_dict.update(vars(raw_cfg))

    if model_cfg_kwargs:
        cfg_dict.update(model_cfg_kwargs)

    # Some training checkpoints store model_state but not config, so this gives
    # ModelConfig at least the inference block size when the caller passes it.
    if block_size is not None:
        cfg_dict.setdefault("block_size", block_size)

    return cfg_dict


def load_nc_channels(
    nc_path: str,
    channel_vars: Sequence[str],
) -> list[np.ndarray]:
    """
    Load one or more 2-D channels from a NetCDF file.

    Args:
        nc_path:
            Path to a .nc file.
        channel_vars:
            Names of the channel variables to load.
            Can contain 1, 2, 3, or more names.

    Returns:
        channels:
            List of float32 arrays, each with shape [H, W].
    """
    channel_vars = tuple(channel_vars)

    if len(channel_vars) == 0:
        raise ValueError("channel_vars must contain at least one variable name")

    with xr.open_dataset(nc_path) as ds:
        missing = [name for name in channel_vars if name not in ds]
        if missing:
            available = list(ds.variables)
            raise KeyError(
                f"Missing channel variable(s) in {nc_path}: {missing}. "
                f"Available variables include: {available}"
            )

        channels = [
            np.asarray(ds[name].values, dtype=np.float32)
            for name in channel_vars
        ]

    for name, arr in zip(channel_vars, channels):
        if arr.ndim != 2:
            raise ValueError(
                f"Input channel '{name}' must be a 2-D array; got shape {arr.shape}."
            )

    ref_shape = channels[0].shape
    bad_shapes = [
        f"{name}{arr.shape}"
        for name, arr in zip(channel_vars, channels)
        if arr.shape != ref_shape
    ]

    if bad_shapes:
        all_shapes = ", ".join(
            f"{name}{arr.shape}"
            for name, arr in zip(channel_vars, channels)
        )
        raise ValueError(f"Input channel shape mismatch: {all_shapes}.")

    return channels


def make_block_inputs(
    channel_raws: Sequence[np.ndarray],
    block_size,
) -> tuple[list[np.ndarray], tuple[int, int]]:
    """
    Build cropped/sanitized model inputs without any token/valid mask.

    Args:
        channel_raws:
            Sequence of raw 2-D input channels. Each channel must have shape [H, W].
        block_size:
            Pixel block size used during training.

    Returns:
        channels_cropped:
            List of sanitized float32 arrays, each with shape [Hc, Wc].
        grid_shape:
            Tuple (Bh, Bw), where Bh and Bw are the block-token grid dimensions.
    """
    channel_raws = list(channel_raws)

    if len(channel_raws) == 0:
        raise ValueError("channel_raws must contain at least one channel")

    for idx, arr in enumerate(channel_raws):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"channel_raws[{idx}] must be a numpy array")
        if arr.ndim != 2:
            raise ValueError(
                f"channel_raws[{idx}] must be a 2-D array; got shape {arr.shape}"
            )

    ref_shape = channel_raws[0].shape
    bad_shapes = [
        f"channel_raws[{idx}]{arr.shape}"
        for idx, arr in enumerate(channel_raws)
        if arr.shape != ref_shape
    ]

    if bad_shapes:
        all_shapes = ", ".join(
            f"channel_raws[{idx}]{arr.shape}"
            for idx, arr in enumerate(channel_raws)
        )
        raise ValueError(f"Input channel shape mismatch: {all_shapes}")

    bh, bw = _ensure_block_size(block_size)

    H, W = ref_shape
    Bh = H // bh
    Bw = W // bw

    if Bh == 0 or Bw == 0:
        raise ValueError(
            f"Block size ({bh}, {bw}) is larger than input shape ({H}, {W})."
        )

    # Same floor-tiling rule as training: drop ragged bottom/right edges.
    crop_h = Bh * bh
    crop_w = Bw * bw

    # Replace NaN/Inf with 0.0 for the actual tensor passed to the model.
    # No validity/token mask is computed or used.
    channels_cropped = [
        np.nan_to_num(
            arr[:crop_h, :crop_w],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        for arr in channel_raws
    ]

    return channels_cropped, (Bh, Bw)


def load_model(
    model_ckpt: str,
    device: torch.device,
    block_size=None,
    model_cfg_kwargs: Optional[dict[str, Any]] = None,
    strict: bool = True,
):
    """
    Load BlocksTokenModel from a checkpoint.
    """
    ckpt = torch.load(model_ckpt, map_location="cpu")
    cfg_kwargs = _checkpoint_config(ckpt, block_size, model_cfg_kwargs)
    cfg = ModelConfig(**cfg_kwargs)

    model = BlocksTokenModel(cfg)
    state_dict = _checkpoint_state_dict(ckpt)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        # Retry once for checkpoints saved with a `module.` prefix.
        model.load_state_dict(_strip_module_prefix(state_dict), strict=strict)

    return model.to(device).eval(), cfg


def infer_model_config_from_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    **overrides: Any,
) -> dict[str, Any]:
    """
    Recover ModelConfig fields from a checkpoint's tensor shapes.

    Checkpoints saved before configs were stored carry no architecture record,
    and getting a field wrong produces a wall of size-mismatch errors. These
    fields are readable straight off the weights:

        in_channels, embed_dim, depth, mlp_ratio, positional

    These are NOT recoverable, because they change the computation without
    changing any tensor shape, and fall back to ModelConfig defaults unless
    passed as overrides:

        num_heads, block_size, sra, sra_ratio, drop, attn_drop

    block_size in particular must match training or every downstream grid will
    be wrong, so pass it explicitly.
    """
    import re

    state_dict = _strip_module_prefix(state_dict)

    stem_key = "stem.net.0.weight"
    if stem_key not in state_dict:
        raise KeyError(
            f"Cannot infer config: checkpoint has no '{stem_key}'. "
            f"Sample keys: {list(state_dict)[:5]}"
        )

    embed_dim, in_channels = state_dict[stem_key].shape[:2]

    block_ids = [
        int(m.group(1))
        for k in state_dict
        if (m := re.match(r"encoder\.blocks\.(\d+)\.", k))
    ]
    if not block_ids:
        raise KeyError("Cannot infer config: checkpoint has no encoder blocks.")

    depth = max(block_ids) + 1

    mlp_key = "encoder.blocks.0.mlp.0.weight"
    mlp_ratio = float(state_dict[mlp_key].shape[0]) / float(embed_dim)

    positional = "learned2d" if any(k.startswith("posenc.") for k in state_dict) else "sincos2d"

    cfg: dict[str, Any] = {
        "in_channels": int(in_channels),
        "embed_dim": int(embed_dim),
        "depth": int(depth),
        "mlp_ratio": mlp_ratio,
        "positional": positional,
    }
    cfg.update(overrides)

    return cfg


def infer_model_config_from_checkpoint(model_ckpt: str, **overrides: Any) -> dict[str, Any]:
    """Convenience wrapper: read a checkpoint file and infer its ModelConfig kwargs."""
    ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    return infer_model_config_from_state_dict(_checkpoint_state_dict(ckpt), **overrides)


def load_backbone_weights(
    model,
    model_ckpt: str,
    require_complete: bool = True,
):
    """
    Load a trained patch-level detector into a dense-head model.

    Stage 1 of dense refinement starts from an existing checkpoint that has no
    dense-head weights, so a strict load would fail. This loads every backbone
    tensor and leaves the dense head at its random initialization.

    Args:
        model:
            A BlocksTokenModel, usually built with ModelConfig(dense_head=True).
        model_ckpt:
            Path to the trained patch-level checkpoint.
        require_complete:
            If True, raise when a non-dense parameter is missing from the
            checkpoint, which would mean the backbone is only partly restored.

    Returns:
        model with backbone weights loaded, plus the load report.
    """
    ckpt = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_checkpoint_state_dict(ckpt))

    try:
        result = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as err:
        # A config mismatch produces one size-mismatch line per tensor, which
        # buries the actual cause. Report the architecture the checkpoint wants.
        inferred = infer_model_config_from_state_dict(state_dict)
        actual = {k: getattr(model.cfg, k) for k in inferred}
        mismatched = {k: (v, actual[k]) for k, v in inferred.items() if v != actual[k]}

        raise RuntimeError(
            "Model config does not match the checkpoint. "
            f"Mismatched fields (checkpoint -> model): "
            f"{ {k: f'{a} -> {b}' for k, (a, b) in mismatched.items()} }. "
            "Build ModelConfig with infer_model_config_from_checkpoint(ckpt, block_size=...) "
            "to read these off the weights automatically. Note that num_heads, "
            "block_size, and sra cannot be inferred and must be set to match training."
        ) from err

    missing_backbone = [k for k in result.missing_keys if not k.startswith("dense_head.")]

    if require_complete and missing_backbone:
        raise RuntimeError(
            "Checkpoint does not contain all backbone weights. Missing: "
            f"{missing_backbone[:10]}{'...' if len(missing_backbone) > 10 else ''}. "
            "Check that ModelConfig matches the checkpoint (in_channels, embed_dim, "
            "depth, block_size, positional)."
        )

    if result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint contains unexpected keys not present in the model: "
            f"{result.unexpected_keys[:10]}"
        )

    return model, result


def dense_inference(
    model_ckpt: str,
    device: torch.device,
    nc_path: str,
    channel_vars: Sequence[str],
    block_size=None,
    threshold: float = 0.5,
    model_cfg_kwargs: Optional[dict[str, Any]] = None,
    strict: bool = True,
) -> dict:
    """
    Run pixel-resolution dense inference for one NetCDF file.

    The checkpoint must come from a model trained with a dense head; pass
    model_cfg_kwargs={"dense_head": True, ...} if the checkpoint predates config
    saving.

    Returns:
        {
            "dense_probs":       FloatTensor [Ht, Wt], one confidence per pixel
            "dense_pred":        BoolTensor  [Ht, Wt]
            "patch_probs":       FloatTensor [Bh, Bw], original detector output
            "patch_pred":        BoolTensor  [Bh, Bw]
            "aggregated_probs":  FloatTensor [Bh, Bw], dense map pooled back to blocks
            "grid_shape":        (Bh, Bw)
            "block_size":        int
            "channel_vars":      tuple[str, ...]
        }

    Ht = Bh * block_size and Wt = Bw * block_size, so the dense map covers exactly
    the region that the block tokenizer sees; ragged bottom/right pixels dropped
    by floor-tiling are not predicted.
    """
    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    channel_vars = tuple(channel_vars)

    if len(channel_vars) == 0:
        raise ValueError("channel_vars must contain at least one variable name")

    cfg_kwargs = dict(model_cfg_kwargs or {})
    cfg_kwargs.setdefault("dense_head", True)

    model, cfg = load_model(
        model_ckpt,
        device,
        block_size=block_size,
        model_cfg_kwargs=cfg_kwargs,
        strict=strict,
    )

    if model.dense_head is None:
        raise ValueError(
            "Loaded model has no dense head. Pass model_cfg_kwargs={'dense_head': True}."
        )

    effective_block_size = block_size if block_size is not None else cfg.block_size

    channel_raws = load_nc_channels(nc_path, channel_vars)
    channels_cropped, (Bh, Bw) = make_block_inputs(channel_raws, effective_block_size)

    img = np.stack(channels_cropped, axis=0).astype(np.float32, copy=False)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t, block_mask=None, return_dense=True)

        model_bh, model_bw = int(out["Bh"]), int(out["Bw"])
        if (model_bh, model_bw) != (Bh, Bw):
            raise ValueError(
                f"Model grid {(model_bh, model_bw)} does not match the preprocessing "
                f"grid {(Bh, Bw)}. Check that inference block_size matches training."
            )

        dense_probs = out["dense_probs"][0, 0].detach().cpu()          # [Ht, Wt]
        patch_probs = out["probs"][0].detach().cpu().reshape(Bh, Bw)   # [Bh, Bw]

    aggregated = torch.nn.functional.avg_pool2d(
        dense_probs[None, None],
        kernel_size=int(effective_block_size),
        stride=int(effective_block_size),
    )[0, 0]

    return {
        "dense_probs": dense_probs,
        "dense_pred": dense_probs >= threshold,
        "patch_probs": patch_probs,
        "patch_pred": patch_probs >= threshold,
        "aggregated_probs": aggregated,
        "grid_shape": (Bh, Bw),
        "block_size": int(effective_block_size),
        "channel_vars": channel_vars,
    }


def inference(
    model_ckpt: str,
    device: torch.device,
    nc_path: str,
    channel_vars: Sequence[str],
    block_size=None,
    threshold: float = 0.5,
    model_cfg_kwargs: Optional[dict[str, Any]] = None,
    strict: bool = True,
    return_probs: bool = True,
    return_dict: bool = False,
):
    """
    Run pure block-token inference for one NetCDF file with no ground-truth mask
    and no token/valid mask.

    This version supports any number of input channels.

    Args:
        model_ckpt:
            Path to a saved checkpoint.
        device:
            torch.device("cuda"), torch.device("mps"), or torch.device("cpu").
        nc_path:
            Path to the .nc file to run inference on.
        channel_vars:
            Names of the 2-D input channel variables.
            Can contain 1, 2, 3, or more names.
            Example:
                ["ssha_unfiltered_rolling_highpass"]
            or:
                ["ssha_unfiltered", "ssha_unfiltered_rolling_highpass"]
        block_size:
            Pixel block size used during training. If None, uses cfg.block_size
            from the checkpoint config.
        threshold:
            Probability threshold used to produce binary block predictions.
        model_cfg_kwargs:
            Required only if the checkpoint does not contain ModelConfig.
            Also useful for overriding config values.
        strict:
            Passed to model.load_state_dict.
        return_probs:
            If True, include the [Bh, Bw] probability grid in tuple output.
        return_dict:
            If True, return a named dictionary instead of a tuple.

    Returns by default:
        pred_block_labels:
            BoolTensor [Bh, Bw], thresholded predictions.
        probs:
            FloatTensor [Bh, Bw], probability for each block.

    If return_probs=False:
        pred_block_labels

    If return_dict=True:
        {
            "pred_block_labels": pred_block_labels,
            "probs": probs,
            "grid_shape": (Bh, Bw),
            "channel_vars": tuple(channel_vars),
        }
    """
    threshold = float(threshold)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    channel_vars = tuple(channel_vars)

    if len(channel_vars) == 0:
        raise ValueError("channel_vars must contain at least one variable name")

    model, cfg = load_model(
        model_ckpt,
        device,
        block_size=block_size,
        model_cfg_kwargs=model_cfg_kwargs,
        strict=strict,
    )

    effective_block_size = block_size if block_size is not None else cfg.block_size

    channel_raws = load_nc_channels(nc_path, channel_vars)
    channels_cropped, (Bh, Bw) = make_block_inputs(
        channel_raws,
        effective_block_size,
    )

    # Stack channels into model input shape:
    #   [C, Hc, Wc] -> [1, C, Hc, Wc]
    img = np.stack(channels_cropped, axis=0).astype(np.float32, copy=False)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # No block_mask is passed. The model/tokenizer should treat all tokens as valid.
        out = model(img_t, block_mask=None, return_heatmap=False)
        probs_flat = out["probs"][0].detach().cpu().numpy()  # [N]

    expected_tokens = Bh * Bw

    if probs_flat.size != expected_tokens:
        model_bh = int(out.get("Bh", Bh))
        model_bw = int(out.get("Bw", Bw))

        raise ValueError(
            "Model output token count does not match preprocessing grid: "
            f"got {probs_flat.size}, expected {expected_tokens} "
            f"from input grid shape {(Bh, Bw)}; model returned grid {(model_bh, model_bw)}. "
            "Check that inference block_size matches the training/model block_size."
        )

    probs_grid = probs_flat.reshape(Bh, Bw).astype(np.float32, copy=False)
    pred_block_labels = probs_grid >= threshold

    pred_labels_t = torch.from_numpy(pred_block_labels.astype(np.bool_, copy=False))
    probs_t = torch.from_numpy(probs_grid)

    if return_dict:
        return {
            "pred_block_labels": pred_labels_t,
            "probs": probs_t,
            "grid_shape": (Bh, Bw),
            "channel_vars": channel_vars,
        }

    if return_probs:
        return pred_labels_t, probs_t

    return pred_labels_t