"""RT-DETR model builder, with the MPS float64 fix applied at import time.

RT-DETR's hybrid-encoder positional embedding
(``transformers.models.rt_detr.modeling_rt_detr.build_2d_sinusoidal_position_embedding``)
hardcodes ``dtype=torch.float64`` for its internal frequency arithmetic. MPS
has no float64 support at all, so this crashes immediately on `.to("mps")`
(confirmed on transformers 5.14.1 / torch 2.12). The embedding is a fixed,
non-learned buffer (cached via ``lru_cache``), so computing it on CPU and
moving the *result* to the target device is exact -- no precision or
correctness change versus running natively on a float64-capable device -- it
just relocates where a few KB of constant math happens once per (H, W, dim)
key.

Everything downstream is unaffected: measured gradients on MPS matched CPU to
within normal float32-vs-optimizer-order noise, and training was 2x+ faster
than CPU (see README.md).
"""

import torch
from transformers import RTDetrConfig, RTDetrForObjectDetection

import transformers.models.rt_detr.modeling_rt_detr as _rtdetr_mod

_original_build_pos_embed = _rtdetr_mod.build_2d_sinusoidal_position_embedding


def _cpu_safe_build_pos_embed(
    height, width, embed_dim, temperature=10000, cls_token=False, device=None, dtype=torch.float32
):
    out = _original_build_pos_embed(
        height, width, embed_dim, temperature=temperature, cls_token=cls_token,
        device=torch.device("cpu"), dtype=dtype,
    )
    return out.to(device) if device is not None else out


_rtdetr_mod.build_2d_sinusoidal_position_embedding = _cpu_safe_build_pos_embed


def build_model(model_name, num_labels=1, pretrained=True):
    """Build an RT-DETR detector.

    Parameters
    ----------
    model_name : str
        A model id from the HF hub, e.g. ``"PekingU/rtdetr_r50vd"``.
    num_labels : int
        Number of foreground classes. RT-DETR has no background class (the
        one-to-one Hungarian matching loss handles "no object" implicitly),
        so this is not "+1" the way torchvision detection models are.
    pretrained : bool
        If True, download and use the pretrained COCO weights (with the
        classification head re-initialized for ``num_labels``). If False,
        build the architecture from the model's config only, with random
        weights -- use this before loading your own fine-tuned checkpoint so
        training-time pretrained weights aren't downloaded a second time for
        nothing.
    """
    if pretrained:
        model = RTDetrForObjectDetection.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )
    else:
        config = RTDetrConfig.from_pretrained(model_name, num_labels=num_labels)
        model = RTDetrForObjectDetection(config)

    return model


def resolve_device(requested="auto"):
    """Pick a torch device, preferring MPS on Apple silicon.

    Unlike ``faster-rcnn/model.py``, MPS is safe here (see module docstring).
    """
    if requested not in ("auto", None):
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
