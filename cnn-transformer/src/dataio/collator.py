# datacollator.py
# Collate a list of dicts from dataset.py into a padded batch suitable for training.
# - Images are stacked to [B, C, H_max, W_max]
# - Labels & valid are padded to [B, Bh_max, Bw_max]
# - A block-level attention mask is provided to exclude padded tiles in loss/attention

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _to_torch_image(x: np.ndarray) -> torch.Tensor:
    """
    x: np.ndarray [C, H, W], float32
    returns: torch.Tensor [C, H, W], float32 contiguous
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("image must be a numpy array")
    if x.ndim != 3:
        raise ValueError(f"image must have shape [C, H, W]; got {x.shape}")
    if x.shape[0] <= 0:
        raise ValueError(f"image must have at least one channel; got {x.shape}")

    t = torch.from_numpy(x)
    if t.dtype != torch.float32:
        t = t.float()
    return t.contiguous()


def _to_torch_labels(x: np.ndarray, as_float: bool) -> torch.Tensor:
    """
    x: np.ndarray [Bh, Bw], uint8
    returns: torch.Tensor [Bh, Bw], dtype float32 for BCE or long for CE
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("labels must be a numpy array")
    if x.ndim != 2:
        raise ValueError(f"labels must have shape [Bh, Bw]; got {x.shape}")

    t = torch.from_numpy(x)
    if as_float:
        # Convert to float in {0.0, 1.0}
        if t.dtype != torch.float32:
            t = t.float()
    else:
        # Keep integer class labels
        if t.dtype != torch.int64:
            t = t.long()
    return t.contiguous()


def _to_torch_valid(x: np.ndarray) -> torch.Tensor:
    """
    x: np.ndarray [Bh, Bw], bool
    returns: torch.BoolTensor [Bh, Bw]
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("valid must be a numpy array")
    if x.ndim != 2:
        raise ValueError(f"valid must have shape [Bh, Bw]; got {x.shape}")

    t = torch.from_numpy(x.astype(np.bool_))
    return t.contiguous()


@dataclass
class DataCollatorBlocks:
    """
    Collate function for block-token training.

    Expected per-item dict from dataset.py:
        {
          "image": np.ndarray [C, H, W] float32,
          "labels": np.ndarray [Bh, Bw] uint8 (0/1),
          "valid":  np.ndarray [Bh, Bw] bool,
          "id": str,
          "block_size": (bh, bw),
          "grid_shape": (Bh, Bw)
        }

    Batch outputs:
        {
          "images": torch.FloatTensor [B, C, H_max, W_max],
          "labels": torch.Tensor      [B, Bh_max, Bw_max], dtype=float32 or long,
          "valid":  torch.BoolTensor  [B, Bh_max, Bw_max],
          "block_attn_mask": torch.BoolTensor [B, Bh_max, Bw_max] or [B, Bh_max*Bw_max],
          "ids":    List[str],
          "grid_shape": torch.LongTensor [B, 2],
          "block_size": Tuple[int, int]
        }

    Notes:
    - This collator supports any fixed channel count C within a batch.
    - Images in the same batch must have the same number of channels.
    - H/W and Bh/Bw may differ across items and will be padded.
    """

    pad_images: bool = True
    image_pad_value: float = 0.0

    labels_as_float: bool = True  # True for BCE/Logits; False for CE
    pad_labels_value: int = 0     # when labels_as_float=True, this becomes 0.0

    make_block_attention_mask: bool = True
    flatten_block_attention_mask: bool = False  # True -> [B, Bh_max*Bw_max]; False -> [B, Bh_max, Bw_max]

    enforce_same_block_size: bool = True
    device: Optional[torch.device] = None  # leave None to keep tensors on CPU; move later in training loop

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch) == 0:
            raise ValueError("Empty batch passed to DataCollatorBlocks")

        # --- Collect shapes & basic checks ---
        imgs: List[torch.Tensor] = []
        lbls: List[torch.Tensor] = []
        valids: List[torch.Tensor] = []
        ids: List[str] = []
        grid_shapes: List[Tuple[int, int]] = []
        block_sizes: List[Tuple[int, int]] = []

        Cs, Hs, Ws = [], [], []
        Bhs, Bws = [], []

        for item in batch:
            # Required keys
            for k in ("image", "labels", "valid", "id", "grid_shape", "block_size"):
                if k not in item:
                    raise KeyError(f"Missing key '{k}' in item for collation")

            # Convert to torch
            ti = _to_torch_image(item["image"])
            tl = _to_torch_labels(item["labels"], as_float=self.labels_as_float)
            tv = _to_torch_valid(item["valid"])

            # Shapes
            C, H, W = ti.shape
            Bh, Bw = tl.shape

            if tv.shape != tl.shape:
                raise ValueError(
                    f"valid shape must match labels shape; got valid {tuple(tv.shape)} "
                    f"and labels {tuple(tl.shape)}"
                )

            # Book-keeping
            imgs.append(ti)
            lbls.append(tl)
            valids.append(tv)
            ids.append(str(item["id"]))
            grid_shapes.append((Bh, Bw))
            block_sizes.append(tuple(item["block_size"]))

            Cs.append(C)
            Hs.append(H)
            Ws.append(W)
            Bhs.append(Bh)
            Bws.append(Bw)

        # Enforce consistent channel count within a batch.
        if len(set(Cs)) != 1:
            raise ValueError(f"Mismatched channel counts in batch: {Cs}")

        # Enforce block_size consistency, optional.
        if self.enforce_same_block_size:
            first_bs = block_sizes[0]
            for bs in block_sizes[1:]:
                if bs != first_bs:
                    raise ValueError(f"Mismatched block_size in batch: {block_sizes}")

        # Determine maxima for padding.
        H_max = max(Hs)
        W_max = max(Ws)
        Bh_max = max(Bhs)
        Bw_max = max(Bws)

        # --- Pad images to [C, H_max, W_max] ---
        if self.pad_images:
            padded_imgs = []
            for t in imgs:
                _, H, W = t.shape
                pad_h = H_max - H
                pad_w = W_max - W

                # Pad on bottom/right so existing content stays indexed at [0:H, 0:W].
                # F.pad expects (left, right, top, bottom) for 2-D spatial dims.
                if pad_h == 0 and pad_w == 0:
                    padded_imgs.append(t)
                    continue

                padded = F.pad(
                    t,
                    (0, pad_w, 0, pad_h),
                    value=self.image_pad_value,
                )
                padded_imgs.append(padded)

            images = torch.stack(padded_imgs, dim=0)  # [B, C, H_max, W_max]
        else:
            # If not padding, we require equal spatial shapes.
            if not (len(set(Hs)) == 1 and len(set(Ws)) == 1):
                raise ValueError("pad_images=False but batch contains mixed image sizes")
            images = torch.stack(imgs, dim=0)  # [B, C, H, W]

        # --- Pad labels/valid to [Bh_max, Bw_max] ---
        padded_lbls, padded_valids = [], []
        for tl, tv in zip(lbls, valids):
            Bh, Bw = tl.shape
            pad_bh = Bh_max - Bh
            pad_bw = Bw_max - Bw

            if pad_bh or pad_bw:
                # Pad on bottom/right; keep existing tiles at top-left.
                # F.pad expects (left, right, top, bottom) for 2-D tensors.
                if self.labels_as_float:
                    pad_val = float(self.pad_labels_value)
                else:
                    pad_val = int(self.pad_labels_value)

                tl_p = F.pad(tl, (0, pad_bw, 0, pad_bh), value=pad_val)
                tv_p = F.pad(tw := tv, (0, pad_bw, 0, pad_bh), value=False)
            else:
                tl_p = tl
                tv_p = tv

            padded_lbls.append(tl_p)
            padded_valids.append(tv_p)

        labels = torch.stack(padded_lbls, dim=0)  # [B, Bh_max, Bw_max]
        valid = torch.stack(padded_valids, dim=0)  # [B, Bh_max, Bw_max]

        # --- Attention mask over blocks ---
        if self.make_block_attention_mask:
            block_attn_mask = valid.clone()  # True for real tiles, False for padded tiles
            if self.flatten_block_attention_mask:
                block_attn_mask = block_attn_mask.view(
                    block_attn_mask.size(0),
                    -1,
                )  # [B, Bh_max*Bw_max]
        else:
            block_attn_mask = None

        # --- Grid shape tensor & device move ---
        grid_shape_tensor = torch.tensor(grid_shapes, dtype=torch.long)  # [B, 2]

        if self.device is not None:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            valid = valid.to(self.device, non_blocking=True)
            grid_shape_tensor = grid_shape_tensor.to(self.device, non_blocking=True)
            if block_attn_mask is not None:
                block_attn_mask = block_attn_mask.to(self.device, non_blocking=True)

        out: Dict[str, Any] = {
            "images": images,                         # [B, C, H_max, W_max], float32
            "labels": labels,                         # [B, Bh_max, Bw_max], float32 or long
            "valid": valid,                           # [B, Bh_max, Bw_max], bool
            "ids": ids,                               # list[str]
            "grid_shape": grid_shape_tensor,          # [B, 2]
            "block_size": block_sizes[0] if self.enforce_same_block_size else None,
        }

        if block_attn_mask is not None:
            out["block_attn_mask"] = block_attn_mask  # [B, Bh_max, Bw_max] or [B, Bh_max*Bw_max]

        return out