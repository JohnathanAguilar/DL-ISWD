# augment.py
# Optional pre-collation augmentation transforms for BlockItemDataset.

import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class RandomFlip:
    """
    Randomly flip an item's image/labels/valid arrays horizontally and/or
    vertically, with independent per-call probabilities.

    Expects (and preserves) the item dict shape produced by dataset.py:
        {
          "image": np.ndarray [C, H, W] float32,
          "labels": np.ndarray [Bh, Bw] uint8,
          "valid":  np.ndarray [Bh, Bw] bool,
          ... (other keys pass through unchanged)
        }

    Horizontal flip mirrors axis=-1 (image width / labels & valid Bw);
    vertical flip mirrors axis=-2 (image height / labels & valid Bh). Since
    dataset.py always crops H/W to an exact multiple of block_size, both
    flips are shape-safe for every item.
    """

    hflip_prob: float = 0.5
    vflip_prob: float = 0.5

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        do_hflip = random.random() < self.hflip_prob
        do_vflip = random.random() < self.vflip_prob

        if not do_hflip and not do_vflip:
            return item

        out = dict(item)

        for key in ("image", "labels", "valid"):
            arr = out[key]
            if do_hflip:
                arr = np.flip(arr, axis=-1)
            if do_vflip:
                arr = np.flip(arr, axis=-2)
            out[key] = np.ascontiguousarray(arr)

        return out
