"""Run the CNN-Transformer block classifier over SWOT L3 files in vertical
along-track tiles instead of whole swaths.

The model's sincos2d positional encoding and full self-attention make its
per-block output sensitive to the along-track grid height. Feeding one 9860-row
swath at a time (a ~493x3 block grid) produced an almost entirely negative map;
feeding 700-row tiles (a 35x3 grid each) makes the model respond. This driver:

  1. crops ``--crop-bottom`` rows off the bottom of each file (9860 -> 9800),
  2. splits the remaining swath into ``--tile-height`` row tiles (14 x 700),
  3. runs the model once per tile,
  4. stacks the tile grids back into one full-swath grid,
  5. writes a slim NetCDF at the cropped 9800x60 size, via
     ``inference/common.py::write_slim_netcdf``.

With ``--dense`` the model's dense head is used (``return_dense=True``): each tile
returns a per-pixel ``[tile_h, 60]`` probability map, the 14 tiles stack straight
into a ``9800x60`` per-pixel field (no block broadcast), and the checkpoint must
be a dense-head model (its embedded config carries ``dense_head: True``).

The source files carry no ``block_mask``/``iw_bbox_mask`` ground truth, so
``base_var`` is the input channel and only ``gaussian_filter_highpass`` is
carried over.

Example
-------
    cd cnn-transformer
    KMP_DUPLICATE_LIB_OK=TRUE PYTORCH_ENABLE_MPS_FALLBACK=1 \\
      python src/inference/infer_tiled.py
    python src/inference/infer_tiled.py --tile-height 980 --device cpu --limit 5
    python src/inference/infer_tiled.py --dense \\
      --weights models/dense_head_rolling_mean_highpass_v0.1.0/best.pt \\
      --out-dir results/dense_head_rolling_mean_highpass_v0.1.0/SWOT_L3_Full_Dataset/netcdf

Run from the ``cnn-transformer/`` directory.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import xarray as xr

_INFERENCE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _INFERENCE_DIR.parent
_MODEL_DIR = _SRC_DIR.parent  # cnn-transformer/

for _p in (_SRC_DIR, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from inference import common
from inference.inference import load_model, load_nc_channels, make_block_inputs

DEFAULT_DATA_DIR = Path(
    "/Users/aguilarj/projects/isw-detection/data/SWOT_L3_Full_Dataset/data"
)
DEFAULT_WEIGHTS = _MODEL_DIR / "models/rolling_mean_higpass_v0.2.0/best.pt"
DEFAULT_OUT_DIR = (
    _MODEL_DIR
    / "results/rolling_mean_higpass_v0.2.0/SWOT_L3_Full_Dataset/netcdf"
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--channel-vars", nargs="+", default=["rolling_mean_highpass"])
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tile-height", type=int, default=700,
                   help="Along-track (num_lines) rows fed to the model at once.")
    p.add_argument("--crop-bottom", type=int, default=60,
                   help="Rows removed from the bottom of each swath before tiling.")
    p.add_argument("--block-size", type=int, default=20)
    p.add_argument("--dense", action="store_true",
                   help="Use the model's dense head (return_dense=True): stitch "
                        "per-pixel [tile_h, W] maps instead of a block grid. The "
                        "checkpoint must be a dense-head model.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--variant", default="rmh")
    p.add_argument("--device", default="mps", help="mps | cpu | cuda | cuda:N")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N files (debug).")
    return p.parse_args(argv)


def tile_bounds(target_h: int, tile_height: int):
    """Disjoint [start, end) row spans covering [0, target_h)."""
    return [(s, min(s + tile_height, target_h))
            for s in range(0, target_h, tile_height)]


def infer_one(model, device, channel_raws, bounds, block_size, dense=False):
    """Run the model on each row tile and stack the results along the row axis.

    Returns a ``(sum(tile_h) // block_size, W // block_size)`` block-probability
    grid, or -- with ``dense=True`` -- a ``(sum(tile_h), W)`` per-pixel map.
    """
    tiles = []
    for s, e in bounds:
        sub = [c[s:e, :] for c in channel_raws]
        cc, (bh, bw) = make_block_inputs(sub, block_size)
        img_t = torch.from_numpy(np.stack(cc, axis=0)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t, block_mask=None, return_dense=dense,
                        return_heatmap=False)
        if dense:
            dp = out["dense_probs"][0, 0].detach().cpu().numpy()
            expected = (bh * block_size, bw * block_size)
            if dp.shape != expected:
                raise ValueError(
                    f"tile [{s}:{e}] dense map {dp.shape} != expected {expected}"
                )
            tiles.append(dp.astype(np.float32, copy=False))
        else:
            p = out["probs"][0].detach().cpu().numpy()
            if p.size != bh * bw:
                raise ValueError(
                    f"tile [{s}:{e}] token count {p.size} != grid {bh}x{bw}"
                )
            tiles.append(p.reshape(bh, bw).astype(np.float32, copy=False))
    return np.concatenate(tiles, axis=0)


def main(argv=None):
    args = parse_args(argv)
    block = args.block_size

    if args.tile_height % block:
        raise ValueError(f"--tile-height {args.tile_height} not a multiple of {block}")

    files = sorted(args.data_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No .nc files in {args.data_dir.resolve()}")
    if args.limit:
        files = files[: args.limit]

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"files={len(files)} device={device} weights={args.weights} "
          f"mode={'dense' if args.dense else 'block'}")
    print(f"channel_vars={args.channel_vars} tile_height={args.tile_height} "
          f"crop_bottom={args.crop_bottom} block_size={block} threshold={args.threshold}")
    print(f"out_dir={args.out_dir.resolve()}")

    model, _ = load_model(str(args.weights), device, block_size=block)
    model.eval()

    n_ok = n_fail = 0

    for i, path in enumerate(files, start=1):
        stem = path.stem
        out_nc = args.out_dir / f"{stem}.nc"
        try:
            channel_raws = load_nc_channels(str(path), args.channel_vars)

            src_h, src_w = channel_raws[0].shape
            if src_w % block:
                raise ValueError(f"width {src_w} not a multiple of {block}")
            target_h = src_h - args.crop_bottom
            if target_h <= 0:
                raise ValueError(f"crop_bottom {args.crop_bottom} >= height {src_h}")
            target_h -= target_h % block  # keep the tiled region block-aligned

            channel_raws = [c[:target_h, :] for c in channel_raws]
            bounds = tile_bounds(target_h, args.tile_height)

            probs_grid = infer_one(
                model, device, channel_raws, bounds, block, dense=args.dense
            )
            pred = probs_grid >= args.threshold

            with xr.open_dataset(path) as opened:
                ds = opened.load()
            ds.attrs["source_l3_file"] = str(path)
            ds = ds.isel(num_lines=slice(0, target_h))

            base_var = args.channel_vars[0]
            full_shape = ds[base_var].squeeze().shape

            if args.dense:
                # probs_grid is already at pixel resolution.
                conf_tile = probs_grid
                mask_tile = pred.astype(np.uint8)
            else:
                conf_tile = np.repeat(
                    np.repeat(probs_grid, block, axis=0), block, axis=1
                )
                mask_tile = np.repeat(
                    np.repeat(pred.astype(np.uint8), block, axis=0), block, axis=1
                )
            crop_h, crop_w = conf_tile.shape

            confidence_full = np.full(full_shape, np.nan, dtype=np.float32)
            confidence_full[:crop_h, :crop_w] = conf_tile
            mask_full = np.zeros(full_shape, dtype=np.uint8)
            mask_full[:crop_h, :crop_w] = mask_tile

            common.write_slim_netcdf(
                ds,
                mask_full,
                confidence_full,
                mask_var_name="internal_wave_bbox_mask",
                conf_mask_var_name="internal_wave_bbox_confidence",
                model_path=args.weights,
                conf_threshold=args.threshold,
                variant_name=args.variant,
                out_path=out_nc,
                base_var=base_var,
            )

            n_ok += 1
            if i % 25 == 0 or i == len(files):
                unit = "pixel" if args.dense else "block"
                print(f"[{i}/{len(files)}] {stem}: "
                      f"{int(pred.sum())}/{pred.size} positive {unit}(s) "
                      f"| grid {probs_grid.shape} | {len(bounds)} tiles")
        except Exception as exc:  # noqa: BLE001 - report, drop stale output, continue
            n_fail += 1
            print(f"[{i}/{len(files)}] {stem}: FAILED: {exc}")
            traceback.print_exc()
            if out_nc.exists():
                out_nc.unlink()
                print(f"    removed stale {out_nc.name}")

    print(f"\ndone. ok={n_ok} failed={n_fail} total={len(files)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
