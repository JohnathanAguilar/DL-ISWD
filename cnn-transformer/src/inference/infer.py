"""Run a trained CNN-Transformer block-classifier model over every NetCDF
file in a SWOT L3 split, writing a copy of the NetCDF with a detection mask
+ confidence mask added.

Same variable-limiting behavior as ``yolo11/src/inference/infer.py``: only
lat/lon, block_mask, iw_bbox_mask, gaussian_filter_highpass, and this run's
mask/confidence variables are kept in the output NetCDF -- not a full
duplicate of the source file (see ``src/inference/common.py::write_slim_netcdf``).

Unlike the YOLO11/Faster R-CNN/RT-DETR detectors, this model (``BlocksTokenModel``)
is a per-20x20-pixel-block binary classifier, not a box detector -- there are
no bounding boxes, so there is no YOLO-format ``.txt`` label output here. The
model's only output is a ``[Bh, Bw]`` per-block probability grid
(``src/inference/inference.py::inference``), broadcast back to pixel
resolution before being written into the NetCDF.

Example
-------
    python src/inference/infer.py --run-name cnntrans_sqm_123 --split test --device 0
    python src/inference/infer.py --run-name cnntrans_shg_456 --split valid --device 1

Outputs land in ``<results-dir>/<run-name>/<split>/netcdf/``. Run from the
``cnn-transformer/`` directory.
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import xarray as xr

_INFERENCE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _INFERENCE_DIR.parent
_MODEL_DIR = _SRC_DIR.parent

for _p in (_SRC_DIR, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from inference import common
    from inference import inference as run_block_inference
except ImportError:
    import common
    from inference import inference as run_block_inference

DEFAULT_NETCDF_ROOT = Path("/workspace/data/bob/netcdf")
DEFAULT_RUNS_DIR = _MODEL_DIR / "runs"
DEFAULT_RESULTS_DIR = _MODEL_DIR / "results"

# Fixed at training time (--block 20) for every one of the 30 existing runs;
# also embedded in each checkpoint's own config, so this is only used to
# broadcast the returned [Bh, Bw] grid back to pixel resolution.
BLOCK_SIZE = 20

# variant short-name -> channel_vars, from cnn-transformer/README.md's Step 1
# table. 10 variants for this family, unlike yolo11's 10-with-different-names
# -- no separate variants.py needed.
VARIANT_CHANNEL_VARS = {
    "sqm": ["ssha_quality_masked"],
    "sch": ["stepped_convolution_highpass"],
    "gm": ["gradient_magnitude"],
    "gfh": ["gaussian_filter_highpass"],
    "rmh": ["rolling_mean_highpass"],
    "fft": ["fft_highpass"],
    "shg": ["ssha_quality_masked", "rolling_mean_highpass", "gradient_magnitude"],
    "s0g": ["ssha_quality_masked", "gradient_magnitude"],
    "sh0": ["ssha_quality_masked", "rolling_mean_highpass"],
    "0hg": ["rolling_mean_highpass", "gradient_magnitude"],
}


def parse_run_name(run_name):
    """"cnntrans_sqm_123" -> ("sqm", 123); "cnntrans_shg_456" -> ("shg", 456)."""
    if not run_name.startswith("cnntrans_"):
        raise ValueError(f"Run name {run_name!r} doesn't match cnntrans_<variant>_<seed>")

    rest = run_name[len("cnntrans_"):]
    variant, _, seed_str = rest.rpartition("_")
    if not variant or not seed_str.isdigit():
        raise ValueError(f"Run name {run_name!r} doesn't match cnntrans_<variant>_<seed>")

    return variant, int(seed_str)


def resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if str(device).isdigit():
        return torch.device(f"cuda:{device}")
    return torch.device(device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--run-name", required=True, help="e.g. cnntrans_sqm_123")
    parser.add_argument("--split", required=True, choices=["test", "valid"])

    parser.add_argument("--netcdf-root", type=Path, default=DEFAULT_NETCDF_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--weights", default=None, help="Overrides <runs-dir>/<run-name>/best.pt (no weights/ subdir for this family)")

    parser.add_argument("--threshold", type=float, default=0.5, help="Block-probability threshold for the binary mask variable.")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda | a bare GPU index like 0 or 1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files (debug).")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    variant, seed = parse_run_name(args.run_name)
    if variant not in VARIANT_CHANNEL_VARS:
        raise ValueError(f"Unknown variant {variant!r} in run name {args.run_name!r}")
    channel_vars = VARIANT_CHANNEL_VARS[variant]

    weights_path = Path(args.weights) if args.weights else args.runs_dir / args.run_name / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = resolve_device(args.device)

    nc_dir = args.netcdf_root / args.split
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {nc_dir}")
    if args.limit:
        nc_files = nc_files[: args.limit]

    out_netcdf_dir = args.results_dir / args.run_name / args.split / "netcdf"

    print(f"run={args.run_name} variant={variant} seed={seed} split={args.split} "
          f"device={device} channel_vars={channel_vars} threshold={args.threshold} files={len(nc_files)}")
    print(f"weights={weights_path}")

    n_ok, n_fail, n_skip = 0, 0, 0

    for i, nc_path in enumerate(nc_files, start=1):
        stem = nc_path.stem
        out_nc = out_netcdf_dir / f"{stem}.nc"

        if out_nc.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            pred_block_labels, probs = run_block_inference(
                str(weights_path), device, str(nc_path), channel_vars,
                block_size=BLOCK_SIZE, threshold=args.threshold,
            )
            probs = probs.cpu().numpy()
            pred_block_labels = pred_block_labels.cpu().numpy()

            with xr.open_dataset(nc_path) as ds:
                ds = ds.load()
            ds.attrs["source_l3_file"] = str(nc_path)

            full_shape = ds[channel_vars[0]].squeeze().shape
            crop_h, crop_w = probs.shape[0] * BLOCK_SIZE, probs.shape[1] * BLOCK_SIZE

            confidence_full = np.full(full_shape, np.nan, dtype=np.float32)
            confidence_full[:crop_h, :crop_w] = np.repeat(
                np.repeat(probs, BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1
            )

            mask_full = np.zeros(full_shape, dtype=np.uint8)
            mask_full[:crop_h, :crop_w] = np.repeat(
                np.repeat(pred_block_labels.astype(np.uint8), BLOCK_SIZE, axis=0), BLOCK_SIZE, axis=1
            )

            common.write_slim_netcdf(
                ds,
                mask_full,
                confidence_full,
                mask_var_name="internal_wave_bbox_mask",
                conf_mask_var_name="internal_wave_bbox_confidence",
                model_path=weights_path,
                conf_threshold=args.threshold,
                variant_name=variant,
                out_path=out_nc,
            )

            n_positive = int(pred_block_labels.sum())
            n_ok += 1
            print(f"[{i}/{len(nc_files)}] {stem}: {n_positive}/{pred_block_labels.size} positive block(s)")
        except Exception as exc:
            n_fail += 1
            print(f"[{i}/{len(nc_files)}] {stem}: FAILED: {exc}")
            traceback.print_exc()

    print(f"done. ok={n_ok} failed={n_fail} skipped={n_skip} total={len(nc_files)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
