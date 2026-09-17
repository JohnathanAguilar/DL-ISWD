"""Run a trained Faster R-CNN model over every NetCDF file in a SWOT L3
split, writing YOLO-format label .txt files and copies of the NetCDF with a
detection mask + confidence mask added.

Same output shape and variable-limiting behavior as
``yolo11/src/inference/infer.py``: only lat/lon, block_mask, iw_bbox_mask,
gaussian_filter_highpass, and this run's mask/confidence variables are kept
in the output NetCDF -- not a full duplicate of the source file (see
``src/inference/common.py::write_slim_netcdf``).

Example
-------
    python src/inference/infer.py --run-name frcnn_gh_s123 --split test --device 0
    python src/inference/infer.py --run-name frcnn_frgb_s456 --split valid --device 1

Outputs land in ``<results-dir>/<run-name>/<split>/{netcdf,text}/``. Run from
the ``faster-rcnn/`` directory, same as ``src/training/train.py``.
"""

import argparse
import sys
import traceback
from pathlib import Path

_INFERENCE_DIR = Path(__file__).resolve().parent
_SRC_DIR = _INFERENCE_DIR.parent
_MODEL_DIR = _SRC_DIR.parent

for _p in (_SRC_DIR, _MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from configs import config
except ImportError:
    import config

try:
    from inference import common
    from inference.gaussian_highpass_frcnn_detector import GaussianHighpassFasterRCNNDetector
    from inference.rgb_composite_frcnn_detector import RGBCompositeFasterRCNNDetector
except ImportError:
    import common
    from gaussian_highpass_frcnn_detector import GaussianHighpassFasterRCNNDetector
    from rgb_composite_frcnn_detector import RGBCompositeFasterRCNNDetector

DEFAULT_NETCDF_ROOT = Path("/workspace/data/bob/netcdf")
DEFAULT_RUNS_DIR = config.DEFAULT_RUNS_DIR
DEFAULT_RESULTS_DIR = _MODEL_DIR / "results"

# variant short-name -> (detector class, that class's own default conf).
# Only 2 variants for this family, unlike yolo11's 10 -- no separate
# variants.py needed.
VARIANT_DETECTORS = {
    "gh": GaussianHighpassFasterRCNNDetector,
    "frgb": RGBCompositeFasterRCNNDetector,
}
VARIANT_DEFAULT_CONF = {
    "gh": 0.10,
    "frgb": 0.65,
}


def parse_run_name(run_name):
    """"frcnn_gh_s123" -> ("gh", 123); "frcnn_frgb_s456" -> ("frgb", 456)."""
    if not run_name.startswith("frcnn_") or "_s" not in run_name:
        raise ValueError(f"Run name {run_name!r} doesn't match frcnn_<variant>_s<seed>")

    rest = run_name[len("frcnn_"):]
    variant, _, seed_str = rest.rpartition("_s")
    if not variant or not seed_str.isdigit():
        raise ValueError(f"Run name {run_name!r} doesn't match frcnn_<variant>_s<seed>")

    return variant, int(seed_str)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--run-name", required=True, help="e.g. frcnn_gh_s123")
    parser.add_argument("--split", required=True, choices=["test", "valid"])

    parser.add_argument("--netcdf-root", type=Path, default=DEFAULT_NETCDF_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--weights", default=None, help="Overrides <runs-dir>/<run-name>/weights/best.pt")

    parser.add_argument(
        "--conf", type=float, default=None,
        help="Defaults to this run's variant's own operating point "
             "(gh=0.10, frgb=0.65) when not given.",
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda | a bare GPU index like 0 or 1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files (debug).")

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    variant, seed = parse_run_name(args.run_name)
    if variant not in VARIANT_DETECTORS:
        raise ValueError(f"Unknown variant {variant!r} in run name {args.run_name!r}")

    detector_cls = VARIANT_DETECTORS[variant]
    conf = args.conf if args.conf is not None else VARIANT_DEFAULT_CONF[variant]

    weights_path = Path(args.weights) if args.weights else args.runs_dir / args.run_name / "weights" / "best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    nc_dir = args.netcdf_root / args.split
    nc_files = sorted(nc_dir.glob("*.nc"))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {nc_dir}")
    if args.limit:
        nc_files = nc_files[: args.limit]

    out_netcdf_dir = args.results_dir / args.run_name / args.split / "netcdf"
    out_text_dir = args.results_dir / args.run_name / args.split / "text"

    print(f"run={args.run_name} variant={variant} seed={seed} split={args.split} "
          f"device={args.device} conf={conf} files={len(nc_files)}")
    print(f"weights={weights_path}")

    detector = detector_cls(str(weights_path), device=args.device)

    n_ok, n_fail, n_skip = 0, 0, 0

    for i, nc_path in enumerate(nc_files, start=1):
        stem = nc_path.stem
        out_nc = out_netcdf_dir / f"{stem}.nc"
        out_txt = out_text_dir / f"{stem}.txt"

        if out_nc.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            ds_with_outputs, _detection_text, _boxed_image, detections = detector.detect_on_l3_file(
                nc_path,
                conf=conf,
                mask_var_name="internal_wave_bbox_mask",
                conf_mask_var_name="internal_wave_bbox_confidence",
            )

            original_shape = ds_with_outputs["internal_wave_bbox_mask"].squeeze().shape
            transform_info = detector_cls.transform_info_from_shape(original_shape)

            common.write_yolo_label_txt(detections, transform_info, out_txt)
            common.write_slim_netcdf(
                ds_with_outputs,
                mask_var_name="internal_wave_bbox_mask",
                conf_mask_var_name="internal_wave_bbox_confidence",
                model_path=weights_path,
                conf=conf,
                variant_name=variant,
                out_path=out_nc,
                family_tag="frcnn",
            )

            n_ok += 1
            print(f"[{i}/{len(nc_files)}] {stem}: {len(detections)} detection(s)")
        except Exception as exc:
            n_fail += 1
            print(f"[{i}/{len(nc_files)}] {stem}: FAILED: {exc}")
            traceback.print_exc()

    print(f"done. ok={n_ok} failed={n_fail} skipped={n_skip} total={len(nc_files)}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
