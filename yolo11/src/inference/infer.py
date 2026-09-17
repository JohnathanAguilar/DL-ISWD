"""Run a trained YOLO11 model over every NetCDF file in a SWOT L3 split,
writing YOLO-format label .txt files and copies of the NetCDF with a
detection mask + confidence mask added.

Example
-------
    python -m src.inference.infer --run-name yolo11n_frgb_s123 --split test --device 0
    python -m src.inference.infer --run-name yolo11n_gauss_s456 --split valid --device 1

Outputs land in ``<results-dir>/<run-name>/<split>/{netcdf,text}/``.
"""

import argparse
import sys
import traceback
from pathlib import Path

import torch

_INFERENCE_DIR = Path(__file__).resolve().parent
_YOLO11_DIR = _INFERENCE_DIR.parent.parent

if str(_YOLO11_DIR) not in sys.path:
    sys.path.insert(0, str(_YOLO11_DIR))

from src.inference import common, variants  # noqa: E402

DEFAULT_NETCDF_ROOT = Path("/workspace/data/bob/netcdf")
DEFAULT_RUNS_DIR = _YOLO11_DIR / "runs"
DEFAULT_RESULTS_DIR = _YOLO11_DIR / "results"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--run-name", required=True, help="e.g. yolo11n_frgb_s123")
    parser.add_argument("--split", required=True, choices=["test", "valid"])

    parser.add_argument("--netcdf-root", type=Path, default=DEFAULT_NETCDF_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--weights", default=None, help="Overrides <runs-dir>/<run-name>/weights/best.pt")

    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N files (debug).")

    return parser.parse_args(argv)


def resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return "0"
        return "cpu"
    return device


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)

    variant, seed = variants.parse_run_name(args.run_name)

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
          f"device={device} conf={args.conf} files={len(nc_files)}")
    print(f"weights={weights_path}")

    detector = variants.build_detector(variant, weights_path, device=device)

    n_ok, n_fail, n_skip = 0, 0, 0

    for i, nc_path in enumerate(nc_files, start=1):
        stem = nc_path.stem
        out_nc = out_netcdf_dir / f"{stem}.nc"
        out_txt = out_text_dir / f"{stem}.txt"

        if out_nc.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            ds, detections, transform_info, mask, confidence_mask = detector.detect_on_l3_file(
                nc_path, conf=args.conf, imgsz=args.imgsz
            )

            common.write_yolo_label_txt(detections, transform_info, out_txt)
            common.write_output_netcdf(
                ds,
                mask,
                confidence_mask,
                model_path=weights_path,
                conf=args.conf,
                variant_name=variant,
                mask_var_name="internal_wave_bbox_mask",
                conf_mask_var_name="internal_wave_bbox_confidence",
                out_path=out_nc,
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
