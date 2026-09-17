"""Run Ultralytics' own model.val() for one trained YOLO11 run on one split
(test or val), and write the resulting box-level metrics to a JSON file.

Example
-------
    python -m yolo_val.yolo_val_one --run-name yolo11n_frgb_s123 --split test --device 0
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO

_YOLO_VAL_DIR = Path(__file__).resolve().parent
YOLO11_DIR = _YOLO_VAL_DIR.parent
RUNS_DIR = YOLO11_DIR / "runs"
RESULTS_DIR = YOLO11_DIR / "yolo_val_results"
TMP_PROJECT_DIR = _YOLO_VAL_DIR / "tmp"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", required=True, choices=["test", "val"])
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--runs-dir", type=Path, default=RUNS_DIR,
        help="Defaults to yolo11/runs -- override to evaluate a sister sweep, "
             "e.g. yolo11/runs_customaug.",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Defaults to yolo11/yolo_val_results -- override to match --runs-dir "
             "for a sister sweep.",
    )
    parser.add_argument(
        "--tmp-dir", type=Path, default=TMP_PROJECT_DIR,
        help="Ultralytics val() scratch project dir. Defaults to yolo_val/tmp -- "
             "override if running two sweeps with the same run names concurrently.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    run_dir = args.runs_dir / args.run_name
    train_args = yaml.safe_load((run_dir / "args.yaml").read_text())
    data_yaml = train_args["data"]
    imgsz = train_args["imgsz"]

    weights_path = run_dir / "weights" / "best.pt"
    model = YOLO(str(weights_path))

    results = model.val(
        data=data_yaml,
        split=args.split,
        imgsz=imgsz,
        batch=16,
        device=args.device,
        plots=False,
        save_json=False,
        project=str(args.tmp_dir),
        name=f"{args.run_name}_{args.split}",
        exist_ok=True,
        verbose=False,
    )

    rd = results.results_dict
    precision = float(rd.get("metrics/precision(B)", float("nan")))
    recall = float(rd.get("metrics/recall(B)", float("nan")))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "run_name": args.run_name,
        "split": args.split,
        "data": data_yaml,
        "imgsz": imgsz,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": float(rd.get("metrics/mAP50(B)", float("nan"))),
        "map50_95": float(rd.get("metrics/mAP50-95(B)", float("nan"))),
        "fitness": float(rd.get("fitness", float("nan"))),
    }

    out_dir = args.results_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    print(f"{args.run_name}/{args.split}: precision={precision:.4f} recall={recall:.4f} "
          f"map50={metrics['map50']:.4f} map50_95={metrics['map50_95']:.4f} -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
