# Faster R-CNN baseline

A torchvision Faster R-CNN (ResNet-50-FPN) trained on the same SWOT chips, at the
same resolution, with the same augmentation and the same evaluation protocol as
the YOLO11 and RT-DETR baselines, so the three box detectors compare directly.
Two representation variants — `gh` (Gaussian high-pass) and `frgb` (scientific RGB
composite) — are trained at three seeds each, run through full-swath NetCDF
inference, and scored with the shared `eval/evaluate_block_metrics.py`.

Dataset: `scientific_rgb_composite` YOLO-format splits (3447 train / 225 val /
254 test chips, 69×709 px, one class `IW`, ~50% of training chips are negatives).

## How the code fits together

```
configs/config.py  →  src/dataio/dataset.py  →  src/models/model.py  →  src/training/{engine,metrics}.py
      │                                                                          │
      └────────────────→  src/training/train.py  /  src/evaluation/evaluate.py  ─┘
                                     │
      src/inference/{gaussian_highpass,rgb_composite}_frcnn_detector.py  →  infer.py  →  common.write_slim_netcdf
                                     │
                     eval/evaluate_block_metrics.py  →  aggregate_block_metrics.py
```

| file | role |
|---|---|
| `configs/config.py` | Module constants: `DEFAULT_DATA_YAML`, `MIN_SIZE = MAX_SIZE = 1280` (long-side resize matching YOLO `imgsz=1280`), the custom anchor ladder `ANCHOR_SIZES ((64),(128),(256),(384),(512))` / `ANCHOR_ASPECT_RATIOS (1,2,4,8)` measured from label statistics, and the eval protocol (`EVAL_IOU 0.50`, `NMS_IOU 0.7`, `CACHE_CONF 0.001`, conf sweep 0.05→0.95). Sets `PYTORCH_ENABLE_MPS_FALLBACK` at import. |
| `src/dataio/dataset.py` | YOLO-format → torchvision-detection adapter. `load_data_config` / `labels_dir_for`; `YoloDetectionDataset` reads CHW float images and `cls cx cy w h` → xyxy pixels, **adding 1 to every class id** (torchvision reserves 0 for background). Augmentation helpers `_hflip` / `_vflip` / `_scale_translate` mirror YOLO's `scale=0.25` / `translate=0.05`. |
| `src/models/model.py` | Faster R-CNN builder. `build_model(...)` constructs the stock torchvision model, then **after construction** swaps in a new `AnchorGenerator` (tall anchors), a fresh `RPNHead` sized to the new anchor count, and a fresh `FastRCNNPredictor`. `_anchor_sizes` rescales the ladder if `--imgsz` changes; `resolve_device` prefers CUDA → MPS → CPU and maps a bare `"0"` to `cuda:0`. `BACKBONES`: `resnet50_fpn` (default), `resnet50_fpn_v2`, `mobilenet_v3_large_fpn`. |
| `src/training/engine.py` | Shared train/predict loops **and the MPS guard** (see below): `gradient_backend_is_sane` (pre-flight grad-norm check), `train_one_epoch` (raises on non-finite loss, skips steps with non-finite grad norm, aborts if >25% of steps skipped), `predict_dataset`. |
| `src/training/metrics.py` | Pure functions — greedy class-aware TP/FP/FN matching at one IoU, precision / recall / F-beta, and 101-point interpolated COCO AP (`mAP50`, `mAP50-95`). **Byte-identical to `rt-detr/src/training/metrics.py`.** |
| `src/training/train.py` | CLI entry (`--name frcnn_frgb`). Seeds, resolves device, builds datasets/loaders, `build_model(num_classes = len(names)+1)`, runs the gradient sanity check (silent CPU fallback if `--device auto` landed on MPS and it fails), then per epoch `train_one_epoch` → `predict_dataset` → `ap_metrics` + `fixed_threshold_metrics`. Writes `weights/{last,best}.pt` (best by `val_mAP50`), `results.csv`, `results.png`, `args.json`. |
| `src/evaluation/evaluate.py` | The 4-step protocol: cache predictions at conf 0.001 for val and test → sweep conf 0.05→0.95 on val for max F-beta → score test at that operating point → also report threshold-free AP. `load_checkpoint` rebuilds the model from the checkpoint's `args`. Writes `eval/faster_rcnn_best_conf_fixed_<beta>_summary.json` + `..._operating_points_summary.csv`. |
| `src/inference/predict.py` | Visualization only: draws N chips with GT boxes (green) vs. predictions ≥ `--conf` (red) → `predictions_<split>.png`. Not part of the metrics path. |
| `src/inference/gaussian_highpass_frcnn_detector.py` | `GaussianHighpassFasterRCNNDetector` (the `gh` variant). Self-contained: L3 load, quality masking, `nan_gaussian_filter` + `gaussian_highpass` (σ=8), magma rendering, orientation helpers, `run_detection` (undoes the class-id +1), and `detections_to_original_mask` / `..._confidence_mask` (rasterize xyxy back through `fliplr` + transpose to the source grid, overlaps take max confidence). `detect_on_l3_file(...)`. |
| `src/inference/rgb_composite_frcnn_detector.py` | `RGBCompositeFasterRCNNDetector` (the `frgb` variant). Same structure; the image is the 3-channel composite (R = robust-normalized SSHA, G = Gaussian high-pass, B = gradient magnitude). Geometry / detection / mask methods copied verbatim from the gaussian detector. |
| `src/inference/infer.py` | Batch NetCDF inference CLI (`--run-name frcnn_gh_s123 --split test|valid --device`). `VARIANT_DETECTORS = {gh, frgb}`, `VARIANT_DEFAULT_CONF = {gh: 0.10, frgb: 0.65}`. Per `.nc`: `detect_on_l3_file` → `common.write_yolo_label_txt` + `common.write_slim_netcdf(family_tag="frcnn")` → `results/<run>/<split>/{netcdf,text}/`. |
| `src/inference/common.py` | Slim-output writer shared by `infer.py`: `write_yolo_label_txt` (normalized `cls cx cy w h conf`) and `write_slim_netcdf` (lat/lon + `block_mask`, `iw_bbox_mask`, `gaussian_filter_highpass` + this run's mask/confidence, zlib-compressed). |
| `aggregate_block_metrics.py` | Reads `eval_results/*/block_metrics_overall.csv`, parses `frcnn_<variant>_s<seed>`, groups by variant → `eval_results/summary_by_variant.csv` (`<col>_mean` / `<col>_std`, `ddof=1`). |
| `pipeline/{train,infer,eval}_phase.sh` | Wave-based launchers for the full 6-model sweep across 3 GPUs. |

## Custom anchors

Internal-wave boxes span nearly the full swath width (median normalized width
1.0) and vary only in along-track height, giving h/w aspect ratios of 1.5–10.2.
torchvision's default ratios (0.5, 1.0, 2.0) cannot cover that, so `build_model`
replaces the anchor generator with ratios `(1, 2, 4, 8)` over sizes
`(64, 128, 256, 384, 512)` px at 1280, rescaled automatically if `--imgsz`
changes. `--default-anchors` restores the stock anchors.

## Apple MPS is broken for this model — train on CPU

PyTorch's MPS backend returns corrupt gradients for the Faster R-CNN ROI head:
the RPN losses look fine, but `loss_classifier` / `loss_box_reg` backward produces
`nan` or ~1e8-scale gradients, turning the weights into `nan` within two optimizer
steps. Forward passes and inference on MPS are fine. `engine.py:gradient_backend_is_sane`
runs a pre-flight check before the first epoch; `train.py` falls back to CPU
automatically when `--device auto` lands on MPS, and `train_one_epoch` skips any
step with non-finite gradients (aborting if more than 25% are skipped). Evaluation
and prediction still default to MPS and are safe there. On a CUDA box none of this
matters.

## Matched to the other baselines, and not

**Matched:** the same `images/{train,val,test}` splits via `data.yaml`; effective
resolution (long side at 1280 px); augmentation (H/V flip 0.5, scale jitter 0.25,
translate 0.05 — no HSV/mosaic/mixup/copy-paste/rotation); and the evaluation
protocol (predictions cached at conf 0.001, val swept 0.05→0.95 for max
fixed-threshold F-beta at IoU 0.50, test reported at that threshold, greedy
class-aware TP/FP/FN matching). The fixed-threshold P/R/F-beta code reproduces the
published YOLO numbers exactly.

**Deliberately different:** the custom tall anchors (above); backbone scale
(ResNet-50-FPN ~41M params vs. YOLO11n's 2.6M — the usual published baseline
pairing, not parameter-matched); and `metrics.py` computes 101-point COCO AP
rather than calling Ultralytics' validator, so `mAP50` / `mAP50-95` are close to
but not bit-identical with the YOLO numbers.

## Running it

Needs `torch`, `torchvision`, `numpy`, `pyyaml`, `pillow`, `matplotlib`, `tqdm`.
Run from `faster-rcnn/`. One `(variant, seed)`, using `frgb`:

```bash
# train (writes runs/frcnn_frgb_s123/)
python src/training/train.py --data <frgb data.yaml> --name frcnn_frgb_s123 --seed 123 --device 0

# pick an operating point on val, report test (writes runs/.../eval/)
python src/evaluation/evaluate.py --weights runs/frcnn_frgb_s123/weights/best.pt --beta 1.0

# full-swath NetCDF inference (writes results/frcnn_frgb_s123/<split>/{netcdf,text}/)
python src/inference/infer.py --run-name frcnn_frgb_s123 --split test  --device 0
python src/inference/infer.py --run-name frcnn_frgb_s123 --split valid --device 0

# block-level eval + aggregate the 3 seeds per variant
python eval/evaluate_block_metrics.py \
    --test-dir faster-rcnn/results/frcnn_frgb_s123/test \
    --val-dir  faster-rcnn/results/frcnn_frgb_s123/valid \
    --majority-threshold 0.25 \
    --overall-csv faster-rcnn/eval_results/frcnn_frgb_s123/block_metrics_overall.csv \
    --per-file-csv faster-rcnn/eval_results/frcnn_frgb_s123/block_metrics_per_file.csv
python aggregate_block_metrics.py
```

The full 6-model sweep is driven by `pipeline/train_phase.sh` →
`pipeline/infer_phase.sh` → `pipeline/eval_phase.sh`.

## Outputs

| directory | contents |
|---|---|
| `runs/frcnn_<variant>_s<seed>/` | `weights/{best,last}.pt`, `args.json`, `results.csv`, `results.png`, `eval/*_summary.{json,csv}` |
| `results/frcnn_<variant>_s<seed>/<split>/{netcdf,text}/` | slim NetCDFs + YOLO-format `.txt` labels with `internal_wave_bbox_mask` / `internal_wave_bbox_confidence` |
| `eval_results/frcnn_<variant>_s<seed>/block_metrics_{overall,per_file}.csv` + `summary_by_variant.csv` | block-level metrics, per run and aggregated by variant |

The `eval/*_summary.csv` files use the `eval_type,split,confidence,tp,fp,fn,precision,recall,f1`
columns of the YOLO ablation tables, so a Faster R-CNN row concatenates onto them
directly (`--tag` controls the `eval_type` value).
