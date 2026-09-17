# YOLO11 baseline

The YOLO11 box detector for internal solitary waves in SWOT L3 data. Training is
pure Ultralytics; the code in this directory is the layer around it — NetCDF
inference wrappers that render each SWOT field the way a given run expects, the
box-level `model.val()` path, the shared block-level evaluation glue, and the
static GPU job-dispatch scripts. Ten preprocessing variants are trained at three
seeds each (30 YOLO11n models), with a planned YOLO11**s** extension for `gauss`
and `frgb`.

Each variant has its own pre-built YOLO-format dataset (`train`/`val`/`test`
image folders + `data.yaml`, single class `IW`). The four `frgb` / `nograd` /
`nohp` / `nossha` variants are the same RGB composite (R = `ssha_quality_masked`,
G = `gaussian_filter_highpass`, B = `gradient_magnitude`) with one channel
zero-filled at a time; the rest are single-channel magma renderings of one
derived NetCDF field.

## How the code fits together

```
train.py (Ultralytics)  →  runs/<name>/weights/best.pt
                              │
  src/inference/variants.py (run-name → field rendering)  →  {highpass,rgb_composite}_yolo_detector.py  →  infer.py  →  common.write_output_netcdf
                              │
        eval_run_all.py  →  eval/evaluate_block_metrics.py  →  aggregate_eval_metrics.py
        yolo_val/yolo_val_one.py (Ultralytics model.val())  →  aggregate_yolo_val_metrics.py
```

| file | role |
|---|---|
| `train.py` | Thin Ultralytics wrapper (`--model-size n|s --name <run>`). `WEIGHTS_BY_SIZE = {n: yolo11n.pt, s: yolo11s.pt}`, `resolve_device` (`auto` → mps > cuda > cpu). `main` just calls `YOLO(weights).train(imgsz=1280, lr0=0.002, patience=100, ...)` with **every augmentation knob (`hsv_*`, `mosaic`, `mixup`, `copy_paste`, `multi_scale`, `fliplr`, `flipud`, `translate`, `scale`, `degrees`) forced to 0** — the training set is pre-augmented offline, matching the faster-rcnn / rt-detr recipes. Outputs are Ultralytics' native `runs/<name>/` (incl. `args.yaml`). |
| `src/inference/variants.py` | Run-name → NetCDF rendering. `VARIANTS` maps the 10 dataset abbreviations to `(family, kwargs)`: the `composite` family (`RGBCompositeYOLODetector`) covers `frgb` (no zeroed channel) / `nograd` / `nohp` / `nossha`; the `highpass` family (`HighpassYOLODetector`) covers `gauss`, `fft`, `gradmag`, `ssha`, `rollmean`, `stepconv`, each with a `source_var`. `parse_run_name("yolo11n_frgb_s123") → ("frgb", 123)`; `build_detector(...)` lazily constructs the right class. |
| `src/inference/common.py` | The family-agnostic geometry + rendering + output layer, ported verbatim from the faster-rcnn detectors so it isn't re-copied per detector: `load_l3_dataset`, `read_2d_var`, `robust_normalize`, `array_to_magma_rgb`, `get_transform_info`, `orient_rgb_for_model`, `run_yolo_detection` (wraps `model.predict`), `detections_to_original_mask` / `..._confidence_mask`, `write_yolo_label_txt`, and `write_output_netcdf` (same slim-file contract as the other families — `block_mask`, `iw_bbox_mask`, `gaussian_filter_highpass` + this run's mask/confidence, `yolo11_*` attrs). |
| `src/inference/highpass_yolo_detector.py` | `HighpassYOLODetector(weights_path, device, source_var)` — one class for all six single-channel magma variants: `read_2d_var` → `array_to_magma_rgb` → `orient_rgb_for_model` → `run_yolo_detection` → mask projection. Returns `(ds, detections, transform_info, mask, confidence_mask)`. |
| `src/inference/rgb_composite_yolo_detector.py` | `RGBCompositeYOLODetector(weights_path, device, zero_channel=None)` — reads R/G/B straight from the source NetCDF, robust-normalizes each, and zero-fills the one named by `zero_channel`. Serves `frgb` + the three channel ablations. |
| `src/inference/infer.py` | Batch NetCDF inference CLI (`python -m src.inference.infer --run-name yolo11n_frgb_s123 --split test|valid --conf 0.05 --imgsz 1280`). `variants.parse_run_name` → `variants.build_detector`; per file → `common.write_yolo_label_txt` + `common.write_output_netcdf` → `results/<run>/<split>/{netcdf,text}/`. |
| `yolo_val/yolo_val_one.py` | The parallel box-level path: reads `runs/<run>/args.yaml` for `data`/`imgsz`, loads `weights/best.pt`, calls Ultralytics `model.val(split=...)`, extracts `precision/recall/mAP50/mAP50-95/fitness` (+ derived F1) → `yolo_val_results/<run>/<split>_metrics.json`. No NetCDF involved. |
| `eval_run_all.py` | For each dir under `results/` with both `test/netcdf` and `valid/netcdf`, shells out `eval/evaluate_block_metrics.py --majority-threshold 0.25` → `eval_results/<run>/`. `--results-dir` / `--eval-results-dir` overridable for sister sweeps. |
| `aggregate_eval_metrics.py` | Rolls up the block-eval runs. Uses `variants.parse_run_name` for `variant` and separately derives `model_size` from the run-name prefix (so `yolo11n_gauss_*` and `yolo11s_gauss_*` aren't pooled). Groups by `(model_size, variant)` → `eval_results/summary_by_variant.csv`. |
| `aggregate_yolo_val_metrics.py` | Rolls up the `yolo_val_results/*/{test,val}_metrics.json`, groups by `(variant, split)` → `yolo_val_results/summary_by_variant.csv`. |
| `run_ood_block_eval.py` | Reuse-only OOD orchestrator over `/workspace/OOD/` (`amazon_mouth`, `mascarene_ridge`, `south_china_sea`): symlink-stage each dataset as a `test` split, run `infer.py`, symlink the run's **in-distribution** `valid` outputs (so the confidence threshold stays in-distribution), call `eval_run_all.py` + `aggregate_eval_metrics.py`, then concatenate the per-dataset summaries into `eval_results_ood/summary_all_datasets.csv`. Presets: `{gauss, frgb} × {123, 456, 789}`. `--majority-threshold` is a guard only — anything ≠ 0.25 errors out. |
| `batch_run*/gen_*.py` | Static GPU job-partition dispatch: build a job list, `assignments[i % NUM_DEVICES].append(job)`, write `deviceN.txt` for per-GPU worker shell scripts. This **replaced a `flock`-based shared queue** that gave no real mutual exclusion on the cluster's network filesystem (all workers grabbed the same item). `batch_run/` = the remaining nano training jobs (+ `gen_queue.py`, kept for history, and `warm_cache.py` to pre-build the Ultralytics label cache); `batch_run_yolo11s/` = the 6 small-model jobs; `batch_run_customaug/` = all 30 retrained *with* geometric augmentation into `runs_customaug/`; the `batch_run_inference*/` variants partition `(run, split)` inference jobs. |

## Two evaluation paths

- **Box-level** — Ultralytics' own `model.val()` (`yolo_val/yolo_val_one.py`),
  computed directly against each model's image-chip splits. Aggregated by
  `aggregate_yolo_val_metrics.py`.
- **Block-level** — the shared `eval/evaluate_block_metrics.py`, run over the
  full-swath NetCDF inference outputs so YOLO11 compares directly against the
  other three families. Aggregated by `aggregate_eval_metrics.py`.

## Running it

Needs `ultralytics`, `xarray`, `netCDF4`, `scikit-learn`, `pyyaml`, `pandas`.
`yolo11n.pt` is at `yolo11/yolo11n.pt`; `yolo11s.pt` is auto-downloaded on first
`--model-size s` run. Run from `yolo11/`. One `(variant, seed)`, using `frgb`:

```bash
# train (writes runs/yolo11n_frgb_s123/)
python3 train.py --model-size n --epochs 100 --name yolo11n_frgb_s123 \
    --data <frgb data.yaml> --seed 123 --batch 16 --workers 8 --device 0

# box-level eval
python3 -m yolo_val.yolo_val_one --run-name yolo11n_frgb_s123 --split test --device 0

# NetCDF inference (writes results/yolo11n_frgb_s123/<split>/{text,netcdf}/)
python3 -m src.inference.infer --run-name yolo11n_frgb_s123 --split test  --device 0 --conf 0.05
python3 -m src.inference.infer --run-name yolo11n_frgb_s123 --split valid --device 0 --conf 0.05

# block-level eval over everything under results/, then aggregate
python3 eval_run_all.py
python3 aggregate_eval_metrics.py
python3 aggregate_yolo_val_metrics.py
```

For the full 30-run sweep use the `batch_run*/gen_*.py` partitioners + their
worker scripts; for OOD use `python3 run_ood_block_eval.py`.

## Outputs

| directory | contents |
|---|---|
| `runs/<run>/` | `weights/{best,last}.pt`, `args.yaml`, per-epoch `results.csv` / `results.png` |
| `yolo_val_results/<run>/{test,val}_metrics.json` + `summary_by_variant.csv` | Ultralytics box-level P/R/mAP, per run and aggregated by variant |
| `results/<run>/<split>/{text,netcdf}/` | YOLO-format `.txt` labels + detection/confidence-mask NetCDFs |
| `eval_results/<run>/block_metrics_{overall,per_file}.csv` + `summary_by_variant.csv` | block-level metrics, per run and aggregated by variant |
