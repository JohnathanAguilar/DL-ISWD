# RT-DETR baseline

An RT-DETR (Real-Time Detection Transformer, [arXiv:2304.08069](https://arxiv.org/abs/2304.08069))
detector trained on the same SWOT chips, at the same effective resolution, with
the same augmentation and the same evaluation protocol as the YOLO11 and Faster
R-CNN baselines. It is built on HuggingFace `transformers` rather than Ultralytics'
`RTDETR` (which force-stretches non-square chips). Two representation variants —
`ghp` (Gaussian high-pass) and `frgb` (scientific RGB composite) — are trained at
three seeds each and scored with the shared `eval/evaluate_block_metrics.py`.

Dataset: `scientific_rgb_composite` YOLO-format splits (3447 train / 225 val /
254 test chips, 69×709 px, one class `IW`).

## How the code fits together

Same directory shape and data flow as `faster-rcnn/`. The differences are the
model wrapper (`transformers`), the image processor owning resize/normalize/pad,
backbone-LR param groups, and no NMS / no background class.

| file | role |
|---|---|
| `configs/config.py` | Mirrors faster-rcnn's, with `SHORTEST_EDGE = LONGEST_EDGE = 1280` instead of `MIN/MAX_SIZE` (`RTDetrImageProcessor`'s `{shortest_edge, longest_edge}` resize reproduces torchvision's for these chips — `709×69 → 1280×125`), `DEFAULT_MODEL_NAME = "PekingU/rtdetr_r50vd"`, `MAX_DETECTIONS = 300`, and **no `NMS_IOU`** (query-based one-to-one matching). |
| `src/dataio/dataset.py` | Same `load_data_config` / `labels_dir_for` and the same `_hflip` / `_vflip` / `_scale_translate` as faster-rcnn (minus the class-labels vector — single class). `YoloDetectionDataset.__getitem__` returns `(uint8 HWC image, COCO annotation dict)`; adds `ground_truth_boxes(idx)`. `RTDetrCollator(processor)` is a picklable class (not a closure — macOS spawn workers) running `processor(images, annotations, return_tensors="pt")` as the DataLoader `collate_fn`. |
| `src/models/model.py` | HuggingFace wrapper + **MPS float64 monkeypatch** (see below). `build_model(model_name, num_labels=1, pretrained=True)` → `RTDetrForObjectDetection.from_pretrained(..., num_labels=num_labels, ignore_mismatched_sizes=True)`. `num_labels` is **not** `+1` — no background class. `resolve_device` does *not* map a bare `"0"` (`torch.device("0")` raises); `infer.py` compensates. |
| `src/training/engine.py` | `build_param_groups(model, lr, backbone_lr_mult, weight_decay)` splits `named_parameters()` on `"backbone"` into two groups — backbone at `lr * backbone_lr_mult`, everything else at `lr` — implementing RT-DETR's recipe of training the pretrained CNN backbone slower than the randomly-initialized encoder/decoder. `gradient_backend_is_sane` here uses a much looser bound (`max_norm=1e7`) because RT-DETR's healthy grad norms are legitimately 5e2–1e5. `train_one_epoch` has the same non-finite-loss-raises / non-finite-grad-skips structure; forward is `model(pixel_values=, pixel_mask=, labels=)`. `predict_dataset` post-processes with `processor.post_process_object_detection(..., use_focal_loss=True)` and indexes results by `image_id` (safe under shuffling). |
| `src/training/metrics.py` | **Byte-identical to `faster-rcnn/src/training/metrics.py`** — greedy class-aware TP/FP/FN matching, 101-point COCO AP. |
| `src/training/train.py` | Same layout/outputs as faster-rcnn's. Builds an `RTDetrImageProcessor(size={shortest_edge, longest_edge}, do_pad=True)` and the collator; `build_optimizer` uses `build_param_groups` (AdamW, `lr=1e-4`, `--backbone-lr-mult 0.1`, `--clip-grad-norm 0.1`); `class_ids=(0,)`. Same MPS→CPU auto-fallback on a failed sanity check. |
| `src/evaluation/evaluate.py` | Same 4-step protocol and same JSON/CSV outputs (`tag="rt_detr"`). `load_checkpoint` rebuilds via `build_model(..., pretrained=False)` and reconstructs the `RTDetrImageProcessor` from the checkpoint. No `--nms-iou` / `--max-det`. |
| `src/inference/predict.py` | Same as faster-rcnn's but routes images through the processor and `post_process_object_detection`, and pulls GT from `dataset.ground_truth_boxes(idx)`. |
| `src/inference/{gaussian_highpass,rgb_composite}_rtdetr_detector.py` | `GaussianHighpassRTDetrDetector` (`ghp`, conf 0.10) and `RGBCompositeRTDetrDetector` (`frgb`, conf 0.05). SSHA prep, high-pass / composite math, magma rendering, orientation, mask projection are copied verbatim from the faster-rcnn detectors. The one substantive change is `run_detection`: it **pads both spatial dims up to the next multiple of 32** before `processor.pad` — the hybrid encoder's FPN upsamples the stride-32 map ×2 and concats with the stride-16 map, which only aligns when every dim is divisible by 32 (otherwise `torch.concat` throws on the ~60–70 px-wide chips). Output attrs are `rtdetr_*`. |
| `src/inference/infer.py` | Same as faster-rcnn's but `VARIANT_DETECTORS = {ghp, frgb}`, `VARIANT_DEFAULT_CONF = {ghp: 0.10, frgb: 0.05}`, `parse_run_name` expects `rt_detr_<variant>_s<seed>`, and it adds `normalize_device()` (bare digit → `cuda:N`). `write_slim_netcdf(family_tag="rtdetr")`. |
| `src/inference/common.py` | Identical to faster-rcnn's except `family_tag="rtdetr"` — each family owns a self-contained `src/`. |
| `aggregate_block_metrics.py` | Same as faster-rcnn's but `RUN_NAME_RE` matches `rt_detr_<variant>_s<seed>`; optional `argv[1]` overrides the eval-results dir. |
| `pipeline/{train,infer,eval}_phase.sh` | Wave-based launchers for the full 6-model sweep. |

## MPS: one narrow bug, patched; otherwise the fast path

Unlike Faster R-CNN (whose MPS backward pass is silently corrupt and unfixable
short of CPU), RT-DETR hit only one concrete issue:
`transformers`' `build_2d_sinusoidal_position_embedding` hardcodes
`dtype=torch.float64` for its internal frequency arithmetic, and MPS has no
float64 support, so `.to("mps")` crashes immediately. That embedding is a fixed,
non-learned, `lru_cache`d buffer, so `model.py` monkeypatches the function at
import time to compute it on CPU and move the *result* to the target device —
exact, no precision change. With that patch MPS gradients match CPU to normal
float32 noise and stay stable, so **MPS is the default and recommended device for
this baseline**. `engine.py` still runs the same pre-flight gradient sanity check
(and CPU fallback) as a guard against a future `transformers`/`torch` version
reintroducing a gap.

## Matched to the other baselines, and not

**Matched:** the same splits via `data.yaml`; effective resolution (long side at
1280 px, aspect ratio preserved); augmentation (H/V flip 0.5, scale jitter 0.25,
translate 0.05 — identical code to `faster-rcnn/src/dataio/dataset.py`); and the
evaluation protocol (`metrics.py` copied verbatim from faster-rcnn, predictions
cached at conf 0.001, val swept 0.05→0.95 for max fixed-threshold F-beta at IoU
0.50, test reported at that operating point).

**Deliberately different:** no NMS (RT-DETR decodes a fixed set of learned queries
with a one-to-one Hungarian-matching loss — duplicate suppression is learned);
default backbone `rtdetr_r50vd` (~42.7M params) chosen to match Faster R-CNN's
backbone family; and RT-DETR's official optimizer recipe (AdamW, backbone LR at
0.1× the head LR, weight decay 1e-4, gradient clip at norm 0.1) rather than
Faster R-CNN's SGD recipe.

## Running it

Needs the `faster-rcnn` deps plus `transformers`. Run from `rt-detr/`. One
`(variant, seed)`, using `frgb`:

```bash
# train (writes runs/rt_detr_frgb_s123/) — MPS is the fast path here
python src/training/train.py --data <frgb data.yaml> --name rt_detr_frgb_s123 --seed 123 --device mps

# pick an operating point on val, report test
python src/evaluation/evaluate.py --weights runs/rt_detr_frgb_s123/weights/best.pt --beta 1.0

# full-swath NetCDF inference (writes results/rt_detr_frgb_s123/<split>/{netcdf,text}/)
python src/inference/infer.py --run-name rt_detr_frgb_s123 --split test  --device 0
python src/inference/infer.py --run-name rt_detr_frgb_s123 --split valid --device 0

# block-level eval + aggregate the 3 seeds per variant
python eval/evaluate_block_metrics.py \
    --test-dir rt-detr/results/rt_detr_frgb_s123/test \
    --val-dir  rt-detr/results/rt_detr_frgb_s123/valid \
    --majority-threshold 0.25 \
    --overall-csv rt-detr/eval_results/rt_detr_frgb_s123/block_metrics_overall.csv \
    --per-file-csv rt-detr/eval_results/rt_detr_frgb_s123/block_metrics_per_file.csv
python aggregate_block_metrics.py
```

The full 6-model sweep is driven by `pipeline/train_phase.sh` →
`pipeline/infer_phase.sh` → `pipeline/eval_phase.sh`.

## Outputs

| directory | contents |
|---|---|
| `runs/rt_detr_<variant>_s<seed>/` | `weights/{best,last}.pt`, `args.json`, `results.csv`, `results.png`, `eval/*_summary.{json,csv}` |
| `results/rt_detr_<variant>_s<seed>/<split>/{netcdf,text}/` | slim NetCDFs + YOLO-format `.txt` labels with `internal_wave_bbox_mask` / `internal_wave_bbox_confidence` |
| `eval_results/rt_detr_<variant>_s<seed>/block_metrics_{overall,per_file}.csv` + `summary_by_variant.csv` | block-level metrics, per run and aggregated by variant |

`evaluate.py --tag` controls the `eval_type` value so rows from all three
baselines concatenate into one comparison table.
