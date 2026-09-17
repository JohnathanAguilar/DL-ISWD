#!/usr/bin/env bash
# RT-DETR Part B1: retrain the 6 models with the swept LR.
# Usage: ./train_phase.sh <BEST_LR>
# README "NetCDF inference" Step-1 command verbatim except --lr <BEST_LR>.
# 2 waves of 3, one run per MIG slice, pinned by UUID (S1 = GPU1/dev1, not ordinal-addressable).
set -u
cd "$(dirname "$0")/.."   # -> rt-detr/
LOGDIR=pipeline/logs
mkdir -p "$LOGDIR"

BEST_LR="${1:?usage: train_phase.sh <BEST_LR>}"
PROJECT=/workspace/isw-detector/rt-detr/runs_lr5e5   # separate from the lr=1e-4 baselines in runs/
GHP_YAML=/workspace/data/bob/highpass_grouped_yolo_splits/gaussian_filter_highpass_magma/data.yaml
FRGB_YAML=/workspace/data/bob/composite_grouped_yolo_splits/scientific_rgb_composite/data.yaml
SEEDS=(123 456 789)
SLICES=(
  MIG-d786569a-875c-5d71-9acc-1a55e0cb78ce   # S1  GPU1/dev1
  MIG-b76979a7-08a3-5fba-9feb-881a571edc7b   # S2  GPU3/dev0
  MIG-7e00f275-b657-5e60-90de-f10fd799cdac   # S3  GPU4/dev0
)

run_wave() {
  local variant="$1" yaml="$2"
  local pids=() names=()
  for i in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$i]}" slice="${SLICES[$i]}"
    local name="rt_detr_${variant}_s${seed}"
    echo "$(date +%T) launch $name  seed=$seed  slice=${slice:0:20}..."
    CUDA_VISIBLE_DEVICES="$slice" python src/training/train.py \
      --data "$yaml" \
      --name "$name" \
      --project "$PROJECT" \
      --seed "$seed" --device cuda \
      --model-name PekingU/rtdetr_r50vd \
      --epochs 100 --batch 16 --workers 4 \
      --shortest-edge 1280 --longest-edge 1280 \
      --optimizer adamw --lr "$BEST_LR" --backbone-lr-mult 0.1 --weight-decay 1e-4 \
      --warmup-epochs 1.0 --lr-schedule cosine --lr-final-factor 0.01 --clip-grad-norm 0.1 \
      --fliplr 0.0 --flipud 0.0 --scale 0.0 --translate 0.0 \
      --val-conf 0.05 --eval-iou 0.5 --patience 0 \
      > "$LOGDIR/train_${name}.log" 2>&1 &
    pids+=($!); names+=("$name")
  done
  local rc=0
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "$(date +%T) OK   ${names[$j]}"
    else echo "$(date +%T) FAIL ${names[$j]} (see $LOGDIR/train_${names[$j]}.log)"; rc=1; fi
  done
  return $rc
}

echo "===== RT-DETR retrain, lr=$BEST_LR ====="
echo "===== WAVE A: ghp ====="
run_wave ghp "$GHP_YAML"; a=$?
echo "===== WAVE B: frgb ====="
run_wave frgb "$FRGB_YAML"; b=$?
echo "===== RT-DETR training done (waveA rc=$a, waveB rc=$b) ====="
exit $(( a || b ))
