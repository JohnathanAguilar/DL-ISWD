#!/usr/bin/env bash
# Phase 1: train the 6 models = {gh, frgb} x {123, 456, 789}.
# Two waves of 3, one seed per GPU (indices 0,1,2). Wave B waits for wave A.
set -u
cd "$(dirname "$0")/.."   # -> faster-rcnn/
LOGDIR=pipeline/logs
mkdir -p "$LOGDIR"

GH_YAML=/workspace/data/bob/highpass_grouped_yolo_splits/gaussian_filter_highpass_magma/data.yaml
FRGB_YAML=/workspace/data/bob/composite_grouped_yolo_splits/scientific_rgb_composite/data.yaml
SEEDS=(123 456 789)
GPUS=(0 1 2)
EPOCHS=100

run_wave() {
  local variant="$1" yaml="$2"
  local pids=() names=()
  for i in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$i]}" gpu="${GPUS[$i]}"
    local name="frcnn_${variant}_s${seed}"
    echo "$(date +%T) launch $name  seed=$seed  gpu=$gpu"
    python src/training/train.py \
      --data "$yaml" \
      --name "$name" \
      --seed "$seed" \
      --epochs "$EPOCHS" \
      --device "$gpu" \
      > "$LOGDIR/train_${name}.log" 2>&1 &
    pids+=($!); names+=("$name")
  done
  local rc=0
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then
      echo "$(date +%T) OK   ${names[$j]}"
    else
      echo "$(date +%T) FAIL ${names[$j]} (see $LOGDIR/train_${names[$j]}.log)"; rc=1
    fi
  done
  return $rc
}

echo "===== WAVE A: gh ====="
run_wave gh "$GH_YAML"; a=$?
echo "===== WAVE B: frgb ====="
run_wave frgb "$FRGB_YAML"; b=$?

echo "===== training phase done (waveA rc=$a, waveB rc=$b) ====="
exit $(( a || b ))
