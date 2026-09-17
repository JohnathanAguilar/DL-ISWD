#!/bin/bash
# Runs a fixed, pre-assigned list of (run_name, split) inference jobs on
# $DEVICE, sequentially. No shared queue/lock - each worker owns its own
# file (same design as yolo11/batch_run_inference/worker_inference.sh).
set -u

DEVICE="$1"

ROOT=/workspace/isw-detector/cnn-transformer
JOBS_FILE="$ROOT/batch_run_inference/device${DEVICE}.txt"
PROGRESS="$ROOT/batch_run_inference/progress_inference.log"

cd "$ROOT" || exit 1

if [ ! -f "$JOBS_FILE" ]; then
    echo "$(date -u +%FT%TZ) device$DEVICE: no jobs file, nothing to do" >> "$PROGRESS"
    exit 0
fi

while IFS=$'\t' read -r run_name split; do
    [ -z "$run_name" ] && continue
    echo "$(date -u +%FT%TZ) device$DEVICE: starting $run_name/$split" >> "$PROGRESS"

    python3 src/inference/infer.py \
        --run-name "$run_name" \
        --split "$split" \
        --device "$DEVICE" \
        --threshold 0.05 \
        > "$ROOT/batch_run_inference/logs/${run_name}_${split}.log" 2>&1
    status=$?

    echo "$(date -u +%FT%TZ) device$DEVICE: finished $run_name/$split (exit $status)" >> "$PROGRESS"
done < "$JOBS_FILE"

echo "$(date -u +%FT%TZ) device$DEVICE: all assigned jobs done, worker exiting" >> "$PROGRESS"
