#!/bin/bash
# Runs a fixed, pre-assigned list of (run_name, split) inference jobs on
# $DEVICE, sequentially. No shared queue/lock - each worker owns its own
# file (same design as ../batch_run/worker_static.sh).
set -u

DEVICE="$1"

ROOT=/workspace/isw-detector/yolo11
JOBS_FILE="$ROOT/batch_run_inference/device${DEVICE}.txt"
PROGRESS="$ROOT/batch_run_inference/progress_inference.log"

export CUDA_VISIBLE_DEVICES="MIG-4c9f92ae-b9bf-56a3-b84f-a9db4a7637ee,MIG-5855c9a5-601e-54e5-98e0-8437cdfe54af,MIG-4ba33449-c056-5af9-91ea-c80b77eabf6c,MIG-b5beefbf-b350-5cbb-b5c4-22c4f2ba08e3,MIG-ce99f083-a10e-5d2f-aab0-9ae215817f11,MIG-ed357fc9-5f50-50f2-9711-ad83657a1804,MIG-d0efc60c-b921-53bb-b784-3d92a1c7ceab"

cd "$ROOT" || exit 1

if [ ! -f "$JOBS_FILE" ]; then
    echo "$(date -u +%FT%TZ) device$DEVICE: no jobs file, nothing to do" >> "$PROGRESS"
    exit 0
fi

while IFS=$'\t' read -r run_name split; do
    [ -z "$run_name" ] && continue
    echo "$(date -u +%FT%TZ) device$DEVICE: starting $run_name/$split" >> "$PROGRESS"

    python3 -m src.inference.infer \
        --run-name "$run_name" \
        --split "$split" \
        --device "$DEVICE" \
        --conf 0.05 \
        > "$ROOT/batch_run_inference/logs/${run_name}_${split}.log" 2>&1
    status=$?

    echo "$(date -u +%FT%TZ) device$DEVICE: finished $run_name/$split (exit $status)" >> "$PROGRESS"
done < "$JOBS_FILE"

echo "$(date -u +%FT%TZ) device$DEVICE: all assigned jobs done, worker exiting" >> "$PROGRESS"
