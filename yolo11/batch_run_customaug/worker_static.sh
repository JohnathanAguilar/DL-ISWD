#!/bin/bash
# Runs a fixed, pre-assigned list of training jobs on $DEVICE, sequentially,
# for the augmented sweep (writes into runs_customaug/, a sister of runs/).
# No shared queue/lock - each worker owns its own file (same design as
# ../batch_run/worker_static.sh).
#
# CUDA_VISIBLE_DEVICES is set to all 7 MIG slice UUIDs explicitly -- without
# this, CUDA only enumerates 5 devices (one per physical GPU), silently
# hiding the second slice on the 2 GPUs that have 2 each. With all 7 UUIDs
# listed, $DEVICE (0-6) indexes into this list the same way every other
# batch job this session has.
set -u

DEVICE="$1"

ROOT=/workspace/isw-detector/yolo11
JOBS_FILE="$ROOT/batch_run_customaug/device${DEVICE}.txt"
PROGRESS="$ROOT/batch_run_customaug/progress_static.log"

export CUDA_VISIBLE_DEVICES="MIG-4c9f92ae-b9bf-56a3-b84f-a9db4a7637ee,MIG-5855c9a5-601e-54e5-98e0-8437cdfe54af,MIG-4ba33449-c056-5af9-91ea-c80b77eabf6c,MIG-b5beefbf-b350-5cbb-b5c4-22c4f2ba08e3,MIG-ce99f083-a10e-5d2f-aab0-9ae215817f11,MIG-ed357fc9-5f50-50f2-9711-ad83657a1804,MIG-d0efc60c-b921-53bb-b784-3d92a1c7ceab"

cd "$ROOT" || exit 1

if [ ! -f "$JOBS_FILE" ]; then
    echo "$(date -u +%FT%TZ) device$DEVICE: no jobs file, nothing to do" >> "$PROGRESS"
    exit 0
fi

while IFS=$'\t' read -r name data seed; do
    [ -z "$name" ] && continue
    echo "$(date -u +%FT%TZ) device$DEVICE: starting $name (data=$data seed=$seed)" >> "$PROGRESS"

    python3 train.py \
        --model-size n \
        --epochs 100 \
        --name "$name" \
        --data "$data" \
        --seed "$seed" \
        --batch 16 \
        --workers 16 \
        --device "$DEVICE" \
        --project "$ROOT/runs_customaug" \
        --fliplr 0.5 --flipud 0.5 --translate 0.05 --scale 0.25 --degrees 0.0 \
        > "$ROOT/batch_run_customaug/logs/${name}.log" 2>&1
    status=$?

    echo "$(date -u +%FT%TZ) device$DEVICE: finished $name (exit $status)" >> "$PROGRESS"
done < "$JOBS_FILE"

echo "$(date -u +%FT%TZ) device$DEVICE: all assigned jobs done, worker exiting" >> "$PROGRESS"
