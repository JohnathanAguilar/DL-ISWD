#!/bin/bash
# Pulls jobs one at a time from the shared queue and trains them on $DEVICE.
# Usage: worker.sh <device_index> [wait_pid]
set -u

DEVICE="$1"
WAIT_PID="${2:-}"

ROOT=/workspace/isw-detector/yolo11
QUEUE="$ROOT/batch_run/queue.txt"
LOCK="$ROOT/batch_run/queue.lock"
PROGRESS="$ROOT/batch_run/progress.log"

# This container only enumerates 5 of the 7 MIG slices on the host by default
# (it silently truncates to the first 5 in `nvidia-smi -L` order). Exporting
# all 7 UUIDs explicitly makes CUDA device indices 0-6 map to all 7 slices:
#   0 = GPU0 (only slice)          4 = GPU3 slice 0
#   1 = GPU1 slice 0               5 = GPU3 slice 1  (unreachable w/o this)
#   2 = GPU1 slice 1               6 = GPU4 (only slice)  (unreachable w/o this)
#   3 = GPU2 (only slice)
export CUDA_VISIBLE_DEVICES="MIG-4c9f92ae-b9bf-56a3-b84f-a9db4a7637ee,MIG-5855c9a5-601e-54e5-98e0-8437cdfe54af,MIG-4ba33449-c056-5af9-91ea-c80b77eabf6c,MIG-b5beefbf-b350-5cbb-b5c4-22c4f2ba08e3,MIG-ce99f083-a10e-5d2f-aab0-9ae215817f11,MIG-ed357fc9-5f50-50f2-9711-ad83657a1804,MIG-d0efc60c-b921-53bb-b784-3d92a1c7ceab"

cd "$ROOT" || exit 1

if [ -n "$WAIT_PID" ]; then
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 10
    done
    echo "$(date -u +%FT%TZ) device$DEVICE: prior job (pid $WAIT_PID) finished, joining queue" >> "$PROGRESS"
fi

while true; do
    job=$(
        exec 200>"$LOCK"
        flock -x 200
        line=$(head -n1 "$QUEUE" 2>/dev/null)
        if [ -n "$line" ]; then
            tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"
        fi
        printf '%s' "$line"
    )

    if [ -z "$job" ]; then
        echo "$(date -u +%FT%TZ) device$DEVICE: queue empty, worker exiting" >> "$PROGRESS"
        break
    fi

    name=$(printf '%s' "$job" | cut -f1)
    data=$(printf '%s' "$job" | cut -f2)
    seed=$(printf '%s' "$job" | cut -f3)

    echo "$(date -u +%FT%TZ) device$DEVICE: starting $name (data=$data seed=$seed)" >> "$PROGRESS"

    python3 train.py \
        --model-size n \
        --epochs 100 \
        --name "$name" \
        --data "$data" \
        --seed "$seed" \
        --batch 16 \
        --workers 8 \
        --device "$DEVICE" \
        > "$ROOT/batch_run/logs/${name}.log" 2>&1
    status=$?

    echo "$(date -u +%FT%TZ) device$DEVICE: finished $name (exit $status)" >> "$PROGRESS"
done
