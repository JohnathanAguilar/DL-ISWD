#!/usr/bin/env bash
# faster-rcnn Phase 2: NetCDF inference. 12 jobs = {gh,frgb} x {123,456,789} x {test,valid}.
# infer.py accepts a bare --device index (-> cuda:N). Waves of 3 on GPUs 0/1/2.
set -u
cd "$(dirname "$0")/.."   # -> faster-rcnn/
LOGDIR=pipeline/logs
mkdir -p "$LOGDIR"
CONF=0.05
GPUS=(0 1 2)

JOBS=()
for variant in gh frgb; do
  for seed in 123 456 789; do
    for split in test valid; do
      JOBS+=("frcnn_${variant}_s${seed}|${split}")
    done
  done
done

rc=0; i=0
while [ $i -lt ${#JOBS[@]} ]; do
  pids=(); names=()
  for g in "${!GPUS[@]}"; do
    [ $i -lt ${#JOBS[@]} ] || break
    IFS='|' read -r run split <<< "${JOBS[$i]}"
    gpu="${GPUS[$g]}"
    echo "$(date +%T) launch $run $split gpu=$gpu"
    python src/inference/infer.py --run-name "$run" --split "$split" --conf "$CONF" --device "$gpu" \
      > "$LOGDIR/infer_${run}_${split}.log" 2>&1 &
    pids+=($!); names+=("$run/$split"); i=$((i+1))
  done
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "$(date +%T) OK   ${names[$j]}"; else echo "$(date +%T) FAIL ${names[$j]}"; rc=1; fi
  done
done
echo "===== faster-rcnn inference phase done (rc=$rc) ====="
exit $rc
