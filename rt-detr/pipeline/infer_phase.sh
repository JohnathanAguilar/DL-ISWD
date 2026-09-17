#!/usr/bin/env bash
# RT-DETR Part B2: NetCDF inference. 12 jobs = {ghp,frgb} x {123,456,789} x {test,valid}.
# --conf 0.05 for all. Waves of 3 across the 3 RT-DETR MIG slices (pinned by UUID).
set -u
cd "$(dirname "$0")/.."   # -> rt-detr/
LOGDIR=pipeline/logs
mkdir -p "$LOGDIR"
CONF=0.05
RUNS_DIR=/workspace/isw-detector/rt-detr/runs_lr5e5
RESULTS_DIR=/workspace/isw-detector/rt-detr/results_lr5e5
SLICES=(
  MIG-d786569a-875c-5d71-9acc-1a55e0cb78ce
  MIG-b76979a7-08a3-5fba-9feb-881a571edc7b
  MIG-7e00f275-b657-5e60-90de-f10fd799cdac
)

JOBS=()
for variant in ghp frgb; do
  for seed in 123 456 789; do
    for split in test valid; do
      JOBS+=("rt_detr_${variant}_s${seed}|${split}")
    done
  done
done

rc=0; i=0
while [ $i -lt ${#JOBS[@]} ]; do
  pids=(); names=()
  for s in "${!SLICES[@]}"; do
    [ $i -lt ${#JOBS[@]} ] || break
    IFS='|' read -r run split <<< "${JOBS[$i]}"
    slice="${SLICES[$s]}"
    echo "$(date +%T) launch $run $split slice=${slice:0:20}..."
    CUDA_VISIBLE_DEVICES="$slice" python src/inference/infer.py \
      --run-name "$run" --split "$split" --conf "$CONF" --device cuda \
      --runs-dir "$RUNS_DIR" --results-dir "$RESULTS_DIR" \
      > "$LOGDIR/infer_${run}_${split}.log" 2>&1 &
    pids+=($!); names+=("$run/$split"); i=$((i+1))
  done
  for j in "${!pids[@]}"; do
    if wait "${pids[$j]}"; then echo "$(date +%T) OK   ${names[$j]}"; else echo "$(date +%T) FAIL ${names[$j]}"; rc=1; fi
  done
done
echo "===== RT-DETR inference phase done (rc=$rc) ====="
exit $rc
