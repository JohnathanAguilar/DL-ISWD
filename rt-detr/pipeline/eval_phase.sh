#!/usr/bin/env bash
# RT-DETR Part B3: block-level eval. 6 jobs (one per model), CPU, all parallel.
# Run from repo root. /netcdf suffix on --test-dir/--val-dir (README omits it -> bug).
set -u
REPO=/workspace/isw-detector
cd "$REPO"
LOGDIR=rt-detr/pipeline/logs
mkdir -p "$LOGDIR"
MAJ=0.25
RESULTS_DIR=rt-detr/results_lr5e5
EVAL_DIR=rt-detr/eval_results_lr5e5

rc=0; pids=(); names=()
for variant in ghp frgb; do
  for seed in 123 456 789; do
    run="rt_detr_${variant}_s${seed}"
    mkdir -p "${EVAL_DIR}/${run}"
    python3 eval/evaluate_block_metrics.py \
      --test-dir "${RESULTS_DIR}/${run}/test/netcdf" \
      --val-dir  "${RESULTS_DIR}/${run}/valid/netcdf" \
      --majority-threshold "$MAJ" \
      --min-conf-threshold 0.05 --max-conf-threshold 0.95 \
      --overall-csv  "${EVAL_DIR}/${run}/block_metrics_overall.csv" \
      --per-file-csv "${EVAL_DIR}/${run}/block_metrics_per_file.csv" \
      > "$LOGDIR/eval_${run}.log" 2>&1 &
    pids+=($!); names+=("$run")
  done
done
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "OK   ${names[$j]}"; else echo "FAIL ${names[$j]}"; rc=1; fi
done
echo "===== RT-DETR eval phase done (rc=$rc) ====="
exit $rc
