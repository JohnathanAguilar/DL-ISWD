#!/usr/bin/env bash
# faster-rcnn Phase 3: block-level eval. 6 jobs (one per model), CPU, all parallel.
# Run from repo root so eval/... and faster-rcnn/... resolve. Note the /netcdf suffix
# on --test-dir / --val-dir (the README omits it, which is a bug).
set -u
REPO=/workspace/isw-detector
cd "$REPO"
LOGDIR=faster-rcnn/pipeline/logs
mkdir -p "$LOGDIR"
MAJ=0.25

rc=0; pids=(); names=()
for variant in gh frgb; do
  for seed in 123 456 789; do
    run="frcnn_${variant}_s${seed}"
    mkdir -p "faster-rcnn/eval_results/${run}"
    python3 eval/evaluate_block_metrics.py \
      --test-dir "faster-rcnn/results/${run}/test/netcdf" \
      --val-dir  "faster-rcnn/results/${run}/valid/netcdf" \
      --majority-threshold "$MAJ" \
      --min-conf-threshold 0.05 --max-conf-threshold 0.95 \
      --overall-csv  "faster-rcnn/eval_results/${run}/block_metrics_overall.csv" \
      --per-file-csv "faster-rcnn/eval_results/${run}/block_metrics_per_file.csv" \
      > "$LOGDIR/eval_${run}.log" 2>&1 &
    pids+=($!); names+=("$run")
  done
done
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then echo "OK   ${names[$j]}"; else echo "FAIL ${names[$j]}"; rc=1; fi
done
echo "===== faster-rcnn eval phase done (rc=$rc) ====="
exit $rc
