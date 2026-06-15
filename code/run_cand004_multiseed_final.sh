#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/seed/combo4_cand004}"
PY="${PY:-/root/miniconda3/bin/python}"
LEARN="$ROOT/code/learn.py"
WELLS="$ROOT/outputs/candidate_groups/cand004"
RUN_ROOT="$ROOT/outputs/multiseed_final/cand004"
LOG_ROOT="$ROOT/logs/multiseed_final"
SEEDS="${SEEDS:-45 46 47 48 49 50 51 52 53 54}"
MAX_JOBS="${MAX_JOBS:-3}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

echo "[$(date '+%F %T')] starting cand004 multiseed final"
echo "ROOT=$ROOT"
echo "SEEDS=$SEEDS"
echo "MAX_JOBS=$MAX_JOBS"

run_seed() {
  local seed="$1"
  local out_dir="$RUN_ROOT/seed_${seed}"
  local log_path="$LOG_ROOT/seed_${seed}.log"

  mkdir -p "$out_dir" "$LOG_ROOT"
  if [ -f "$out_dir/metrics_summary.csv" ] && [ -f "$out_dir/run_metadata.json" ]; then
    echo "[$(date '+%F %T')] seed=$seed skipped_existing"
    return 0
  fi

  {
    echo "COMMAND: $PY $LEARN --wells_dir $WELLS --out_dir $out_dir --lookback 18 --dropout 0.10 --seed $seed"
    echo "START: $(date '+%F %T')"
    "$PY" "$LEARN" \
      --wells_dir "$WELLS" \
      --out_dir "$out_dir" \
      --well_parallel_jobs 1 \
      --lstm_head_activation gelu \
      --lookback 18 \
      --dropout 0.10 \
      --holdout_steps 52 \
      --epochs 60 \
      --gate_epochs 80 \
      --batch_size 128 \
      --mc_dropout_samples 30 \
      --seed "$seed" \
      --disable_intervals \
      --disable_explain \
      --disable_peak_analysis \
      --include_selection_metrics \
      --train_ratio 0.65 \
      --val_ratio 0.15 \
      --selection_ratio 0.10 \
      --calib_ratio 0.05
    rc=$?
    echo "END: $(date '+%F %T')"
    echo "RETURNCODE: $rc"
    exit "$rc"
  } > "$log_path" 2>&1
}

fail=0
active=0
for seed in $SEEDS; do
  run_seed "$seed" &
  echo "[$(date '+%F %T')] launched seed=$seed pid=$!"
  active=$((active + 1))
  if [ "$active" -ge "$MAX_JOBS" ]; then
    set +e
    wait -n
    status=$?
    set -e
    if [ "$status" -ne 0 ]; then
      fail=1
    fi
    active=$((active - 1))
  fi
done

while [ "$active" -gt 0 ]; do
  set +e
  wait -n
  status=$?
  set -e
  if [ "$status" -ne 0 ]; then
    fail=1
  fi
  active=$((active - 1))
done

echo "[$(date '+%F %T')] cand004 multiseed final finished fail=$fail"
exit "$fail"
