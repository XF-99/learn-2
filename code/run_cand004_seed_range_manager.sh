#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/seed/combo4_cand004}"
PY="${PY:-/root/miniconda3/bin/python}"
LEARN="$ROOT/code/learn.py"
WELLS="$ROOT/outputs/candidate_groups/cand004"
RUN_ROOT="$ROOT/outputs/multiseed_final/cand004"
LOG_ROOT="$ROOT/logs/multiseed_final"
SEEDS="${SEEDS:-40 41 42 43 44 45 46 47 48 49 50}"
MAX_JOBS="${MAX_JOBS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

active_count() {
  pgrep -af "$LEARN" | grep -F "$RUN_ROOT/seed_" | grep -v grep | wc -l
}

seed_active() {
  local seed="$1"
  pgrep -af "$LEARN" | grep -F "$RUN_ROOT/seed_${seed}" | grep -qv grep
}

seed_complete() {
  local seed="$1"
  test -f "$RUN_ROOT/seed_${seed}/metrics_summary.csv" && test -f "$RUN_ROOT/seed_${seed}/run_metadata.json"
}

run_seed_bg() {
  local seed="$1"
  local out_dir="$RUN_ROOT/seed_${seed}"
  local log_path="$LOG_ROOT/seed_${seed}.log"
  mkdir -p "$out_dir"
  (
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
  ) > "$log_path" 2>&1 &
  echo "[$(date '+%F %T')] launched seed=$seed pid=$!"
}

echo "[$(date '+%F %T')] seed-range manager started"
echo "ROOT=$ROOT"
echo "SEEDS=$SEEDS"
echo "MAX_JOBS=$MAX_JOBS"

while true; do
  remaining=0
  for seed in $SEEDS; do
    if seed_complete "$seed" || seed_active "$seed"; then
      continue
    fi
    remaining=$((remaining + 1))
    while [ "$(active_count)" -ge "$MAX_JOBS" ]; do
      sleep "$POLL_SECONDS"
    done
    if seed_complete "$seed" || seed_active "$seed"; then
      continue
    fi
    run_seed_bg "$seed"
  done

  incomplete=0
  for seed in $SEEDS; do
    if ! seed_complete "$seed"; then
      incomplete=$((incomplete + 1))
    fi
  done
  if [ "$incomplete" -eq 0 ]; then
    break
  fi
  if [ "$remaining" -eq 0 ]; then
    sleep "$POLL_SECONDS"
  fi
done

echo "[$(date '+%F %T')] seed-range manager finished"
