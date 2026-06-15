#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/seed/combo4_cand004}"
PY="${PY:-/root/miniconda3/bin/python}"
LEARN="$ROOT/code/learn.py"
WELLS="$ROOT/outputs/candidate_groups/cand004"
RUN_ROOT="$ROOT/outputs/multiseed_final/cand004"
LOG_ROOT="$ROOT/logs/multiseed_final"
SEEDS="${SEEDS:-40 41 42 43 44 45 46 47 48 49}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

seed_active() {
  local seed="$1"
  pgrep -af "$LEARN" | grep -F "$RUN_ROOT/seed_${seed}" | grep -qv grep
}

for seed in $SEEDS; do
  out_dir="$RUN_ROOT/seed_${seed}"
  log_path="$LOG_ROOT/seed_${seed}.log"
  mkdir -p "$out_dir"

  if [ -f "$out_dir/metrics_summary.csv" ] && [ -f "$out_dir/run_metadata.json" ]; then
    echo "seed=$seed skipped_complete"
    continue
  fi
  if seed_active "$seed"; then
    echo "seed=$seed skipped_active"
    continue
  fi

  nohup "$PY" "$LEARN" \
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
    --calib_ratio 0.05 \
    > "$log_path" 2>&1 < /dev/null &

  echo "seed=$seed launched pid=$!"
done

echo "active learn.py processes:"
pgrep -af "$LEARN" | grep -F "$RUN_ROOT/seed_" | grep -v grep || true
