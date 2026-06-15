#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/seed/combo4_cand004
PY=/root/miniconda3/bin/python
cd $ROOT
exec $PY $ROOT/code/learn.py \
  --wells_dir $ROOT/outputs/candidate_groups/cand004 \
  --out_dir $ROOT/outputs/interval_wci/seed_45 \
  --well_parallel_jobs 10 \
  --lstm_head_activation gelu \
  --lookback 18 \
  --dropout 0.10 \
  --holdout_steps 52 \
  --epochs 60 \
  --gate_epochs 80 \
  --batch_size 128 \
  --mc_dropout_samples 30 \
  --seed 45 \
  --disable_explain \
  --disable_peak_analysis \
  --include_selection_metrics \
  --train_ratio 0.65 \
  --val_ratio 0.15 \
  --selection_ratio 0.10 \
  --calib_ratio 0.05 \
  --gamma 0.5 \
  --lambda_t 0.02 \
  --interval_width_scale 1.0 \
  --cwc_penalty_lambda 1.0
