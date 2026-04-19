# Changelog

## 2026-04-19 - Dropout generalization experiment

Added a standalone dropout sweep experiment and uploaded the completed results.

### Added

- `dropout_experiment.py`: runs dropout sensitivity experiments without changing `learn.py`.
- `outputs_dropout/`: completed experiment outputs for dropout values `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, and `0.5`.
- `outputs_dropout/dropout_sweep_metrics.csv`: per-well and per-model metrics for every dropout value.
- `outputs_dropout/dropout_sweep_summary.csv`: grouped mean/std metrics by dropout and model.
- `outputs_dropout/dropout_rmse_comparison.png`: RMSE comparison plot across dropout values.
- `outputs_dropout/dropout_nse_comparison.png`: NSE comparison plot across dropout values.

### Main result

Across the three wells, `dropout=0.0` gave the best average generalization for LSTM, Transformer, and Stacking. TCN was slightly different: `dropout=0.1` produced the lowest average RMSE, while `dropout=0.2` produced the highest average NSE. Overall, stronger dropout did not improve this experiment and often reduced model performance.

### Reference commit

- `ec68052 Add dropout sweep experiment results`
