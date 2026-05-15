# Changelog

## 2026-05-15 - Replace Project With 9-Well Strict Holdout Forecasting Run

### Changed

- Replaced the previous 3-type output set with a 9-well dataset: 3 wells for 裂隙水, 3 wells for 岩溶水, and 3 wells for 孔隙水.
- Aligned all 9 wells to the same weekly span: `1969-01-13` to `2016-01-18`.
- Updated the model input features to use all available weekly variables:
  - `GWL`
  - `TASMAX`
  - `TAS`
  - `TASMIN`
  - `Humidity`
  - `Precipitation`
- Reworked the evaluation split to the stricter order:
  - `train -> val -> selection -> calib -> test -> future_holdout`
- Reserved the final 30 weeks as `future_holdout`.
- Kept `selection` only for hyperparameter selection and `calib` only for interval calibration.
- Updated `future_holdout` forecasting to use recursive GWL predictions while still using the true future weather variables.
- Updated persistence baseline behavior:
  - `test`: previous week's true GWL.
  - `future_holdout`: recursive constant baseline from the last pre-holdout true GWL.
- Added aquifer-type averaging so the three wells of each type are averaged for final cross-type comparison.

### Hyperparameters

- Re-ran lookback experiments with `batch_size=128`.
- Selected `lookback=18`.
- Re-ran dropout experiments with `lookback=18` and `batch_size=128`.
- Selected `dropout=0.4`.
- Updated `learn.py` defaults to:
  - `lookback=18`
  - `dropout=0.4`
  - `batch_size=128`

### Added

- `prepare_nine_well_common_data.py` for building the 9-well common-span dataset.
- `prepare_selected_weekly_data.py` and `validate_selected_weekly_data.py` for selected-well data preparation and validation.
- `test_learn_data_flow.py` for strict split, scaler, future holdout, and baseline checks.
- `selected_weekly_data/` with the first selected 3-well dataset.
- `selected_weekly_data_9wells_common/` with the final 9-well common-span dataset.
- `outputs_9wells_lookback/` with the new lookback sweep results.
- `outputs_9wells_dropout/` with the new dropout sweep results.

### Outputs

- Replaced `outputs/` with the final 9-well full run.
- The final full run includes:
  - `metrics_summary.csv`
  - `metrics_summary.json`
  - `metrics_by_type_summary.csv`
  - `rmse_comparison.png`
  - `nse_comparison.png`
  - `peak_metrics_summary.csv`
  - per-well test and future-holdout predictions
  - conformal 95% interval columns
  - SHAP and Transformer attention figures
  - peak analysis figures and metrics

### Validation

- `python -m py_compile learn.py`
- Final full run:

```powershell
python learn.py --out_dir outputs_final_9wells_full
```

- Final output check confirmed:
  - 9 well directories.
  - no missing expected output files.
  - every `future_holdout_predictions.csv` has exactly 30 rows.
  - `metrics_summary.csv` has 90 rows.
  - `metrics_by_type_summary.csv` has 30 rows.
  - both `test` and `future_holdout` splits contain `Persistence`, `LSTM`, `Transformer`, `TCN`, and `Stacking`.

## 2026-05-08 - Previous Lookback/Dropout Experiment Refresh

- Earlier version selected lookback/dropout using the previous 3-type setup.
- This result has been superseded by the 2026-05-15 9-well strict holdout workflow.

## 2026-04-29 - Initial Interval, SHAP, Peak, and Hyperparameter Experiments

- Added interval prediction, SHAP/attention interpretation outputs, and peak analysis.
- Added early lookback and dropout sweep scripts.
- This result has been superseded by the 2026-05-15 9-well strict holdout workflow.
