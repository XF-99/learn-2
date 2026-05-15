# Groundwater Forecasting

This repository trains and evaluates weekly groundwater level forecasting models for three aquifer types:

- `f` / 裂隙水
- `k` / 岩溶水
- `p` / 孔隙水

The current version uses 9 wells in total, 3 wells per aquifer type. All wells are aligned to the same weekly time span so model metrics can be averaged by aquifer type.

## Current Dataset

The model reads weekly CSV files from:

```text
selected_weekly_data_9wells_common/
```

The common time span is:

```text
1969-01-13 to 2016-01-18
```

Each well file has 2454 weekly rows and the same columns:

```text
Date,TASMAX,TAS,TASMIN,Humidity,Precipitation,GWL
```

The selected wells are documented in:

```text
selected_weekly_data_9wells_common/nine_wells_summary.csv
```

The legacy 3-type CSV files in the repository root are also updated, but the main training code now uses the 9-well common-span dataset.

## Model Pipeline

`learn.py` runs a strict time-series workflow:

```text
train -> val -> selection -> calib -> test -> future_holdout
```

The split usage is:

- `train`: neural network fitting.
- `val`: early stopping and stacking residual model training.
- `selection`: lookback/dropout hyperparameter selection only.
- `calib`: conformal interval calibration only.
- `test`: historical final evaluation.
- `future_holdout`: final 30 weeks, recursively forecast with true future weather variables.

The model features are:

```text
GWL,TASMAX,TAS,TASMIN,Humidity,Precipitation
```

The model comparison includes:

- Persistence baseline
- LSTM
- Transformer
- TCN
- XGBoost residual stacking

For `future_holdout`, both model forecasts and the persistence baseline use recursive GWL input. The first future step starts from the last observed GWL before the holdout; later steps use prior predictions.

## Final Settings

The current default final settings are:

```text
lookback = 18
dropout = 0.4
batch_size = 128
holdout_steps = 30
selection_ratio = 0.1
calib_ratio = 0.15
```

The selected hyperparameters came from selection-split experiments, so the test and future holdout splits are not used for hyperparameter selection.

## Run

Full run with intervals, SHAP, and peak analysis:

```powershell
python learn.py --out_dir outputs
```

Basic run without interval prediction, SHAP, or peak analysis:

```powershell
python learn.py --out_dir outputs_basic --disable_intervals --disable_explain --disable_peak_analysis --batch_size 128
```

Lookback sweep:

```powershell
python lookback_experiment.py --batch_size 128
```

Dropout sweep:

```powershell
python dropout_experiment.py --lookback 18 --batch_size 128
```

## Outputs

The committed final outputs are in:

```text
outputs/
```

Root-level outputs include:

```text
metrics_summary.csv
metrics_summary.json
metrics_by_type_summary.csv
rmse_comparison.png
nse_comparison.png
peak_metrics_summary.csv
```

Each well directory includes:

```text
test_predictions.csv
future_holdout_predictions.csv
test_predictions.png
future_holdout_predictions.png
stacking_residuals.png
time_frequency.png
explain/
peak/
```

Prediction CSV files include conformal 95% interval columns:

```text
PI95_Lower,PI95_Upper
```

The `explain/` directory contains Transformer attention and SHAP figures. The `peak/` directory contains peak detection plots and peak metrics.

Hyperparameter experiment outputs are stored in:

```text
outputs_9wells_lookback/
outputs_9wells_dropout/
```

## Validation

The data-flow tests check the strict split logic, train-only scaler fitting, future weather alignment, recursive holdout behavior, and persistence baselines:

```powershell
python -m pytest test_learn_data_flow.py
```

The final full run was also checked for:

- 9 well output directories.
- 90 individual metric rows.
- 30 type-averaged metric rows.
- `test` and `future_holdout` metrics for all 5 models.
- 30 rows in every `future_holdout_predictions.csv`.
- No missing SHAP, interval, or peak-analysis outputs.
