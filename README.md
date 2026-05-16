# 地下水水位预测项目

本项目用于三类地下水含水层的周尺度地下水水位预测：

- `f`：裂隙水
- `k`：岩溶水
- `p`：孔隙水

当前版本共使用 9 口井，每类地下水 3 口井。所有井的数据已经统一到相同的周时间跨度，便于按地下水类型计算平均指标并进行对比。

## 当前数据集

主程序默认读取以下目录中的 9 井周数据：

```text
selected_weekly_data_9wells_common/
```

9 口井统一后的时间范围为：

```text
1969-01-13 至 2016-01-18
```

每口井均有 2454 条周数据，字段顺序一致：

```text
Date,TASMAX,TAS,TASMIN,Humidity,Precipitation,GWL
```

9 口井清单记录在：

```text
selected_weekly_data_9wells_common/nine_wells_summary.csv
```

仓库中已移除旧版 3 井数据集和根目录类型级周数据文件；当前训练主程序只使用 `selected_weekly_data_9wells_common/` 中的 9 井共同时间跨度数据。

## 模型流程

`learn.py` 使用严格的时间序列划分：

```text
train -> val -> selection -> calib -> test -> future_holdout
```

各数据段用途如下：

- `train`：训练神经网络模型。
- `val`：用于早停，并训练 stacking 残差模型。
- `selection`：只用于 lookback 和 dropout 等超参数选择。
- `calib`：只用于 conformal 区间预测校准。
- `test`：历史测试集最终评估。
- `future_holdout`：最后 30 周，模拟未来预测，只使用真实气象变量，GWL 递推预测。

模型输入特征为：

```text
GWL,TASMAX,TAS,TASMIN,Humidity,Precipitation
```

对比模型包括：

- Persistence 基线模型
- LSTM
- Transformer
- TCN
- XGBoost 残差 Stacking

在 `future_holdout` 中，模型和 Persistence 基线都采用递推口径。第一周使用 holdout 前最后一周真实 GWL，之后每一步使用上一周预测 GWL 继续递推。

## 最终参数

当前最终默认参数为：

```text
lookback = 18
dropout = 0.4
batch_size = 128
holdout_steps = 30
selection_ratio = 0.1
calib_ratio = 0.15
```

这些超参数来自 `selection` 数据段上的实验结果，没有使用 `test` 或 `future_holdout` 选择参数，因此不会污染最终测试集。

## 运行方式

完整运行，包含区间预测、SHAP 解释和峰值分析：

```powershell
python learn.py --out_dir outputs
```

只跑基础预测，不跑区间预测、SHAP 和峰值分析：

```powershell
python learn.py --out_dir outputs_basic --disable_intervals --disable_explain --disable_peak_analysis --batch_size 128
```

运行 lookback 超参数实验：

```powershell
python lookback_experiment.py --batch_size 128
```

运行 dropout 超参数实验：

```powershell
python dropout_experiment.py --lookback 18 --batch_size 128
```

## 输出结果

当前 GitHub 中已提交的最终结果位于：

```text
outputs/
```

根目录输出包括：

```text
metrics_summary.csv
metrics_summary.json
metrics_by_type_summary.csv
rmse_comparison.png
nse_comparison.png
peak_metrics_summary.csv
```

每口井目录中包括：

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

预测 CSV 中包含 95% 区间预测列：

```text
PI95_Lower,PI95_Upper
```

`explain/` 目录中包含 Transformer attention 和 SHAP 图。`peak/` 目录中包含峰值识别图和峰值指标。

超参数实验结果位于：

```text
outputs_9wells_lookback/
outputs_9wells_dropout/
```

## 验证

数据流测试覆盖严格时序划分、训练集 scaler 拟合、future holdout 气象变量对齐、递推预测和 Persistence baseline：

```powershell
python test_learn_data_flow.py
```

最终完整运行已经检查：

- 共 9 个井输出目录。
- `metrics_summary.csv` 共 90 行。
- `metrics_by_type_summary.csv` 共 30 行。
- `test` 和 `future_holdout` 都包含 5 个模型。
- 每个 `future_holdout_predictions.csv` 正好 30 行。
- 区间预测、SHAP 解释和峰值分析文件均已生成。
