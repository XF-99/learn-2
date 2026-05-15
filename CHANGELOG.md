# 更新日志

## 2026-05-15 - 替换为 9 井严格留出预测流程

### 主要变化

- 将原来的 3 类单井结果替换为 9 井结果：裂隙水、岩溶水、孔隙水各 3 口井。
- 将 9 口井统一到相同周时间跨度：`1969-01-13` 至 `2016-01-18`。
- 模型输入特征更新为全部可用周变量：
  - `GWL`
  - `TASMAX`
  - `TAS`
  - `TASMIN`
  - `Humidity`
  - `Precipitation`
- 将评估流程改为更严格的时序划分：
  - `train -> val -> selection -> calib -> test -> future_holdout`
- 最后 30 周单独作为 `future_holdout`。
- `selection` 只用于超参数选择，`calib` 只用于区间预测校准。
- `future_holdout` 使用真实未来气象变量，但 GWL 使用递推预测。
- Persistence 基线同步改为公平口径：
  - `test`：使用目标日前一周真实 GWL。
  - `future_holdout`：使用递推口径，即保持 holdout 前最后一周真实 GWL。
- 新增按地下水类型求平均指标，用于三类地下水之间比较。

### 超参数

- 使用 `batch_size=128` 重新运行 lookback 实验。
- 最终选择 `lookback=18`。
- 使用 `lookback=18` 和 `batch_size=128` 重新运行 dropout 实验。
- 最终选择 `dropout=0.4`。
- `learn.py` 默认参数已更新为：
  - `lookback=18`
  - `dropout=0.4`
  - `batch_size=128`

### 新增内容

- `prepare_nine_well_common_data.py`：生成 9 井共同时间跨度数据集。
- `prepare_selected_weekly_data.py`：生成第一版 3 井筛选数据集。
- `validate_selected_weekly_data.py`：检查筛选数据完整性。
- `test_learn_data_flow.py`：测试严格划分、scaler、future holdout 和 baseline 逻辑。
- `selected_weekly_data/`：第一版 3 井筛选数据。
- `selected_weekly_data_9wells_common/`：最终 9 井共同时间跨度数据。
- `outputs_9wells_lookback/`：新的 lookback 超参数实验结果。
- `outputs_9wells_dropout/`：新的 dropout 超参数实验结果。

### 输出结果

- `outputs/` 已替换为最终 9 井完整运行结果。
- 最终结果包括：
  - `metrics_summary.csv`
  - `metrics_summary.json`
  - `metrics_by_type_summary.csv`
  - `rmse_comparison.png`
  - `nse_comparison.png`
  - `peak_metrics_summary.csv`
  - 每口井的 test 与 future_holdout 预测结果
  - 95% conformal 区间预测
  - SHAP 和 Transformer attention 解释图
  - 峰值分析图和指标

### 验证

- 语法检查：

```powershell
python -m py_compile learn.py lookback_experiment.py dropout_experiment.py
```

- 数据流测试：

```powershell
python test_learn_data_flow.py
```

- 最终完整运行：

```powershell
python learn.py --out_dir outputs_final_9wells_full
```

- 最终输出检查确认：
  - 共 9 个井目录。
  - 没有缺失预期输出文件。
  - 每个 `future_holdout_predictions.csv` 正好 30 行。
  - `metrics_summary.csv` 共 90 行。
  - `metrics_by_type_summary.csv` 共 30 行。
  - `test` 和 `future_holdout` 都包含 `Persistence`、`LSTM`、`Transformer`、`TCN`、`Stacking`。

## 2026-05-08 - 旧版 lookback/dropout 实验更新

- 之前基于 3 类单井设置进行 lookback 和 dropout 实验。
- 该结果已被 2026-05-15 的 9 井严格留出预测流程替代。

## 2026-04-29 - 初始区间预测、SHAP、峰值和超参数实验

- 添加区间预测、SHAP/attention 解释输出和峰值分析。
- 添加早期 lookback 和 dropout 实验脚本。
- 该结果已被 2026-05-15 的 9 井严格留出预测流程替代。
