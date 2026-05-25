# 更新日志

## 2026-05-25 - 15 口井 DynamicGatedStacking 主线

### 主要变化

- 仓库主线只保留 15 口井探索性筛选实验，三类含水层各 5 口井。
- 新增并保留 `test/` 独立实验区，包含 15 井数据准备、GPU 运行入口、筛选优化、区间预测、SHAP、峰值分析、lookback sweep 和实验日志。
- 根目录 `learn.py` 默认读取 `test/selected_weekly_data_15wells_current/`。
- 当前模型输出保留 `LSTM`、`Transformer`、`TCN`、`Stacking`、`DynamicGatedStacking`。
- 实验输出不保留 `Persistence`、`DynamicGatedOnly`、`AdaptiveWeightedStacking`。
- `reproducible_model_selection.py` 升级为 test-only、自动识别模型列、输出 loss quantile、R-distribution 和成对可重复优势概率，并生成 DynamicGatedStacking 专用报告。
- README 改写为 15 井主线说明，明确该结果为探索性筛选，不作为独立无偏泛化结论。

### 当前 15 井 test 平均结果

- `DynamicGatedStacking`: RMSE 0.1853, NSE 0.9210, PICP95 0.9518, MPIW95 0.7622。
- `Transformer`: RMSE 0.1891, NSE 0.9195, PICP95 0.9600, MPIW95 0.8173。
- `LSTM`: RMSE 0.1927, NSE 0.9127, PICP95 0.9496, MPIW95 0.8098。
- `TCN`: RMSE 0.2078, NSE 0.9023, PICP95 0.9562, MPIW95 0.8466。

### 验证

```powershell
python -m py_compile learn.py reproducible_model_selection.py test_reproducible_model_selection.py validate_selected_weekly_data.py
python validate_selected_weekly_data.py
python -m unittest test_reproducible_model_selection.py
```
