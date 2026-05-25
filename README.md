# 地下水水位预测项目

本仓库当前只保留 **15 口井 DynamicGatedStacking 探索性筛选实验**。15 口井按含水层类型均衡选择：

- `f`：裂隙水，5 口井
- `k`：岩溶水，5 口井
- `p`：孔隙水，5 口井

当前结果是探索性筛选结果，不作为独立无偏泛化结论。筛选和复现判断均只关注 `test` split，`future_holdout` 不参与当前模型选择结论。

## 当前数据

15 口井周尺度数据位于：

```text
test/selected_weekly_data_15wells_current/
```

每口井使用自己的可用时间跨度，不强制统一起止日期。数据字段为：

```text
Date,TASMAX,TAS,TASMIN,Humidity,Precipitation,GWL
```

井清单、候选记录和筛选日志：

```text
test/selected_weekly_data_15wells_current/selected_wells_summary.csv
test/selected_weekly_data_15wells_current/candidate_wells_ranked.csv
test/RUN_LOG.md
test/attempts_summary.csv
```

## 模型

当前保留并输出的模型为：

- LSTM
- Transformer
- TCN
- Stacking
- DynamicGatedStacking

实验输出中不保留 `Persistence`、`DynamicGatedOnly`、`AdaptiveWeightedStacking`。`DynamicGatedStacking` 使用 MC Dropout 深度模型预测、动态门控融合和弱 XGBoost 残差修正，并在 selection split 上判断残差是否启用。

## 运行方式

15 井点预测入口：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_test_focus.py
```

15 井区间预测入口：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_interval_focus.py
```

15 井 SHAP 和峰值分析入口：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_shap_focus.py
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_peak_focus.py
```

所有 15 井入口都会检查 GPU；如果 `torch.cuda.is_available()` 为 false，会直接停止。

## 当前输出

当前保留的正式 15 井区间预测输出：

```text
test/outputs_15wells_interval_focus/
```

核心汇总文件：

```text
metrics_summary.csv
metrics_by_type_summary.csv
rmse_comparison.png
nse_comparison.png
reproducible_selection/
```

`test` split 上 15 口井平均指标如下：

| 模型 | RMSE | NSE | PICP95 | MPIW95 |
| --- | ---: | ---: | ---: | ---: |
| DynamicGatedStacking | 0.1853 | 0.9210 | 0.9518 | 0.7622 |
| Transformer | 0.1891 | 0.9195 | 0.9600 | 0.8173 |
| LSTM | 0.1927 | 0.9127 | 0.9496 | 0.8098 |
| TCN | 0.2078 | 0.9023 | 0.9562 | 0.8466 |

按类型看，DynamicGatedStacking 在岩溶水平均 RMSE 最优；孔隙水和裂隙水上 Transformer 略优，但 15 口井整体平均 RMSE 仍由 DynamicGatedStacking 最低。

## 可重复性分析

`reproducible_model_selection.py` 使用 test 逐点 squared loss、block bootstrap 和成对概率 `P(R_A < R_B)`，构造 paper-inspired R-distribution 风险分布。默认只分析 `test`，并排除 `Persistence`、`Stacking`、`DynamicGatedOnly`、`AdaptiveWeightedStacking`。

推荐命令：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe reproducible_model_selection.py --out_dir test\outputs_15wells_interval_focus --splits test --bootstrap_method block --block_size 8 --bootstrap_samples 5000 --seed 42 --target_model DynamicGatedStacking
```

输出目录：

```text
test/outputs_15wells_interval_focus/reproducible_selection/
```

主要输出包括：

```text
loss_quantile_functions.csv
risk_distribution_samples.csv
risk_distribution_summary.csv
pairwise_reproducible_dominance.csv
stable_model_rejections.csv
model_reproducibility_ranking.csv
dynamic_gated_reproducibility_report.md
risk_distribution_plot_test.png
pairwise_probability_heatmap_test.png
```

当前复现分析显示：DynamicGatedStacking 的平均 empirical risk 最低；对 LSTM 和 TCN 是趋势优势，对 Transformer 的优势不稳定，因此只写作探索性筛选优势，不写作稳定显著优于。

## 验证

编译检查：

```powershell
python -m py_compile learn.py reproducible_model_selection.py test\learn.py test\prepare_15wells.py test\run_15wells_test_focus.py test\run_15wells_interval_focus.py test\optimize_15wells_dynamic_gate.py validate_selected_weekly_data.py
```

单元测试：

```powershell
python -m unittest test_learn_data_flow.py test_reproducible_model_selection.py
```

数据检查：

```powershell
python validate_selected_weekly_data.py
```
