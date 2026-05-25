# 地下水水位预测项目

本仓库当前只保留 **15 口井 DynamicGatedStacking 探索性筛选实验**。旧实验主线、早期 dropout/lookback 实验和无关输出已经从主线删除。

本结果是探索性筛选结果，不作为独立无偏泛化结论。当前判断只关注 `test` split，`future_holdout` 不参与模型选择结论。

## 目录结构

```text
.
├── README.md
├── CHANGELOG.md
└── test/
    ├── learn.py
    ├── run_15wells_test_focus.py
    ├── run_15wells_interval_focus.py
    ├── run_15wells_shap_focus.py
    ├── run_15wells_peak_focus.py
    ├── run_15wells_lookback_sweep.py
    ├── reproducible_model_selection.py
    ├── selected_weekly_data_15wells_current/
    └── outputs_15wells_interval_focus/
```

根目录只放项目级说明文件。15 口井实验的代码、数据、输出和测试都集中在 `test/` 目录。

## 当前数据

15 口井数据位于：

```text
test/selected_weekly_data_15wells_current/
```

三类含水层各 5 口井：

| 类型代码 | 中文类型 | 井数 |
| --- | --- | ---: |
| f | 裂隙水 | 5 |
| k | 岩溶水 | 5 |
| p | 孔隙水 | 5 |

每口井使用自己的原始可用时间跨度，不强制统一起止日期。字段固定为：

```text
Date,TASMAX,TAS,TASMIN,Humidity,Precipitation,GWL
```

井清单和候选记录：

```text
test/selected_weekly_data_15wells_current/selected_wells_summary.csv
test/selected_weekly_data_15wells_current/candidate_wells_ranked.csv
test/RUN_LOG.md
test/attempts_summary.csv
```

## 当前模型

当前保留并输出这些模型：

- `LSTM`
- `Transformer`
- `TCN`
- `DynamicGatedStacking`

当前不再输出：

- `Persistence`
- `Stacking`
- `DynamicGatedOnly`
- `AdaptiveWeightedStacking`

`DynamicGatedStacking` 使用 MC Dropout 深度模型预测、动态门控加权融合，以及弱 XGBoost 残差修正。残差是否启用由 selection split 做安全判断。

## 运行命令

点预测：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_test_focus.py
```

区间预测：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_interval_focus.py
```

SHAP 和峰值分析：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_shap_focus.py
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_peak_focus.py
```

所有 15 口井入口都会检查 GPU。如果 `torch.cuda.is_available()` 为 false，脚本会直接停止。

## 当前输出

当前保留的正式区间预测输出：

```text
test/outputs_15wells_interval_focus/
```

核心结果：

| 模型 | RMSE | NSE | PICP95 | MPIW95 |
| --- | ---: | ---: | ---: | ---: |
| DynamicGatedStacking | 0.1853 | 0.9210 | 0.9518 | 0.7622 |
| Transformer | 0.1891 | 0.9195 | 0.9600 | 0.8173 |
| LSTM | 0.1927 | 0.9127 | 0.9496 | 0.8098 |
| TCN | 0.2078 | 0.9023 | 0.9562 | 0.8466 |

按类型看，DynamicGatedStacking 在岩溶水上平均 RMSE 最优；孔隙水和裂隙水上 Transformer 略优。但 15 口井整体平均 RMSE 仍是 DynamicGatedStacking 最低。

## 可重复性分析

可重复性分析脚本位于：

```text
test/reproducible_model_selection.py
```

它使用 test 逐点 squared loss、block bootstrap 和成对概率 `P(R_A < R_B)` 构造 paper-inspired R-distribution 风险分布。默认只分析 `test`，并排除 `Persistence`、`Stacking`、`DynamicGatedOnly`、`AdaptiveWeightedStacking`。

推荐命令：

```powershell
C:\Users\xf-99\.conda\envs\Python39\python.exe test\reproducible_model_selection.py --out_dir test\outputs_15wells_interval_focus --splits test --bootstrap_method block --block_size 8 --bootstrap_samples 5000 --seed 42 --target_model DynamicGatedStacking
```

主要输出：

```text
test/outputs_15wells_interval_focus/reproducible_selection/
```

当前可重复性分析显示：DynamicGatedStacking 的平均 empirical risk 最低；对 LSTM 和 TCN 是趋势优势，对 Transformer 的优势不稳定。因此这里只写作探索性筛选优势，不写作稳定显著优于。

## 验证

编译检查：

```powershell
python -m py_compile test\learn.py test\reproducible_model_selection.py test\run_15wells_test_focus.py test\run_15wells_interval_focus.py test\run_15wells_shap_focus.py test\run_15wells_peak_focus.py test\run_15wells_lookback_sweep.py
```
