# WRR论文故事线：可重复评估的地下水水位动态集成预测

这个仓库是 `combo4_cand004` 实验的 GitHub 展示版。当前首页优先展示面向 Water Resources Research (WRR) 投稿的论文故事线：研究问题是什么、核心证据是什么、结果边界在哪里，以及后续写论文该怎样展开。

完整故事线文档见：

- [WRR论文故事线草案](docs/WRR_manuscript_story_zh.md)
- [WRR投稿叙事与证据边界](docs/WRR_story_and_evidence_zh.md)
- [中文详细说明](README_zh.md)
- [English package guide](README_en.md)
- [结果目录说明](results/README_zh.md)

## 一句话核心论点

在非平稳地下水水位预测中，`DynamicGatedStacking` 通过动态整合 `LSTM`、`Transformer` 和 `TCN`，在 11 个随机 seed 的测试期点预测中表现出最低平均 NRMSE 和最稳定排名；`WCI/CWC` 区间诊断、峰值预测和 `SHAP` 分析进一步揭示了模型在不确定性、事件捕捉和解释性方面的行为，同时 `future holdout` 的欠覆盖说明长期外推仍然存在明确边界。

**English argument sentence**

> In groundwater-level forecasting under non-stationary conditions, we show that a DynamicGatedStacking ensemble of LSTM, Transformer and TCN models improves the repeatability of test-period point prediction across 11 random seeds, while WCI/CWC interval diagnostics, peak-event evaluation and SHAP analysis expose remaining uncertainty and extrapolation limits in future-holdout conditions.

## 推荐题目

> Repeatability-aware dynamic ensemble forecasting of groundwater levels under non-stationary conditions

备选题目：

1. Dynamic ensemble learning improves repeatability in groundwater-level forecasting
2. A repeatability-aware dynamic ensemble workflow for groundwater-level forecasting
3. Dynamic ensemble forecasting reveals predictive gains and extrapolation limits in groundwater-level prediction
4. Repeatability, uncertainty and interpretability in dynamic ensemble forecasting of groundwater levels

## WRR叙事弧

### 1. Field-scale need

地下水水位预测关系到水资源管理、供水安全、生态流量和地下水系统响应评估。实际管理中，预测不只是追求平均误差低，还需要知道模型在不同井、不同含水层类型、不同时段和不同初始化下是否稳定。

### 2. Unresolved bottleneck

地下水水位序列往往具有非平稳性，且井之间含水层条件不同。单一深度学习模型可能在某些井或某些 seed 下表现很好，但在换 seed、换时段、进入 `future holdout` 后稳定性下降。因此，单次训练结果不足以支撑 WRR 论文中的强结论。

### 3. Proposed move

本文的关键动作不是“造一个更复杂模型”，而是建立一个可重复评估的动态集成预测流程：用 `LSTM`、`Transformer` 和 `TCN` 捕捉不同时间结构，再由 `DynamicGatedStacking` 根据输入和模型行为动态组合预测。

### 4. Decisive evidence

主证据来自 `results/01_repeatability_multiseed/`。在 seeds 40-50 共 11 个随机 seed 上，`DynamicGatedStacking` 在测试期取得最低平均 NRMSE，并在 9/11 个 seed 中排名第一。

| split | model | mean NRMSE | std NRMSE | mean rank | best count |
|---|---:|---:|---:|---:|---:|
| test | DynamicGatedStacking | 0.070388 | 0.000873 | 1.18 | 9/11 |
| test | Transformer | 0.071492 | 0.001190 | 1.82 | 2/11 |
| test | LSTM | 0.078062 | 0.000917 | 3.00 | 0/11 |
| test | TCN | 0.083291 | 0.001772 | 4.00 | 0/11 |

可用英文句：

> Across 11 random seeds, DynamicGatedStacking achieved the lowest mean test-period NRMSE (0.0704 +/- 0.0009) and ranked first in 9 of 11 seeds, indicating a repeatable point-prediction advantage over the individual LSTM, Transformer and TCN models.

### 5. Boundary

`future holdout` 是论文的诚实边界，也是 WRR 叙事中很有价值的一部分。`DynamicGatedStacking` 在 `future holdout` 的平均 NRMSE 最低，但 best count 少于 `LSTM`；seed45 的 `WCI` 区间在测试期接近 95% 覆盖，而 `future holdout` 覆盖率明显下降。

| split | model | mean NRMSE | std NRMSE | mean rank | best count |
|---|---:|---:|---:|---:|---:|
| future holdout | DynamicGatedStacking | 0.167971 | 0.008953 | 1.82 | 3/11 |
| future holdout | LSTM | 0.169122 | 0.009487 | 1.64 | 6/11 |
| future holdout | Transformer | 0.180153 | 0.017989 | 2.91 | 2/11 |
| future holdout | TCN | 0.195277 | 0.011070 | 3.64 | 0/11 |

可用英文句：

> The ensemble retained competitive future-holdout performance, but it did not rank first in most seeds, and interval coverage decreased substantially in the future holdout period. This contrast suggests that dynamic ensembling improves test-period robustness without eliminating the difficulty of longer-horizon extrapolation.

## 摘要故事骨架

Groundwater-level forecasting supports water-resources planning, but data-driven models can be sensitive to random initialization and may provide overconfident forecasts under non-stationary conditions. This limits the usefulness of single-run model comparisons for selecting forecasting workflows. Here we evaluated a repeatability-aware dynamic ensemble framework for groundwater-level forecasting using 15 wells from the selected `combo4_cand004` dataset. The framework combined LSTM, Transformer and TCN base learners through DynamicGatedStacking and was assessed across 11 random seeds, with additional WCI/CWC interval diagnostics, peak-event evaluation and SHAP explainability analysis for a representative seed. Across seeds 40-50, DynamicGatedStacking achieved the lowest mean test-period NRMSE (0.0704 +/- 0.0009) and ranked first in 9 of 11 seeds, indicating improved repeatability relative to the individual models. In the representative seed-45 interval analysis, test-period 95% prediction intervals approached nominal coverage, whereas future-holdout coverage decreased substantially, revealing unresolved uncertainty under later-period conditions. These results indicate that dynamic ensembling can provide a practical and repeatable groundwater-level forecasting workflow, while future-holdout diagnostics remain necessary to identify extrapolation limits.

## 结果故事线

### Result 1：多 seed 点预测显示测试期可重复优势

`DynamicGatedStacking` 在测试期点预测中具有最稳的平均优势：

- test mean NRMSE = 0.070388 +/- 0.000873
- mean rank = 1.18
- best count = 9/11

### Result 2：selection 支持但优势较接近

selection split 中 `DynamicGatedStacking` 和 `Transformer` 很接近，这说明不能夸大“全阶段压倒性优势”：

- DynamicGatedStacking: 0.059545 +/- 0.000477, best count = 6/11
- Transformer: 0.059708 +/- 0.000778, best count = 5/11

### Result 3：future holdout 暴露外推边界

`future holdout` 中 `DynamicGatedStacking` 平均 NRMSE 最低，但 `LSTM` best count 更高；长期外推仍然困难：

- DynamicGatedStacking mean NRMSE = 0.167971, best count = 3/11
- LSTM mean NRMSE = 0.169122, best count = 6/11

### Result 4：WCI/CWC 显示测试期区间接近标称覆盖，但 future holdout 欠覆盖

seed45 的 `WCI` 区间在测试期接近 95% 覆盖，`future holdout` 覆盖率明显下降。

| split | model | PICP95 | PINAW95 | CWC95 |
|---|---:|---:|---:|---:|
| test | DynamicGatedStacking | 0.954930 | 0.428172 | 0.550712 |
| test | Transformer | 0.958685 | 0.415417 | 0.522739 |
| future holdout | DynamicGatedStacking | 0.675641 | 0.443962 | 0.959975 |
| future holdout | LSTM | 0.692308 | 0.468612 | 0.926874 |

### Result 5：峰值和 SHAP 作为诊断，不作为主胜利证据

峰值预测和 `SHAP` 增强了水文事件行为和模型透明度的讨论，但它们是 seed45 代表性诊断，不是多 seed 主证据。

## 图表故事线

| Figure/Table | 主问题 | 建议内容 | 数据来源 |
|---|---|---|---|
| Figure 1 | 研究对象和流程是什么？ | 研究区/井类型 + 数据划分 + DynamicGatedStacking 流程图 | `data/`, `code/learn.py` |
| Figure 2 | 主模型是否可重复地更好？ | seeds 40-50 的 test/selection/future holdout NRMSE 和排名 | `results/01_repeatability_multiseed/` |
| Table 1 | 主结果数字是什么？ | mean NRMSE、std、mean rank、best count | `cand004_multiseed_model_summary.csv`, `cand004_multiseed_rank_summary.csv` |
| Figure 3 | 不确定性在哪里失效？ | test vs future holdout 的 PICP95/PINAW95/CWC95 | `results/03_interval_prediction_wci_cwc_seed45/` |
| Figure 4 | 峰值事件是否被捕捉？ | 三类含水层代表井峰值图 | `results/04_peak_prediction_seed45/representative_examples/` |
| Supplementary Fig. S1-S15 | 所有井峰值表现 | 15口井全图 | `results/04_peak_prediction_seed45/all_wells/` |
| Supplementary Fig. S16-S30 | SHAP解释性 | 15口井全图 | `results/05_shap_explainability_seed45/all_wells/` |
| Supplementary Table | 可重复性补充诊断 | pairwise dominance, risk distribution | `results/06_repeatability_diagnostics_optional/` |

## 结论-证据映射

| Claim | Evidence | Status |
|---|---|---|
| `DynamicGatedStacking` 在测试期点预测中具有可重复平均优势。 | 11 seeds, test mean NRMSE 最低，mean rank=1.18，best count=9/11。 | supported |
| `DynamicGatedStacking` 是当前最方便、最稳的主集成选择。 | 测试期多 seed 结果 + 代码和结果已整理成展示包。 | supported, limited to current dataset/workflow |
| `DynamicGatedStacking` 解决了长期外推问题。 | future holdout best count 只有 3/11，区间欠覆盖。 | not supported |
| `WCI` 区间在测试期接近 95% 覆盖。 | seed45 test PICP95 约 0.94-0.96。 | supported for representative seed |
| future holdout 存在明显欠覆盖。 | seed45 future holdout PICP95 约 0.63-0.69。 | supported for representative seed |
| 峰值预测支持水文事件行为诊断。 | seed45 peak metrics and plots。 | supported as diagnostic |
| `SHAP` 揭示地下水变化因果机制。 | 当前只有模型解释性图。 | not supported |

## 不要越界写

| 不建议写 | 替换成 |
|---|---|
| The proposed model significantly outperformed all baselines in all conditions. | The ensemble showed the most repeatable test-period advantage across random seeds. |
| The model solved long-term groundwater forecasting. | Future-holdout diagnostics revealed remaining extrapolation uncertainty. |
| SHAP revealed the hydrological mechanism. | SHAP provided model-level interpretability support. |
| WCI is an evaluation metric. | WCI is an interval calibration method; CWC95 is an interval evaluation metric. |
| The selected candidate won the screening. | The study used the selected `combo4_cand004` dataset for model evaluation. |
| The results are broadly replicable. | The results are repeatable across 11 random seeds. |

## 仓库结构

```text
.
├── code/       # 训练、评估、多 seed、WCI/CWC 和汇总脚本
├── data/       # combo4_cand004 的 15 口井周尺度输入数据
├── docs/       # WRR 故事线、证据边界和写作说明
├── results/    # 多 seed 主结果和 seed45 代表性诊断结果
├── README.md
├── README_zh.md
└── README_en.md
```

## 如何复跑

这些脚本最初在 Linux + CUDA 环境下运行。移动到其他机器时，需要检查 shell 脚本中的路径和 Python 环境。

多 seed final 点预测：

```bash
bash code/run_cand004_multiseed_final.sh
```

seed45 区间预测/WCI/CWC：

```bash
bash code/launch_interval_wci_seed45_parallel10.sh
```

多 seed 汇总：

```bash
python code/summarize_cand004_multiseed.py
```

## 下一步写作任务

1. 补研究区和数据背景：研究区位置、含水层类型、井的时间跨度、周尺度处理、缺失值处理。
2. 明确输入特征：模型到底用了哪些变量、滞后项和预测步长。
3. 画 Figure 1：研究区/井类型 + 数据划分 + `DynamicGatedStacking` 框架。
4. 把 `results/01_repeatability_multiseed/` 整理成主文 Table 1 和 Figure 2。
5. 从 `results/04_peak_prediction_seed45/all_wells/` 和 `results/05_shap_explainability_seed45/all_wells/` 中挑选补充材料图。
6. 写 Methods 的防泄漏说明：stacking 和 interval calibration 必须说清楚训练、selection、test、future holdout 的信息边界。
7. 正式写 Introduction 前，先确定文献框架：地下水水位预测、深度学习时序模型、集成模型、不确定性/conformal prediction、可解释性。
