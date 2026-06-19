# WRR叙事与证据使用说明

这个文件用于把 `combo4_cand004` 分享包中的代码、数据和结果，组织成更贴合 Water Resources Research (WRR) 投稿的论文叙事。它不是运行日志，而是给合作者、导师或审稿前自查使用的“怎么看、怎么复跑、怎么引用结论”的说明。

WRR 的官方定位是水文学和水资源领域的跨学科原创研究，关注水文过程、水资源管理及其物理、化学、生物、生态和社会维度。因此本文叙事不建议写成“某个深度学习模型获胜”，而应写成：

> 在非平稳地下水水位预测问题中，本研究通过 LSTM、Transformer 和 TCN 的 DynamicGatedStacking 集成框架，并结合多 seed 可重复性评估、WCI/CWC 区间诊断、峰值预测和 SHAP 解释性分析，展示了一个实用的地下水水位预测流程；其主要优势体现在测试期点预测的稳定提升，同时 future holdout 的区间欠覆盖揭示了长期外推边界。

官方参考：

- WRR 期刊定位：[Water Resources Research - AGU](https://wrr-submit.agu.org/)
- AGU 数据与软件要求：[Data and Software for Authors](https://www.agu.org/publications/authors/journals/data-software-for-authors)

## 1. 推荐主线

建议论文主线是：

1. 地下水水位预测面临非平稳性、含水层差异和极端/峰值事件捕捉困难。
2. 单一深度学习模型在不同井、不同时间段和不同随机初始化下可能不稳定。
3. DynamicGatedStacking 将 LSTM、Transformer 和 TCN 的预测信息进行动态集成，用于提高测试期点预测的平均表现和 seed 间稳定性。
4. 多 seed final 实验提供主文核心证据，说明集成模型在测试期最稳定。
5. WCI/CWC 区间预测、峰值预测和 SHAP 解释性分析作为代表性 seed 诊断，说明模型不确定性、事件行为和可解释性。
6. future holdout 的性能和区间覆盖下降应作为外推边界讨论，而不是强行写成全面胜利。

一句话判断：这篇稿子最稳的 WRR 卖点不是“新模型最强”，而是“一个可重复评估、带不确定性和解释性诊断的地下水水位预测框架”。

## 2. 术语统一表

| 术语 | 建议写法 | 说明 |
|---|---|---|
| 候选数据组 | `combo4_cand004` | 论文和分享包中使用的最终数据支持；不建议在正文展开候选筛选过程。 |
| 主集成模型 | `DynamicGatedStacking` | 作为主模型写，不要频繁替换成其他中文译名。 |
| 基线模型 | `LSTM`, `Transformer`, `TCN` | 三个单模型基线。 |
| 可重复性 | 可重复 | 用户已指定用“可重复”，避免换用其他中文译法。 |
| WCI | Weighted Conformal Inference | 区间构造/校准方法，不是评价指标。 |
| CWC95 | Coverage Width-based Criterion at 95% | 区间评价指标，综合覆盖率和区间宽度。 |
| PICP95 | Prediction Interval Coverage Probability at 95% | 95% 预测区间覆盖率。 |
| PINAW95 | Prediction Interval Normalized Average Width at 95% | 归一化区间宽度，越小通常表示区间越窄。 |
| future holdout | future holdout | 建议保留英文或在首次出现时写“未来保留期”。 |
| SHAP | SHAP explainability analysis | 解释性诊断，不作为因果机制证据。 |

## 3. 证据层级

| 证据层级 | 目录 | 能支持什么 | 不能支持什么 |
|---|---|---|---|
| 主文核心证据 | `results/01_repeatability_multiseed/` | seeds 40-50 的点预测可重复性；DynamicGatedStacking 在测试期平均 NRMSE 最低且排名最稳定。 | 不支持区间、峰值、SHAP 的多 seed 可重复性。 |
| 单次代表性点预测 | `results/02_point_prediction_seed45/` | 展示 seed45 的完整点预测结果和图件。 | 不能替代多 seed 主证据。 |
| 区间预测诊断 | `results/03_interval_prediction_wci_cwc_seed45/` | WCI 区间构造、PICP95/PINAW95/CWC95 评价；用于不确定性和外推边界讨论。 | 当前不是多 seed 区间可重复性证据。 |
| 峰值预测诊断 | `results/04_peak_prediction_seed45/` | 代表性 seed 下的峰值事件捕捉能力；`all_wells/` 可用于挑选论文图。 | 不能说峰值预测已经跨 seed 验证。 |
| SHAP 解释性诊断 | `results/05_shap_explainability_seed45/` | 代表性 seed 下的模型解释性图；`all_wells/` 可用于挑选补充材料图。 | 不能写成因果解释或水文机制证明。 |
| 补充诊断 | `results/06_repeatability_diagnostics_optional/` | 可作为补充材料支撑模型稳定性讨论。 | 不建议取代主表。 |

注意：当前多 seed 结果是 seeds 40-50，共 11 个 seed。论文中如果写“10 个 seed”，需要重新汇总为 10 个 seed 或在文字中统一改为“11 个随机 seed”。

## 4. 当前最稳结论

### 4.1 点预测主结论

多 seed final 点预测是当前最强证据。主文可以写：

> Across 11 random seeds, DynamicGatedStacking showed the lowest mean NRMSE and the most stable ranking on the test period, indicating that dynamic ensemble learning improved the repeatability of groundwater-level point prediction.

对应结果：

| split | 模型 | mean NRMSE | std NRMSE | 排名信息 |
|---|---:|---:|---:|---|
| test | DynamicGatedStacking | 0.070388 | 0.000873 | mean rank = 1.18; best count = 9/11 |
| test | Transformer | 0.071492 | 0.001190 | mean rank = 1.82; best count = 2/11 |
| test | LSTM | 0.078062 | 0.000917 | mean rank = 3.00; best count = 0/11 |
| test | TCN | 0.083291 | 0.001772 | mean rank = 4.00; best count = 0/11 |

selection split 也支持类似判断，但优势更接近：

| split | 模型 | mean NRMSE | std NRMSE | 排名信息 |
|---|---:|---:|---:|---|
| selection | DynamicGatedStacking | 0.059545 | 0.000477 | mean rank = 1.45; best count = 6/11 |
| selection | Transformer | 0.059708 | 0.000778 | mean rank = 1.55; best count = 5/11 |

future holdout 要谨慎写：

| split | 模型 | mean NRMSE | std NRMSE | 排名信息 |
|---|---:|---:|---:|---|
| future holdout | DynamicGatedStacking | 0.167971 | 0.008953 | mean rank = 1.82; best count = 3/11 |
| future holdout | LSTM | 0.169122 | 0.009487 | mean rank = 1.64; best count = 6/11 |
| future holdout | Transformer | 0.180153 | 0.017989 | mean rank = 2.91; best count = 2/11 |
| future holdout | TCN | 0.195277 | 0.011070 | mean rank = 3.64; best count = 0/11 |

推荐写法：

> DynamicGatedStacking retained competitive future-holdout performance, with the lowest mean per-well NRMSE, but it did not rank first in most seeds. This contrast suggests that the ensemble mainly improved test-period robustness, whereas longer-horizon extrapolation remained challenging.

不要写：

- DynamicGatedStacking 在所有时间段、所有指标上都显著最优。
- 集成模型已经解决长期外推问题。
- future holdout 证明模型可以稳定预测未来地下水变化。

## 5. 区间预测、CWC 和 WCI 怎么写

区间预测目录：`results/03_interval_prediction_wci_cwc_seed45/`

最重要的口径：

- WCI 是区间构造/校准方法。
- CWC95 是评价指标，用来惩罚“覆盖率不足”或“区间过宽”的情况。
- PICP95 接近 0.95 表示 95% 预测区间的覆盖率接近标称水平。
- future holdout 覆盖率下降是重要发现，应写成“不确定性诊断揭示外推边界”。

seed45 的汇总均值：

| split | 模型 | NRMSE_range | PICP95 | PINAW95 | CWC95 |
|---|---|---:|---:|---:|---:|
| test | DynamicGatedStacking | 0.071576 | 0.954930 | 0.428172 | 0.550712 |
| test | Transformer | 0.074703 | 0.958685 | 0.415417 | 0.522739 |
| test | LSTM | 0.078142 | 0.952113 | 0.446332 | 0.579250 |
| test | TCN | 0.085391 | 0.940845 | 0.440292 | 0.592388 |
| future holdout | LSTM | 0.170078 | 0.692308 | 0.468612 | 0.926874 |
| future holdout | DynamicGatedStacking | 0.174264 | 0.675641 | 0.443962 | 0.959975 |
| future holdout | TCN | 0.188104 | 0.670513 | 0.445226 | 0.965831 |
| future holdout | Transformer | 0.207103 | 0.630769 | 0.431543 | 0.955738 |

推荐写法：

> In the representative seed-45 run, WCI-calibrated intervals achieved near-nominal 95% coverage on the test period for most models. However, coverage decreased substantially in the future holdout period, indicating that the conformal intervals calibrated from historical behavior did not fully absorb the distributional shift in later groundwater dynamics.

中文理解：

> 测试期区间基本能覆盖真实水位，但 future holdout 覆盖率明显不足。这不是坏事，反而是 WRR 审稿人会关心的诚实边界：模型在历史检验期表现稳定，但面对更远期、可能非平稳的地下水过程，不确定性仍被低估。

## 6. 峰值预测怎么放

峰值预测目录：`results/04_peak_prediction_seed45/`

建议用途：

- 主文可以放 1 个综合峰值指标表或 1 个代表性峰值图。
- 补充材料可以放三类含水层代表井图。
- `all_wells/` 中 15 个井的全部图件用于后续挑选，不必全部进入主文。

推荐写法：

> Peak-event diagnostics were used to examine whether the models captured hydrologically relevant short-term fluctuations, rather than only minimizing average errors. These diagnostics were conducted for a representative seed and should be interpreted as event-behavior evidence, not as a multi-seed repeatability result.

不要写：

- 峰值预测结果已经跨 seed 验证。
- 峰值预测证明模型可以预测极端地下水事件。
- 单个漂亮图可以代表全部井。

## 7. SHAP 怎么放

SHAP 目录：`results/05_shap_explainability_seed45/`

建议用途：

- SHAP 放补充材料最稳。
- 主文可只放一张解释性概览图，说明模型使用的时间滞后/输入特征具有可解释模式。
- 不要把 SHAP 写成因果机制证明。

推荐写法：

> SHAP analysis was used as a diagnostic tool to inspect the relative contribution of input features and lagged groundwater states in a representative seed. The results provide interpretability support for the forecasting workflow but do not establish causal hydrological mechanisms.

中文理解：

> SHAP 是解释模型“看重什么”，不是证明地下水系统“为什么这样变化”。WRR 里可以用它增强透明度，但不要让它承担因果解释。

## 8. 主文图表建议

建议主文结构：

| 图表 | 内容 | 数据来源 | 放主文还是补充 |
|---|---|---|---|
| Figure 1 | 数据区、井类型、建模流程、DynamicGatedStacking 框架 | `data/`, `code/learn.py` | 主文 |
| Figure 2 | 多 seed test/selection/future holdout 点预测表现 | `01_repeatability_multiseed/` | 主文 |
| Table 1 | seeds 40-50 的 mean NRMSE、std、mean rank、best count | `cand004_multiseed_model_summary.csv`, `cand004_multiseed_rank_summary.csv` | 主文 |
| Figure 3 | seed45 区间预测覆盖率和区间宽度，突出 future holdout 欠覆盖 | `03_interval_prediction_wci_cwc_seed45/` | 主文或补充 |
| Figure 4 | 代表井峰值预测图 | `04_peak_prediction_seed45/representative_examples/` | 主文或补充 |
| Supplementary Figure | 15 个井 SHAP/峰值图 | `04_peak_prediction_seed45/all_wells/`, `05_shap_explainability_seed45/all_wells/` | 补充 |
| Supplementary Table | optional repeatability diagnostics | `06_repeatability_diagnostics_optional/` | 补充 |

主文篇幅紧张时，推荐保留 Figure 1、Figure 2、Table 1，把区间、峰值、SHAP 放补充材料。但 Discussion 中仍要引用区间欠覆盖，因为这是很好的 WRR 式边界讨论。

## 9. 结论-证据映射

| 可写结论 | 证据 | 状态 |
|---|---|---|
| DynamicGatedStacking 是当前最方便、最稳的主集成模型选择。 | 多 seed test split 中 mean NRMSE 最低，best count = 9/11，mean rank = 1.18。 | 支持 |
| DynamicGatedStacking 在测试期点预测中有可重复的平均优势。 | `01_repeatability_multiseed/` seeds 40-50。 | 支持 |
| DynamicGatedStacking 在 future holdout 中保持竞争力。 | future holdout mean NRMSE 最低，但 best count 少于 LSTM。 | 部分支持，需要谨慎 |
| WCI 区间在测试期接近 95% 标称覆盖。 | seed45 `PICP95` 均值约 0.94-0.96。 | 支持，限 seed45 |
| future holdout 中存在明显区间欠覆盖。 | seed45 future holdout `PICP95` 均值约 0.63-0.69。 | 支持，限 seed45 |
| 峰值预测能帮助检查水文事件捕捉能力。 | `04_peak_prediction_seed45/` 指标与图件。 | 支持，限 seed45 |
| SHAP 提供模型解释性诊断。 | `05_shap_explainability_seed45/` 图件。 | 支持，限 seed45 |
| 模型解释揭示了地下水变化的因果机制。 | 当前只有 SHAP，不是因果实验。 | 不支持 |
| 模型已解决所有未来预测和非平稳外推问题。 | future holdout 性能和覆盖率显示仍有边界。 | 不支持 |

## 10. 怎么复跑

这些脚本是在 Linux + CUDA 环境下运行的。如果移动目录，需要先检查脚本中的路径和数据目录。

多 seed final 点预测：

```bash
bash code/run_cand004_multiseed_final.sh
```

seed45 区间预测、WCI 和 CWC：

```bash
bash code/launch_interval_wci_seed45_parallel10.sh
```

多 seed 汇总：

```bash
python code/summarize_cand004_multiseed.py
```

建议复跑顺序：

1. 先跑多 seed final 点预测，这是主文核心。
2. 再跑 seed45 区间预测，用于不确定性诊断。
3. 峰值预测和 SHAP 用 seed45 作为代表性补充即可，除非论文后续想把它们也写成多 seed 证据。

## 11. AGU/WRR 数据与代码说明建议

AGU 要求支撑论文的数据、软件和其他研究对象在审稿和发表阶段可获取，并在 Open Research / Availability Statement 中说明如何访问。当前分享包已经接近这个需求，但正式投稿前还建议：

1. 把最终代码和结果上传到可长期保存的仓库，例如 Zenodo、Figshare、OSF 或 GitHub + Zenodo DOI。
2. 在论文中给出数据和软件的 DOI 或稳定链接。
3. 明确哪些数据是原始数据，哪些是处理后的周尺度水位序列。
4. 明确运行环境，包括 Python、PyTorch、CUDA、主要依赖版本和 GPU 信息。
5. 给出从数据到主表/主图的最短复跑路径。

可以使用的 Data and Software Availability 草稿：

> The processed weekly groundwater-level time series, model evaluation outputs, and scripts used to reproduce the main analyses are archived in [repository name and DOI to be added]. The archived package includes the selected `combo4_cand004` data, training and evaluation scripts, multi-seed point-prediction summaries, representative WCI/CWC interval diagnostics, peak-prediction diagnostics, and SHAP explainability outputs. The main repeatability results can be regenerated using `code/run_cand004_multiseed_final.sh`, followed by `code/summarize_cand004_multiseed.py`.

## 12. 摘要/结果/讨论可用句子

### 摘要结果句

> Across 11 random seeds, the dynamic gated ensemble achieved the lowest mean test-period NRMSE and ranked first in 9 of 11 seeds, indicating improved repeatability relative to individual LSTM, Transformer, and TCN models.

### 摘要边界句

> Future-holdout interval diagnostics further showed reduced coverage under later-period conditions, highlighting the remaining uncertainty of groundwater-level extrapolation under non-stationary dynamics.

### Results 句

> On the test period, DynamicGatedStacking achieved a mean NRMSE of 0.0704 ± 0.0009 across seeds 40-50, outperforming Transformer, LSTM, and TCN in mean error and ranking first in 9 of 11 seeds.

### Discussion 句

> The contrast between stable test-period performance and reduced future-holdout interval coverage suggests that dynamic ensembling improves short-term predictive robustness but does not remove the difficulty of extrapolating groundwater dynamics beyond the calibration regime.

### Limitations 句

> Interval prediction, peak-event diagnostics, and SHAP analyses were conducted for a representative seed and therefore should be interpreted as diagnostic evidence rather than as full multi-seed repeatability tests.

## 13. 最重要的写作边界

可以大胆写：

- DynamicGatedStacking 是当前主模型最合理的选择。
- 测试期点预测具有多 seed 可重复优势。
- 区间预测揭示了 future holdout 外推不确定性。
- 峰值预测和 SHAP 增强了结果解释和水文诊断。

必须谨慎写：

- future holdout 中 DynamicGatedStacking 不是每个 seed 都第一。
- 区间、峰值、SHAP 目前是 seed45 代表性诊断。
- SHAP 不是因果机制证据。

不要写：

- 本研究证明模型在所有地下水系统中普适最优。
- 本研究彻底解决非平稳地下水预测。
- CWC 是一个训练方法。
- WCI 是一个评价指标。
- combo4 是通过复杂筛选得到的最终胜者。正文不需要讲候选筛选过程，直接把它作为本研究分析数据集即可。
