# WRR论文故事线草案：可重复评估的地下水水位动态集成预测

这个文件是后续写 Water Resources Research (WRR) 论文用的“故事蓝图”。它不替代 `docs/WRR_story_and_evidence_zh.md`，而是把当前 `combo4_cand004` 的证据组织成一篇论文应该怎样展开：主论点、章节叙事、图表顺序、英文可用句和不能越界的地方。

WRR 叙事的核心不是“一个深度学习模型赢了”，而是：

> 面向非平稳地下水水位预测，本研究提出并评估了一个可重复的动态集成预测流程。该流程以 LSTM、Transformer 和 TCN 为基础模型，通过 DynamicGatedStacking 进行动态集成，并用多 seed 点预测、WCI/CWC 区间诊断、峰值预测和 SHAP 解释性分析共同评估预测能力、外推不确定性和模型透明度。

## 1. Skill路由和写作定位

本故事按 `nature-writing` 的写作逻辑组织：

- `paper_type = research`：不是纯算法论文，而是以地下水预测问题和水资源意义为中心的研究论文。
- `section = full manuscript story`：覆盖题目、摘要、引言、方法、结果、讨论和结论。
- `language = zh-to-en`：中文思路组织，保留可直接迁移到英文稿的关键句。
- `journal = WRR-oriented generic`：WRR 不是 Nature 系列，但同样需要清楚回答“为什么这个水科学问题重要、结果可信吗、边界在哪里”。

写作变体选择：`application-first`。先从地下水水位预测和水资源管理需求进入，再收窄到非平稳性、随机 seed 不稳定、外推不确定性这些技术瓶颈，最后引出 DynamicGatedStacking 和诊断评估体系。

## 2. 一句话核心论点

**中文核心论点**

在非平稳地下水水位预测中，DynamicGatedStacking 通过动态整合 LSTM、Transformer 和 TCN，在 11 个随机 seed 的测试期点预测中表现出最低平均 NRMSE 和最稳定排名；WCI/CWC 区间诊断、峰值预测和 SHAP 分析进一步揭示了模型在不确定性、事件捕捉和解释性方面的行为，同时 future holdout 的欠覆盖说明长期外推仍然存在明确边界。

**English argument sentence**

In groundwater-level forecasting under non-stationary conditions, we show that a DynamicGatedStacking ensemble of LSTM, Transformer and TCN models improves the repeatability of test-period point prediction across 11 random seeds, while WCI/CWC interval diagnostics, peak-event evaluation and SHAP analysis expose remaining uncertainty and extrapolation limits in future-holdout conditions.

这句话就是整篇论文的“脊柱”。后面每个章节都要服务它。

## 3. 术语统一表

| Canonical term | First-use definition | 写作决定 |
|---|---|---|
| groundwater-level forecasting | groundwater-level forecasting | 用作论文任务名，避免频繁换成 groundwater prediction / water-level simulation。 |
| DynamicGatedStacking | DynamicGatedStacking ensemble | 主模型名称保持英文，不再另起中文译名。 |
| LSTM | long short-term memory (LSTM) network | 第一次展开，之后用 LSTM。 |
| Transformer | Transformer model | 保持首字母大写。 |
| TCN | temporal convolutional network (TCN) | 第一次展开，之后用 TCN。 |
| repeatability | repeatability / 可重复 | 中文统一写“可重复”，不要换用其他中文译法。 |
| WCI | weighted conformal inference (WCI) | 区间构造/校准方法，不是指标。 |
| CWC95 | coverage width-based criterion at 95% nominal coverage (CWC95) | 区间评价指标。 |
| PICP95 | prediction interval coverage probability at 95% nominal coverage (PICP95) | 95% 区间覆盖率。 |
| PINAW95 | prediction interval normalized average width at 95% nominal coverage (PINAW95) | 归一化区间宽度。 |
| future holdout | future holdout period | 建议保留英文，首次可写“未来保留期”。 |
| SHAP | SHapley Additive exPlanations (SHAP) | 解释性诊断，不写成因果机制证明。 |

## 4. 故事弧：WRR审稿人应该怎样读懂这篇论文

### 4.1 Field-scale need

地下水水位预测关系到水资源管理、供水安全、生态流量和地下水系统响应评估。实际管理中，预测不只是追求平均误差低，还需要知道模型在不同井、不同含水层类型、不同时段和不同初始化下是否稳定。

可用英文句：

> Reliable groundwater-level forecasts are needed for water-resources planning, but operational usefulness depends not only on average accuracy, but also on repeatability, uncertainty characterization and robustness under changing hydrologic conditions.

### 4.2 Unresolved bottleneck

地下水水位序列往往具有非平稳性，且井之间含水层条件不同。单一深度学习模型可能在某些井或某些 seed 下表现很好，但在换 seed、换时段、进入 future holdout 后稳定性下降。因此，单次训练结果不足以支撑 WRR 论文中的强结论。

可用英文句：

> A single trained model can provide an apparently strong forecast, yet such evidence is fragile when model rankings change across random initializations or when calibrated uncertainty fails under later-period conditions.

### 4.3 Proposed move

本文的关键动作不是“造一个更复杂模型”，而是建立一个可重复评估的动态集成预测流程：用 LSTM、Transformer 和 TCN 捕捉不同时间结构，再由 DynamicGatedStacking 根据输入和模型行为动态组合预测。

可用英文句：

> We therefore evaluated a dynamic ensemble workflow that combines recurrent, attention-based and convolutional temporal learners, and assessed it using repeatability-aware point prediction and diagnostic uncertainty analyses.

### 4.4 Decisive evidence

主证据来自 `results/01_repeatability_multiseed/`。在 seeds 40-50 共 11 个随机 seed 上，DynamicGatedStacking 在测试期取得最低平均 NRMSE，并在 9/11 个 seed 中排名第一。

| split | model | mean NRMSE | std NRMSE | mean rank | best count |
|---|---:|---:|---:|---:|---:|
| test | DynamicGatedStacking | 0.070388 | 0.000873 | 1.18 | 9/11 |
| test | Transformer | 0.071492 | 0.001190 | 1.82 | 2/11 |
| test | LSTM | 0.078062 | 0.000917 | 3.00 | 0/11 |
| test | TCN | 0.083291 | 0.001772 | 4.00 | 0/11 |

可用英文句：

> Across 11 random seeds, DynamicGatedStacking achieved the lowest mean test-period NRMSE (0.0704 +/- 0.0009) and ranked first in 9 of 11 seeds, indicating a repeatable point-prediction advantage over the individual LSTM, Transformer and TCN models.

### 4.5 Broader implication

这说明动态集成可以作为当前数据条件下“最方便、最稳”的主模型选择。注意这里的“最稳”限定在测试期点预测和当前数据集，不扩展到所有地下水系统、所有指标或所有未来情景。

可用英文句：

> These results support DynamicGatedStacking as a practical default ensemble choice for this groundwater-level forecasting setting, particularly when model selection must account for random-seed sensitivity.

### 4.6 Boundary

future holdout 是论文的诚实边界，也是 WRR 叙事中很有价值的一部分。DynamicGatedStacking 在 future holdout 的平均 NRMSE 最低，但 best count 少于 LSTM；seed45 的 WCI 区间在测试期接近 95% 覆盖，而 future holdout 覆盖率明显下降。

| split | model | mean NRMSE | std NRMSE | mean rank | best count |
|---|---:|---:|---:|---:|---:|
| future holdout | DynamicGatedStacking | 0.167971 | 0.008953 | 1.82 | 3/11 |
| future holdout | LSTM | 0.169122 | 0.009487 | 1.64 | 6/11 |
| future holdout | Transformer | 0.180153 | 0.017989 | 2.91 | 2/11 |
| future holdout | TCN | 0.195277 | 0.011070 | 3.64 | 0/11 |

可用英文句：

> The ensemble retained competitive future-holdout performance, but it did not rank first in most seeds, and interval coverage decreased substantially in the future holdout period. This contrast suggests that dynamic ensembling improves test-period robustness without eliminating the difficulty of longer-horizon extrapolation.

## 5. 题目候选

最推荐：

> Repeatability-aware dynamic ensemble forecasting of groundwater levels under non-stationary conditions

备选：

1. Dynamic ensemble learning improves repeatability in groundwater-level forecasting
2. A repeatability-aware dynamic ensemble workflow for groundwater-level forecasting
3. Dynamic ensemble forecasting reveals predictive gains and extrapolation limits in groundwater-level prediction
4. Repeatability, uncertainty and interpretability in dynamic ensemble forecasting of groundwater levels

我的判断：第 1 个最贴 WRR，因为它把“地下水水位”“可重复”“动态集成”“非平稳”都放进去了，而且没有夸成普适最优。

## 6. 摘要故事骨架

摘要建议按 `context -> gap -> approach -> key result -> diagnostic result -> implication/boundary` 写。

**Draft abstract skeleton**

Groundwater-level forecasting supports water-resources planning, but data-driven models can be sensitive to random initialization and may provide overconfident forecasts under non-stationary conditions. This limits the usefulness of single-run model comparisons for selecting forecasting workflows. Here we evaluated a repeatability-aware dynamic ensemble framework for groundwater-level forecasting using 15 wells from the selected `combo4_cand004` dataset. The framework combined LSTM, Transformer and TCN base learners through DynamicGatedStacking and was assessed across 11 random seeds, with additional WCI/CWC interval diagnostics, peak-event evaluation and SHAP explainability analysis for a representative seed. Across seeds 40-50, DynamicGatedStacking achieved the lowest mean test-period NRMSE (0.0704 +/- 0.0009) and ranked first in 9 of 11 seeds, indicating improved repeatability relative to the individual models. In the representative seed-45 interval analysis, test-period 95% prediction intervals approached nominal coverage, whereas future-holdout coverage decreased substantially, revealing unresolved uncertainty under later-period conditions. These results indicate that dynamic ensembling can provide a practical and repeatable groundwater-level forecasting workflow, while future-holdout diagnostics remain necessary to identify extrapolation limits.

中文说明：这个摘要没有写“最先进”“显著优于所有模型”，而是把强证据放在 test-period repeatability，把边界放在 future holdout。这个口径稳。

## 7. Introduction故事

### Paragraph 1：地下水预测为什么重要

目的：建立 WRR 读者关心的水资源问题，而不是一上来讲深度学习。

可写内容：

> 地下水水位是含水层状态和水资源管理的重要指标。可靠预测可以支持供水调度、干旱风险识别、地下水开发约束和生态保护。但实际应用中，模型不仅要在历史测试期准确，还要在井间差异、非平稳水文条件和未来时期中保持可信。

English opening:

> Groundwater levels integrate climatic forcing, aquifer properties and human water use, making their forecasts important for water-resources management. For such forecasts to be useful, however, model evaluation must move beyond single-run accuracy and address repeatability and uncertainty under changing conditions.

### Paragraph 2：现有数据驱动模型的瓶颈

目的：把 LSTM/Transformer/TCN 放进已有方法背景，同时指出单模型和单 seed 比较的不足。

可写内容：

> LSTM、Transformer 和 TCN 等深度学习模型已被广泛用于时间序列预测，但它们捕捉时间依赖的方式不同，对训练初始化和数据切分的敏感性也不同。若只报告单次运行的最优模型，很容易把偶然 seed 或某一时段优势误写成稳定结论。

English sentence:

> Deep temporal models capture different aspects of groundwater-level dynamics, but single-run comparisons can confound architectural advantage with random-seed sensitivity.

### Paragraph 3：本文的缺口

目的：定义本文真正填补的空白。

推荐缺口：

> 现有地下水水位预测研究常强调模型精度提升，但较少同时评估随机 seed 可重复性、预测区间覆盖、峰值事件行为和解释性诊断。对于 WRR，缺口不应只写成“缺少一个新模型”，而应写成“缺少一个可重复、可诊断、能暴露外推边界的预测评估流程”。

English gap sentence:

> What remains less clear is whether an ensemble forecasting workflow can provide repeatable gains while also exposing uncertainty and failure modes relevant to groundwater applications.

### Paragraph 4：本文做什么

目的：清楚交代贡献，但不堆太多结果数字。

English contribution paragraph:

> Here we develop and evaluate a repeatability-aware dynamic ensemble workflow for groundwater-level forecasting. The workflow combines LSTM, Transformer and TCN base models through DynamicGatedStacking and evaluates performance across 11 random seeds. We further use WCI/CWC interval diagnostics, peak-event analysis and SHAP explainability to examine uncertainty, hydrologically relevant event behavior and model transparency. The study asks not only which model performs best on average, but also when the forecasting workflow remains reliable and where its extrapolation limits become visible.

## 8. Methods故事

Methods 不能写成代码流水账。建议结构如下：

### 8.1 Data and forecasting task

任务：说明 `combo4_cand004` 是本文使用的数据支持，包含 15 口地下水井的周尺度水位序列和含水层类型信息。正文不展开候选筛选过程。

需要补的信息：

- 研究区位置和水文地质背景。
- 每口井时间跨度、周尺度聚合方法、缺失值处理。
- 输入特征到底包含哪些变量和滞后项。
- train/selection/test/future holdout 的时间划分依据。

### 8.2 Base learners

任务：说明 LSTM、Transformer、TCN 分别作为时间序列基线。每个模型只写与地下水时间序列预测有关的角色，不需要教科书式介绍。

写作口径：

> The three base learners were selected to represent recurrent, attention-based and convolutional temporal modeling strategies.

### 8.3 DynamicGatedStacking

任务：说明动态门控集成如何组合三个基模型输出。这里要让读者知道它不是简单平均，而是根据样本/状态动态调整权重。

需要补的信息：

- gating 网络输入是什么。
- 输出权重是否约束为非负且和为 1。
- 训练目标和基模型训练顺序。
- 是否有防止信息泄漏的 stacking 流程。

Methods 可用句：

> DynamicGatedStacking was designed to combine complementary temporal learners by assigning data-dependent weights to their predictions, allowing the ensemble to adapt across wells and time periods.

### 8.4 Repeatability-aware evaluation

任务：说明为什么跑 seeds 40-50。这里是论文的主方法亮点。

可用句：

> To distinguish stable model behavior from seed-specific outcomes, we repeated the final evaluation across 11 random seeds and summarized mean NRMSE, rank stability and best-count frequency for each model.

### 8.5 Interval, peak and SHAP diagnostics

任务：把三类诊断放在“代表性 seed”框架下。

写作口径：

> Interval prediction, peak-event diagnostics and SHAP analyses were conducted for seed 45 as representative diagnostic analyses. They were used to interpret uncertainty, event behavior and model transparency, rather than to claim multi-seed repeatability for these secondary analyses.

## 9. Results故事

Results 建议按证据强度排序，不按代码目录排序。

### Result 1：多seed点预测显示测试期可重复优势

主结论：

> DynamicGatedStacking 在测试期点预测中具有最稳的平均优势。

证据：

- test mean NRMSE = 0.070388 +/- 0.000873。
- mean rank = 1.18。
- best count = 9/11。

英文段落：

> The repeatability analysis identified DynamicGatedStacking as the most stable model on the test period. Across seeds 40-50, the ensemble achieved a mean NRMSE of 0.0704 +/- 0.0009, compared with 0.0715 +/- 0.0012 for Transformer, 0.0781 +/- 0.0009 for LSTM and 0.0833 +/- 0.0018 for TCN. It also ranked first in 9 of 11 seeds, whereas Transformer ranked first in 2 seeds and the other individual models did not rank first. These results indicate that the ensemble advantage was not driven by a single favorable initialization.

### Result 2：selection支持但优势较接近

主结论：

> selection split 中 DynamicGatedStacking 和 Transformer 很接近，这说明不能夸大“全阶段压倒性优势”。

证据：

- DynamicGatedStacking: 0.059545 +/- 0.000477, best count = 6/11。
- Transformer: 0.059708 +/- 0.000778, best count = 5/11。

英文句：

> On the selection split, DynamicGatedStacking and Transformer showed closely matched performance, indicating that the ensemble advantage was strongest and most repeatable on the test period rather than uniformly dominant across all splits.

### Result 3：future holdout暴露外推边界

主结论：

> future holdout 中 DynamicGatedStacking 平均 NRMSE 最低，但 LSTM best count 更高；长期外推仍然困难。

证据：

- DynamicGatedStacking mean NRMSE = 0.167971，best count = 3/11。
- LSTM mean NRMSE = 0.169122，best count = 6/11。

英文段落：

> In the future holdout period, DynamicGatedStacking retained the lowest mean NRMSE, but it did not rank first in most seeds. LSTM ranked first in 6 of 11 seeds, compared with 3 of 11 for the ensemble. This result suggests that the dynamic ensemble remained competitive under later-period conditions, but the future holdout did not provide the same clear repeatability advantage observed on the test period.

### Result 4：WCI/CWC显示测试期区间接近标称覆盖，但future holdout欠覆盖

主结论：

> seed45 的 WCI 区间在测试期接近 95% 覆盖，future holdout 覆盖率明显下降。

关键数据：

| split | model | PICP95 | PINAW95 | CWC95 |
|---|---:|---:|---:|---:|
| test | DynamicGatedStacking | 0.954930 | 0.428172 | 0.550712 |
| test | Transformer | 0.958685 | 0.415417 | 0.522739 |
| future holdout | DynamicGatedStacking | 0.675641 | 0.443962 | 0.959975 |
| future holdout | LSTM | 0.692308 | 0.468612 | 0.926874 |

英文段落：

> In the representative seed-45 analysis, WCI-calibrated intervals achieved near-nominal 95% coverage on the test period. For DynamicGatedStacking, PICP95 reached 0.955 with a PINAW95 of 0.428. In contrast, future-holdout coverage decreased to 0.676 for the ensemble, despite comparable interval width. This under-coverage indicates that interval calibration based on historical behavior did not fully absorb later-period distributional change.

### Result 5：峰值和SHAP作为诊断，不作为主胜利证据

主结论：

> 峰值预测和 SHAP 增强了水文事件行为和模型透明度的讨论，但它们是 seed45 代表性诊断，不是多 seed 主证据。

英文句：

> Peak-event diagnostics and SHAP analyses were used to inspect hydrologically relevant short-term behavior and model transparency for a representative seed. These analyses complement the repeatability results but should not be interpreted as multi-seed validation.

## 10. Discussion故事

Discussion 不要重复 Results，而要解释“为什么这些结果对 WRR 有意义”。

### 10.1 主要解释：动态集成提高了测试期稳定性

可以解释为：

> LSTM、Transformer 和 TCN 对时间依赖的表达方式不同，DynamicGatedStacking 利用它们的互补性，在测试期降低了对单一模型结构和随机初始化的依赖。

英文句：

> The repeatable test-period advantage suggests that dynamic ensembling reduced dependence on any single temporal representation and made model selection less sensitive to random initialization.

### 10.2 对水文预测的意义

WRR 读者会关心：这个模型对水资源问题有什么用？

建议写：

> 这套流程能为地下水预测提供一个更稳的模型选择依据，而不是只给出一次训练的漂亮曲线。多 seed 可重复性、区间覆盖、峰值事件和解释性诊断共同构成了更接近实际管理需求的评估方式。

英文句：

> For groundwater applications, the main value of the workflow is not only improved point accuracy, but a more explicit accounting of repeatability and uncertainty in model evaluation.

### 10.3 竞争解释和边界

必须正面写：

> future holdout 没有呈现与测试期同等清晰的集成优势；区间覆盖也明显不足。这可能反映了后期地下水动态与校准期存在分布差异，或输入变量没有充分捕捉后期驱动条件。

英文句：

> The weaker future-holdout ranking and reduced interval coverage suggest that later-period groundwater dynamics differed from the conditions represented during calibration, or that the available predictors did not fully encode the drivers of future change.

### 10.4 SHAP的解释边界

必须写清楚：

> SHAP 说明模型预测依赖哪些输入/滞后特征，但不能证明地下水变化的因果机制。

英文句：

> SHAP analysis improves transparency by identifying influential inputs and lagged states, but it does not establish causal hydrologic mechanisms.

### 10.5 Practical implication

可以写：

> 对类似地下水预测任务，建议用 DynamicGatedStacking 作为默认主集成方案，但保留 future holdout、区间覆盖和代表性诊断作为模型部署前的边界检查。

英文句：

> A practical implication is that dynamic ensemble forecasts should be paired with future-period and interval diagnostics before being used for long-horizon groundwater decision support.

## 11. 图表故事线

| Figure/Table | 主问题 | 建议内容 | 数据来源 |
|---|---|---|---|
| Figure 1 | 研究对象和流程是什么？ | 研究区/井类型 + 数据划分 + DynamicGatedStacking流程图 | `data/`, `code/learn.py` |
| Figure 2 | 主模型是否可重复地更好？ | seeds 40-50 的 test/selection/future holdout NRMSE 和排名 | `results/01_repeatability_multiseed/` |
| Table 1 | 主结果数字是什么？ | mean NRMSE、std、mean rank、best count | `cand004_multiseed_model_summary.csv`, `cand004_multiseed_rank_summary.csv` |
| Figure 3 | 不确定性在哪里失效？ | test vs future holdout 的 PICP95/PINAW95/CWC95 | `results/03_interval_prediction_wci_cwc_seed45/` |
| Figure 4 | 峰值事件是否被捕捉？ | 三类含水层代表井峰值图 | `results/04_peak_prediction_seed45/representative_examples/` |
| Supplementary Fig. S1-S15 | 所有井峰值表现 | 15口井全图 | `results/04_peak_prediction_seed45/all_wells/` |
| Supplementary Fig. S16-S30 | SHAP解释性 | 15口井全图 | `results/05_shap_explainability_seed45/all_wells/` |
| Supplementary Table | 可重复性补充诊断 | pairwise dominance, risk distribution | `results/06_repeatability_diagnostics_optional/` |

如果主文篇幅紧，Figure 4 和 SHAP 可以放补充材料，但 Discussion 仍然要引用它们的诊断意义。

## 12. 结论-证据映射

| Claim | Evidence | Status |
|---|---|---|
| DynamicGatedStacking 在测试期点预测中具有可重复平均优势。 | 11 seeds, test mean NRMSE最低，mean rank=1.18，best count=9/11。 | supported |
| DynamicGatedStacking 是当前最方便、最稳的主集成选择。 | 测试期多 seed 结果 + 代码和结果已整理成展示包。 | supported, limited to current dataset/workflow |
| DynamicGatedStacking 解决了长期外推问题。 | future holdout best count 只有3/11，区间欠覆盖。 | not supported |
| WCI 区间在测试期接近95%覆盖。 | seed45 test PICP95 约0.94-0.96。 | supported for representative seed |
| future holdout存在明显欠覆盖。 | seed45 future holdout PICP95 约0.63-0.69。 | supported for representative seed |
| 峰值预测支持水文事件行为诊断。 | seed45 peak metrics and plots。 | supported as diagnostic |
| SHAP揭示地下水变化因果机制。 | 当前只有模型解释性图。 | not supported |

## 13. 不能写和应该替换的表达

| 不建议写 | 替换成 |
|---|---|
| The proposed model significantly outperformed all baselines in all conditions. | The ensemble showed the most repeatable test-period advantage across random seeds. |
| The model solved long-term groundwater forecasting. | Future-holdout diagnostics revealed remaining extrapolation uncertainty. |
| SHAP revealed the hydrological mechanism. | SHAP provided model-level interpretability support. |
| WCI is an evaluation metric. | WCI is an interval calibration method; CWC95 is an interval evaluation metric. |
| The selected candidate won the screening. | The study used the selected `combo4_cand004` dataset for model evaluation. |
| The results are broadly replicable. | The results are repeatable across 11 random seeds. |

## 14. 结论草稿

**English conclusion draft**

This study evaluated a repeatability-aware dynamic ensemble workflow for groundwater-level forecasting. Across 11 random seeds, DynamicGatedStacking provided the lowest mean test-period NRMSE and the most stable ranking among LSTM, Transformer and TCN-based alternatives, supporting its use as a practical ensemble choice for the selected groundwater-level dataset. Representative WCI/CWC interval diagnostics, peak-event evaluation and SHAP analysis further showed how the workflow can be examined for uncertainty, event behavior and transparency. The future-holdout results also exposed an important boundary: dynamic ensembling improved test-period robustness but did not remove the challenge of extrapolating groundwater dynamics under later-period conditions. These findings support repeatability-aware and uncertainty-aware evaluation as a necessary part of data-driven groundwater forecasting.

**中文意思**

本文最稳的结论是：DynamicGatedStacking 可以作为当前数据集上的主集成模型，因为它在测试期多 seed 点预测中最稳；但这不是长期外推问题的终点。WRR 叙事要把“可重复优势”和“外推边界”同时写出来，这样论文更诚实，也更像水资源研究而不是单纯模型展示。

## 15. 下一步写作任务

1. 补研究区和数据背景：研究区位置、含水层类型、井的时间跨度、周尺度处理、缺失值处理。
2. 明确输入特征：模型到底用了哪些变量、滞后项和预测步长。
3. 画 Figure 1：研究区/井类型 + 数据划分 + DynamicGatedStacking框架。
4. 把 `results/01_repeatability_multiseed/` 整理成主文 Table 1 和 Figure 2。
5. 从 `results/04_peak_prediction_seed45/all_wells/` 和 `results/05_shap_explainability_seed45/all_wells/` 中挑选补充材料图。
6. 写 Methods 的防泄漏说明：stacking 和 interval calibration 必须说清楚训练、selection、test、future holdout 的信息边界。
7. 正式写 Introduction 前，先确定文献框架：地下水水位预测、深度学习时序模型、集成模型、不确定性/ conformal prediction、可解释性。

## 16. 最短论文路线

如果要最快推进，按这个顺序写：

1. Results：先写多 seed 点预测，再写 interval/future holdout，再写 peak/SHAP。
2. Figure/Table：先做 Figure 2 和 Table 1，因为它们支撑主结论。
3. Methods：把数据划分、DynamicGatedStacking 和多 seed evaluation 写清楚。
4. Introduction：围绕“地下水预测需要可重复和不确定性诊断”搭框架。
5. Discussion：重点写测试期优势、future holdout边界、SHAP非因果解释。
6. Abstract/Title：最后压缩。

这条路线比从 Introduction 开始写更稳，因为现在最强的是结果证据，故事应该从证据往外长。
