# results 目录说明

这个目录按“论文证据层级”重新排版。

## 01_repeatability_multiseed

多 seed final 点预测结果，是当前最核心的可重复性证据。用于支持 DynamicGatedStacking 在测试期平均表现最强、排名最稳定。

## 02_point_prediction_seed45

seed 45 的单次 final 点预测结果。用于查看一个完整代表性运行，不作为多 seed 可重复性主证据。

## 03_interval_prediction_wci_cwc_seed45

seed 45 的区间预测结果。包含 WCI 区间校准信息和 CWC 区间评价指标。用于不确定性分析和外推边界讨论。

## 04_peak_prediction_seed45

seed 45 的峰值预测诊断。包含总体峰值指标、`representative_examples/` 中每类含水层一个代表井的峰值图，以及 `all_wells/` 中 15 个井的全部峰值指标和峰值图。

## 05_shap_explainability_seed45

seed 45 的 SHAP 解释性诊断。包含 `representative_examples/` 中每类含水层一个代表井的 SHAP 图，以及 `all_wells/` 中 15 个井的全部 SHAP/解释性图。

## 06_repeatability_diagnostics_optional

额外可重复性诊断文件。这里是补充材料性质，不是最核心的主表。

## results_overview.csv

精简总览表，方便快速查看多 seed 主结果和 seed 45 代表性诊断。
