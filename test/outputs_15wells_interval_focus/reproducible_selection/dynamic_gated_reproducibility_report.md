# DynamicGatedStacking 可重复性分析报告

本报告是受论文思想启发的可重复模型选择分析。这里使用 `test` 集逐点 loss、block bootstrap 得到的 R-distribution，以及成对优势概率 `P(R_A < R_B)` 判断模型优势是否稳定。

这不是完整的 hierarchical beta process 复刻，因为当前的 LSTM、Transformer、TCN 和 DynamicGatedStacking 是预测模型，不是严格的生成式概率模型。

## split: test

- 平均 empirical risk 最低的模型：`DynamicGatedStacking`。
- `DynamicGatedStacking` 平均 empirical risk：0.0667454。
- `DynamicGatedStacking` 平均成对胜率：0.688。
- 可重复优势阈值：0.95。
- 趋势优势阈值：0.70。

## 成对比较

- `P(R_DynamicGatedStacking < R_LSTM)` 的 15 个场景平均值为 0.748。说明 DGS 平均更好，但不足以按 0.95 阈值稳定拒绝 LSTM；其中 6 个井级场景达到可重复优势。
- `P(R_DynamicGatedStacking < R_TCN)` 的 15 个场景平均值为 0.830。说明 DGS 平均更好，但不足以按 0.95 阈值稳定拒绝 TCN；其中 10 个井级场景达到可重复优势。
- `P(R_DynamicGatedStacking < R_Transformer)` 的 15 个场景平均值为 0.486。说明 DGS 对 Transformer 的优势不稳定；其中 4 个井级场景达到可重复优势。

## 结论

DynamicGatedStacking 在固定 15 口井 `test` 集上的平均风险最低，但从 R-distribution 成对概率看，它对 Transformer 的优势不稳定。因此当前只能表述为探索性筛选中整体平均表现最优，不能表述为稳定显著优于所有单模型。
