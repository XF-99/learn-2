# 更新记录

## 2026-04-29 - 每周数据与主结果重新生成

本次使用新的三份每周数据集替换仓库中的原始数据，并重新运行 `learn.py` 生成主实验结果。旧的 `outputs/` 结果已由本次运行结果覆盖，旧的 `outputs_dropout/` dropout 扫描结果已删除，避免继续保留与当前数据集不一致的历史输出。

### 更新内容

- 替换三份每周数据 CSV：孔隙水、裂隙水、岩溶水。
- 重新同步 `outputs/` 下的主实验结果、预测文件、未来预测、峰值分析图表和解释性分析图表。
- 删除旧的 `outputs_dropout/` 目录及其 dropout 扫描结果，因为本次只重新运行了主实验。

### 新结果概览

本次主实验输出包含 3 类水文数据、4 类模型的指标汇总与图表结果。`outputs/metrics_summary.csv`、`outputs/metrics_summary.json` 和 `outputs/peak_metrics_summary.csv` 已随本次结果一起更新，后续分析应以本次 2026-04-29 重新运行生成的 `outputs/` 为准。

## 2026-04-29 - Dropout 实验结果更新

本次使用新的三口代表井每周数据重新运行了 dropout 参数敏感性实验，并用新生成的 `outputs_dropout/` 整体替换了仓库中的旧结果。

### 更新内容

- 删除旧的 `outputs_dropout/` 结果目录后，重新同步本次运行生成的完整 dropout 结果。
- 覆盖更新 `dropout_0.0` 到 `dropout_0.5` 各组实验输出、逐井预测结果、对比图和汇总表。
- 更新 `dropout_sweep_metrics.csv`、`dropout_sweep_summary.csv`、`dropout_rmse_comparison.png`、`dropout_nse_comparison.png`。

### 新结果概览

新的汇总结果显示，不同模型的最佳平均 RMSE 分布在不同 dropout 设置上：Transformer 在 `dropout=0.3` 下最低，LSTM 和 Stacking 在 `dropout=0.4` 下表现较好，TCN 在 `dropout=0.3` 下表现较好。后续分析应以本次 2026-04-29 重新运行的输出为准。

## 2026-04-19 - Dropout 泛化性能实验

本次更新新增了一个独立的 dropout 参数敏感性实验，并上传了已经跑完的实验结果，方便之后检查不同 dropout 取值对模型泛化性能的影响。

### 新增内容

- `dropout_experiment.py`：独立实验脚本，不修改原来的 `learn.py`，运行时在内存中调整 LSTM、Transformer、TCN 的 dropout 取值。
- `outputs_dropout/`：dropout 取值 `0.0`、`0.1`、`0.2`、`0.3`、`0.4`、`0.5` 的完整实验输出。
- `outputs_dropout/dropout_sweep_metrics.csv`：每个 dropout、每口井、每个模型的详细指标结果。
- `outputs_dropout/dropout_sweep_summary.csv`：按 dropout 和模型聚合后的均值、标准差结果。
- `outputs_dropout/dropout_rmse_comparison.png`：不同 dropout 下各模型 RMSE 对比图。
- `outputs_dropout/dropout_nse_comparison.png`：不同 dropout 下各模型 NSE 对比图。

### 主要结论

三口井的平均结果显示，`dropout=0.0` 在 LSTM、Transformer 和 Stacking 上取得了最好的平均泛化表现。TCN 略有不同：`dropout=0.1` 的平均 RMSE 最低，`dropout=0.2` 的平均 NSE 最高。整体来看，在当前数据规模和模型设置下，较强的 dropout 没有提升泛化性能，反而多数情况下会降低模型表现。

### 对应提交

- `ec68052 Add dropout sweep experiment results`：上传 dropout 实验脚本和实验结果。
- `134dfad Document dropout experiment update`：首次添加英文版更新说明。


