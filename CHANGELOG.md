# 更新记录

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
