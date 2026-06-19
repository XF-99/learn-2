# combo4_cand004 展示包说明

这个目录是 `combo4_cand004` 实验的精简展示包，用来给其他人查看代码、输入数据和主要结果。这里保留复核实验所需的核心内容，同时去掉日志、缓存、备份文件、超参数搜索中间结果和并行临时目录。

如果要面向 WRR 投稿组织论文叙事和证据边界，先看：

- `docs/WRR_manuscript_story_zh.md`
- `docs/WRR_story_and_evidence_zh.md`

## 证据层级和目录位置

- `code/`
  - 主训练、评估、多 seed、WCI/CWC 区间预测和结果汇总脚本。

- `data/candidate_group_cand004/`
  - 15 个入选地下水井的周尺度水位时间序列 CSV 文件。
  - `selected_wells_summary.csv` 记录井列表和含水层类型标签。

- `results/01_repeatability_multiseed/`
  - 这是最核心的“可重复性”证据。
  - 包含 seeds 40-50 的 final 点预测结果。
  - 重点看 `cand004_multiseed_model_summary.csv` 和 `cand004_multiseed_rank_summary.csv`。

- `results/02_point_prediction_seed45/`
  - seed 45 的代表性点预测结果。
  - 用于查看单次完整运行的结果，不作为多 seed 可重复性主证据。

- `results/03_interval_prediction_wci_cwc_seed45/`
  - seed 45 的代表性区间预测结果。
  - 包含 WCI 区间校准字段，以及 `PICP95`、`MPIW95`、`PINAW95`、`CWC95` 等区间评价指标。
  - 这是不确定性诊断，不是多 seed 可重复性主证据。

- `results/04_peak_prediction_seed45/`
  - seed 45 的代表性峰值预测诊断。
  - 包含 `peak_metrics_summary.csv`。
  - `representative_examples/` 里为每类含水层放了一个代表井，方便快速浏览。
  - `all_wells/` 里放了 15 个井的全部峰值指标和峰值图，方便后面挑选论文图。
  - 这是水文事件捕捉能力分析，不是多 seed 可重复性主证据。

- `results/05_shap_explainability_seed45/`
  - seed 45 的代表性 SHAP/解释性分析。
  - `representative_examples/` 里为每类含水层放了一个代表井，方便快速浏览。
  - `all_wells/` 里放了 15 个井的全部 SHAP/解释性图，方便后面挑选论文图。
  - 这是模型解释性诊断，不是多 seed 可重复性主证据。

- `results/06_repeatability_diagnostics_optional/`
  - 更丰富的可重复性诊断结果。
  - 为了控制展示包体积，未放入很大的 raw sample 表。

- `results/results_overview.csv`
  - 精简总览表，把多 seed 点预测主证据和 seed 45 代表性诊断结果放在一起。

## 主要结论口径

可以把 `DynamicGatedStacking` 作为主集成模型，因为它在多 seed 测试期点预测中表现最稳定、平均误差最低，并且在 seeds 40-50 中多数 seed 排名第一。

但要注意：区间预测、峰值预测和 SHAP 当前是 seed 45 的代表性诊断，不应说成已经完成多 seed 可重复性验证。更稳妥的论文口径是：

> DynamicGatedStacking 在多 seed 测试期点预测中表现出最强且最稳定的平均优势；WCI/CWC 区间预测、峰值预测和 SHAP 分析作为代表性 seed 的补充诊断，用于说明不确定性、峰值事件捕捉能力和模型解释性。

## 如何复跑主要实验

这些脚本是在 Linux + CUDA 环境下运行的。如果把这个目录移动到其他机器，shell 脚本中的路径可能需要根据新位置调整。

多 seed final 点预测实验：

```bash
bash code/run_cand004_multiseed_final.sh
```

代表性 seed 45 的区间预测/WCI 实验：

```bash
bash code/launch_interval_wci_seed45_parallel10.sh
```

## 不包含的内容

这个目录不是完整工作目录，而是给别人查看的干净展示包。以下内容被有意排除：

- `logs/`
- `__pycache__/`
- `.bak` 备份文件
- 超参数搜索输出
- 完整 per-well 图件目录
- `_well_parallel_tmp/` 并行临时目录

完整原始工作目录仍保留在：

```text
C:\Users\xf-99\Desktop\learn-2\yun\combo4_cand004
```
