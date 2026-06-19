# Groundwater Level Forecasting With DynamicGatedStacking

这个仓库是 `combo4_cand004` 实验的 GitHub 展示版，保留了论文写作和结果复核需要的核心代码、输入数据和主要结果。旧的筛选过程、中间日志、缓存、临时并行目录和过大的原始工作区文件不再放入仓库。

当前最稳的论文口径是：

> DynamicGatedStacking 在多 seed 测试期点预测中表现出最强且最稳定的平均优势；WCI/CWC 区间预测、峰值预测和 SHAP 分析作为代表性 seed 的补充诊断，用于说明不确定性、峰值事件捕捉能力和模型解释性。

## 快速入口

- 中文详细说明：[README_zh.md](README_zh.md)
- English package guide: [README_en.md](README_en.md)
- WRR 论文故事线草案：[docs/WRR_manuscript_story_zh.md](docs/WRR_manuscript_story_zh.md)
- WRR 投稿叙事与证据边界：[docs/WRR_story_and_evidence_zh.md](docs/WRR_story_and_evidence_zh.md)
- 结果目录说明：[results/README_zh.md](results/README_zh.md)

## 仓库结构

```text
.
├── code/       # 训练、评估、多 seed、WCI/CWC 和汇总脚本
├── data/       # combo4_cand004 的 15 口井周尺度输入数据
├── docs/       # WRR 叙事、证据边界和写作说明
├── results/    # 多 seed 主结果和 seed45 代表性诊断结果
├── README.md
├── README_zh.md
└── README_en.md
```

## 主要证据

### 1. 多 seed 点预测

主证据位于：

```text
results/01_repeatability_multiseed/
```

重点文件：

- `cand004_multiseed_model_summary.csv`
- `cand004_multiseed_rank_summary.csv`

测试期 seeds 40-50 的核心结果：

| Model | Test mean NRMSE | Std | Mean rank | Best count |
|---|---:|---:|---:|---:|
| DynamicGatedStacking | 0.070388 | 0.000873 | 1.18 | 9/11 |
| Transformer | 0.071492 | 0.001190 | 1.82 | 2/11 |
| LSTM | 0.078062 | 0.000917 | 3.00 | 0/11 |
| TCN | 0.083291 | 0.001772 | 4.00 | 0/11 |

这部分支持“DynamicGatedStacking 在测试期点预测中具有可重复的平均优势”。

### 2. 区间预测和不确定性诊断

代表性 seed45 结果位于：

```text
results/03_interval_prediction_wci_cwc_seed45/
```

WCI 是区间构造/校准方法，CWC95 是区间评价指标。seed45 结果显示测试期 PICP95 接近 95%，但 future holdout 覆盖率明显下降，因此更适合写作“不确定性诊断揭示长期外推边界”，不要写成长期预测已经被解决。

### 3. 峰值预测和 SHAP

代表性 seed45 诊断结果位于：

```text
results/04_peak_prediction_seed45/
results/05_shap_explainability_seed45/
```

每个目录中：

- `representative_examples/` 保留三类含水层各一个代表井，方便快速浏览。
- `all_wells/` 保留 15 口井全部图件，方便后续挑选主文或补充材料图。

峰值预测和 SHAP 当前是代表性诊断，不是多 seed 可重复性主证据。

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

## 结果使用边界

可以写：

- DynamicGatedStacking 是当前最方便、最稳的主集成模型选择。
- 测试期点预测具有多 seed 可重复优势。
- 区间预测揭示了 future holdout 外推不确定性。
- 峰值预测和 SHAP 增强了结果解释和水文诊断。

不要写：

- DynamicGatedStacking 在所有时间段和所有指标上都显著最优。
- 模型已经解决非平稳地下水长期外推。
- SHAP 证明了地下水变化的因果机制。
- 区间、峰值和 SHAP 已经完成多 seed 验证。
