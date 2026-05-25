# test 目录说明

`test/` 是当前 15 口井实验的独立工作区。后续和 15 口井相关的代码、数据、输出和日志都放在这里。

## 代码

| 文件 | 作用 |
| --- | --- |
| `learn.py` | 核心训练和预测代码，包含 LSTM、Transformer、TCN、DynamicGatedStacking。 |
| `prepare_15wells.py` | 从原始地下水和气象数据中准备 15 口井周尺度数据。 |
| `optimize_15wells_dynamic_gate.py` | 探索性筛选脚本，负责换井、seed 尝试、失败输出清理和 confirm run。 |
| `run_15wells_test_focus.py` | 点预测入口，只关注 `test` RMSE/NSE，强制使用 GPU。 |
| `run_15wells_interval_focus.py` | 区间预测入口，强制使用 GPU。 |
| `run_15wells_shap_focus.py` | SHAP 分析入口，强制使用 GPU。 |
| `run_15wells_peak_focus.py` | 峰值分析入口，强制使用 GPU。 |
| `run_15wells_lookback_sweep.py` | 15 口井 lookback 超参数实验入口，强制使用 GPU。 |
| `reproducible_model_selection.py` | 基于 R-distribution 和成对概率的可重复性分析。 |
| `validate_selected_weekly_data.py` | 检查当前 15 口井数据是否完整、字段是否正确。 |
| `test_learn_data_flow.py` | `learn.py` 的数据流和模型逻辑单元测试。 |
| `test_reproducible_model_selection.py` | 可重复性分析脚本的单元测试。 |
| `vendor_bootstrap.py` | 可选本地依赖路径初始化。 |

## 数据和输出

| 路径 | 作用 |
| --- | --- |
| `selected_weekly_data_15wells_current/` | 当前固定 15 口井周尺度数据。 |
| `outputs_15wells_interval_focus/` | 当前保留的 15 口井区间预测输出。 |
| `outputs_15wells_interval_focus/reproducible_selection/` | 可重复性分析输出。 |
| `RUN_LOG.md` | 人工可读实验日志。 |
| `attempts_summary.csv` | 所有筛选 attempt 的完整机器可读摘要。 |
| `confirm_summary.csv` | 固定 15 口井后的 confirm seed 结果。 |
| `confirm_model_summary.csv` | confirm run 的模型平均表现。 |

## 推荐命令

```powershell
python test\validate_selected_weekly_data.py
python -m unittest discover -s test -p "test_*.py"
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_test_focus.py
C:\Users\xf-99\.conda\envs\Python39\python.exe test\run_15wells_interval_focus.py
```
