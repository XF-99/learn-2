# 更新日志

## 2026-05-25 - 整理为 15 口井主线

### 结构整理

- 根目录只保留项目级说明文件：`README.md`、`CHANGELOG.md`、`.gitignore`。
- 15 口井实验相关代码、数据、输出和测试全部集中到 `test/` 目录。
- 删除根目录重复的 `learn.py`、`vendor_bootstrap.py`。
- 将可重复性分析、数据校验和单元测试脚本移动到 `test/`。
- `RUN_LOG.md`、README 和可重复性报告改为中文说明。

### 当前实验主线

- 当前主线是 15 口井 DynamicGatedStacking 探索性筛选实验。
- 三类含水层各 5 口井：裂隙水 5 口、岩溶水 5 口、孔隙水 5 口。
- 当前结论只关注 `test` split；`future_holdout` 不参与模型选择判断。
- 保留输出模型：`LSTM`、`Transformer`、`TCN`、`DynamicGatedStacking`。
- 不再输出：`Persistence`、`Stacking`、`DynamicGatedOnly`、`AdaptiveWeightedStacking`。

### 当前 15 口井 test 平均结果

| 模型 | RMSE | NSE | PICP95 | MPIW95 |
| --- | ---: | ---: | ---: | ---: |
| DynamicGatedStacking | 0.1853 | 0.9210 | 0.9518 | 0.7622 |
| Transformer | 0.1891 | 0.9195 | 0.9600 | 0.8173 |
| LSTM | 0.1927 | 0.9127 | 0.9496 | 0.8098 |
| TCN | 0.2078 | 0.9023 | 0.9562 | 0.8466 |

### 可重复性分析

- `test/reproducible_model_selection.py` 已升级为 test-only、自动识别模型列、输出 loss quantile、R-distribution 和成对可重复优势概率。
- 当前分析显示 DynamicGatedStacking 平均 empirical risk 最低，但对 Transformer 的优势不稳定，所以结论只写作探索性筛选优势。

### 验证命令

```powershell
python -m py_compile test\learn.py test\reproducible_model_selection.py test\validate_selected_weekly_data.py test\prepare_15wells.py test\run_15wells_test_focus.py test\run_15wells_interval_focus.py test\optimize_15wells_dynamic_gate.py
python test\validate_selected_weekly_data.py
python -m unittest discover -s test -p "test_*.py"
```
