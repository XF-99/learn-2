# 地下水位预测与可解释性分析（3口井周尺度数据）

本项目使用 `LSTM + Transformer + TCN + XGBoost Stacking` 进行地下水位（`GWL`）预测，并输出：
- 点预测结果
- 95% 置信区间（加权 conformal）
- 注意力可视化（默认开启）
- SHAP 可解释图（环境安装 `shap` 时自动开启）

## 1. 数据文件

目录下默认使用以下 3 个 CSV：
- `岩溶水_每周数据.csv`
- `孔隙水_每周数据.csv`
- `裂隙水_每周数据.csv`

字段约定：
- `Date`
- `GWL`
- `TASMAX`
- `TAS`
- `Precipitation`

## 2. 模型与流程

主脚本：`learn.py`

核心流程：
1. 按时间排序并构造滑动窗口序列（`lookback/horizon`）
2. 按时间切分 `train/val/calib/test`
3. 训练 LSTM、Transformer、TCN 三个基模型
4. 用 XGBoost 学习 stacking 残差修正
5. 使用 calibration 集做加权 conformal，输出 **95% 区间**
6. 输出测试集与未来滚动预测图/表
7. 解释输出：
   - attention：默认生成
   - SHAP：检测到 `shap` 后自动生成

## 3. 运行方式

默认运行（会生成预测图、95%区间、attention；若已安装 shap 则额外生成 SHAP）：

```bash
python learn.py
```

快速烟雾测试（低成本）：

```bash
python learn.py --epochs 1 --patience 1 --future_steps 2
```

关闭解释输出：

```bash
python learn.py --disable_explain
```

常用参数：
- `--lookback` 默认 `12`
- `--horizon` 默认 `1`
- `--train_ratio` 默认 `0.6`
- `--val_ratio` 默认 `0.1`
- `--calib_ratio` 默认 `0.15`
- `--future_steps` 默认 `30`
- `--out_dir` 默认 `outputs`

## 4. 输出目录结构

默认输出到 `outputs/`，每口井一个子目录：

```text
outputs/
  metrics_summary.csv
  metrics_summary.json
  rmse_comparison.png
  nse_comparison.png
  岩溶水/
    predictions.csv
    future_predictions.csv
    test_predictions.png
    stacking_predictions.png
    future_forecast.png
    explain/
      transformer_attention_heatmap_sample.png
      transformer_shap_beeswarm.png
      transformer_shap_heatmap_sample.png
      transformer_shap_waterfall_sample.png
      stacking_xgb_shap_beeswarm.png
      stacking_xgb_shap_waterfall_sample.png
  孔隙水/
  裂隙水/
```

说明：
- 预测 CSV 中仅保留 `PI95_Lower/PI95_Upper`
- 指标中仅保留 95% 区间相关指标（如 `PICP95`、`MPIW95`）

## 5. 环境依赖

基础依赖：
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torch`
- `xgboost`
- `scipy`（用于 STFT，可选）

可解释性增强：
- `shap`（安装后自动输出 SHAP 图）

安装示例：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch xgboost scipy shap
```

## 6. 常见问题

### Q1：图里中文显示成方框
代码已内置中文字体自动检测并设置。若系统缺少中文字体，请安装以下任一字体：
- Microsoft YaHei
- SimHei
- Noto Sans CJK SC

### Q2：为什么没有 SHAP 图
请确认运行环境安装了 `shap`，并且未使用 `--disable_explain`。

### Q3：为什么看到 `PermutationExplainer` 进度
这是 SHAP 解释过程的正常输出，不影响结果。

