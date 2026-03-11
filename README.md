# 数据最晚的3口井 - 合并的每周数据

## 选定的3口井


### 井 1: 岩溶水
- **所属州**: BB
- **时间范围**: 2000-01-03 至 2018-06-04
- **数据点数**: 962 周
- **时间跨度**: 6727 天 (18.4 年)

### 井 2: 孔隙水
- **所属州**: SH
- **时间范围**: 2000-01-03 至 2018-06-04
- **数据点数**: 962 周
- **时间跨度**: 6727 天 (18.4 年)

### 井 3: 裂隙水
- **所属州**: BB
- **时间范围**: 2000-01-03 至 2018-05-21
- **数据点数**: 960 周
- **时间跨度**: 6713 天 (18.4 年)

## 数据文件

每口井一个CSV文件,包含5列:

- **Date**: 地下水位采样日期
- **TASMAX**: 该周最高温度平均值(°C)
- **TAS**: 该周平均温度平均值(°C)
- **Precipitation**: 该周降水量总和(mm)
- **GWL**: 地下水位(m)

## 文件列表

- `岩溶水_每周数据.csv`
- `孔隙水_每周数据.csv`
- `裂隙水_每周数据.csv`

## 数据说明

### 聚合方法
- **温度**: 每周7天的加权平均(算术平均)
- **降水**: 每周7天的总和
- **地下水位**: 每周采样值

### 时间对齐
以地下水位采样日期为基准,气象数据为该日期往前7天的聚合值。

### 数据完整性
- 所有数据无缺失值
- 时间序列连续
- 每周包含完整的7天气象数据

## 使用示例

```python
import pandas as pd

# 读取数据
df = pd.read_csv('岩溶水_每周数据.csv')
df['Date'] = pd.to_datetime(df['Date'])

# 查看数据
print(df.head())
print(df.info())

# 特征和目标
X = df[['TASMAX', 'TAS', 'Precipitation']]
y = df['GWL']
```

## 数据来源

- **地下水数据**: GEMS-GER项目
- **气象数据**: HYRAS v6.0, 德国气象局(DWD)
- **处理日期**: 2026年1月17日


## Weighted conformal intervals (Stacking only)

- The pipeline now uses a time-ordered split: `train/val/calib/test = 60/10/15/15`.
- Conformal scores are computed on the independent calibration block: `s_i = |y_i - yhat_i|`.
- Intervals are produced for both test and future forecasts at 90% and 95%.
- Reported interval metrics: `PICP90`, `PICP95`, `MPIW90`, `MPIW95`.
- Future rolling intervals are conditioned on the script's assumption about future exogenous feature handling (same assumption used by the point forecast rollout).
