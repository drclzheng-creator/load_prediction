# 日前模型特征维度与参数设置

## 数据与评估背景

- 数据集：`inputs/EWELD/eweld_industrial_u141_2018_2019_15min.csv`
- 任务：日前负荷预测
- 频率：15 分钟
- 单窗口预测长度：96 点
- 训练/测试切分：按时间顺序 7:3
- 离线回测：测试集按 96 点滚动窗口逐日预测
- 训练集监督样本数：48384 行

## 特征维度总览

| 模型 | 送入模型的主要数据形态 | 显式特征维度 | 说明 |
|---|---|---:|---|
| LightGBM | 项目内 `FeatureBuilder` 生成的监督学习矩阵 | 53 | 与 sklearn 使用同一套特征 |
| sklearn HistGradientBoosting | 项目内 `FeatureBuilder` 生成的监督学习矩阵 | 53 | 与 LightGBM 使用同一套特征 |
| AutoGluon RecursiveTabular GBM | `TimeSeriesDataFrame`，包含 target 和 known covariates | 7 个外部协变量 | AutoGluon 内部会继续构造递归时序特征，最终内部维度由 AutoGluon 管理 |

## LightGBM / sklearn 的 53 维特征

LightGBM 和 sklearn 当前使用完全相同的特征工程结果：

- 日历特征：9 维
- 周期时间特征：4 维
- item ID 特征：1 维
- 数值天气协变量：6 维
- `wind_direction` 类别编码：18 维
- lag 特征：9 维
- rolling mean/std 特征：6 维

合计：

```text
9 + 4 + 1 + 6 + 18 + 9 + 6 = 53
```

具体特征列表：

```text
feature__calendar__dayofweek
feature__calendar__is_weekend
feature__calendar__year
feature__calendar__month
feature__calendar__day
feature__calendar__quarter
feature__calendar__dayofyear
feature__calendar__hour
feature__calendar__minute
feature__time__sin_day
feature__time__cos_day
feature__time__sin_week
feature__time__cos_week
feature__item__U141
feature__cov__temperature_f
feature__cov__dew_point_f
feature__cov__humidity_pct
feature__cov__wind_direction__category__CALM
feature__cov__wind_direction__category__E
feature__cov__wind_direction__category__ENE
feature__cov__wind_direction__category__ESE
feature__cov__wind_direction__category__N
feature__cov__wind_direction__category__NE
feature__cov__wind_direction__category__NNE
feature__cov__wind_direction__category__NNW
feature__cov__wind_direction__category__NW
feature__cov__wind_direction__category__S
feature__cov__wind_direction__category__SE
feature__cov__wind_direction__category__SSE
feature__cov__wind_direction__category__SSW
feature__cov__wind_direction__category__SW
feature__cov__wind_direction__category__VAR
feature__cov__wind_direction__category__W
feature__cov__wind_direction__category__WNW
feature__cov__wind_direction__category__WSW
feature__cov__wind_speed_mph
feature__cov__wind_gust_mph
feature__cov__pressure_in
feature__lag__1
feature__lag__2
feature__lag__4
feature__lag__8
feature__lag__16
feature__lag__32
feature__lag__96
feature__lag__192
feature__lag__672
feature__rolling_mean__4
feature__rolling_std__4
feature__rolling_mean__16
feature__rolling_std__16
feature__rolling_mean__96
feature__rolling_std__96
```

## AutoGluon 输入特征

AutoGluon 接收的是长表时序数据，核心列包括：

- `item_id`
- `timestamp`
- `target`
- known covariates

当前传入 AutoGluon 的 known covariates 是 7 个：

| 特征 | AutoGluon 识别类型 |
|---|---|
| `temperature_f` | continuous |
| `dew_point_f` | continuous |
| `humidity_pct` | continuous |
| `wind_direction` | categorical |
| `wind_speed_mph` | continuous |
| `wind_gust_mph` | continuous |
| `pressure_in` | continuous |

AutoGluon 日志中的类型识别结果：

```text
categorical:        ['wind_direction']
continuous (float): ['temperature_f', 'dew_point_f', 'humidity_pct', 'wind_speed_mph', 'wind_gust_mph', 'pressure_in']
```

需要注意：AutoGluon 的 `RecursiveTabular` 会在内部基于 target、时间索引、known covariates 构造自己的递归时序特征，因此它的内部训练矩阵维度不等同于外部传入的 7 个协变量，也不等同于项目内 `FeatureBuilder` 的 53 维。

### AutoGluon 内部 RecursiveTabular 特征

通过读取保存的模型：

```text
outputs/autogluon/day_ahead/model/models/RecursiveTabular/W0/model.pkl
```

可以看到 AutoGluon 当前使用的是：

```text
MultiWindowBacktestingModel
  -> RecursiveTabularModel
    -> MLForecast
      -> AutoGluon Tabular LGBModel
```

最终内部 LGBModel 的特征数是 51 维：

- category：1 维
- float：50 维

内部 LGBModel 的具体 feature names：

```text
temperature_f
dew_point_f
humidity_pct
wind_speed_mph
pressure_in
__scaled_temperature_f
__scaled_dew_point_f
__scaled_humidity_pct
__scaled_wind_speed_mph
__scaled_pressure_in
lag1
lag2
lag3
lag4
lag5
lag6
lag7
lag8
lag9
lag10
lag11
lag12
lag13
lag14
lag95
lag96
lag97
lag191
lag192
lag193
lag287
lag288
lag289
lag383
lag384
lag385
lag479
lag480
lag481
lag575
lag576
lag577
lag671
lag672
lag673
minute_of_hour
hour_of_day
day_of_week
day_of_month
day_of_year
wind_direction
```

内部特征可以分成：

| 类型 | 数量 | 说明 |
|---|---:|---|
| 原始数值 known covariates | 5 | `temperature_f`、`dew_point_f`、`humidity_pct`、`wind_speed_mph`、`pressure_in` |
| 缩放后的数值 known covariates | 5 | `__scaled_*` |
| target lag | 35 | `lag1` 到 `lag673` 中的一组自动选择滞后 |
| 日期时间特征 | 5 | minute/hour/day/week/year 相关 |
| 类别 known covariate | 1 | `wind_direction` |

AutoGluon 内部 target lags 是：

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
95, 96, 97,
191, 192, 193,
287, 288, 289,
383, 384, 385,
479, 480, 481,
575, 576, 577,
671, 672, 673
```

这些 lag 的含义是：

- 近邻短滞后：1 到 14
- 前一天附近：95、96、97
- 前 2 天附近：191、192、193
- 前 3 天附近：287、288、289
- 一直到前 7 天附近：671、672、673

因为日前预测的窗口是 96 个 15 分钟点，所以 `lag96` 对应前一天同一时刻，`lag672` 对应前一周同一时刻。

AutoGluon 当前还启用了：

```text
target_scaler: standard
differences: [96]
```

其中 `differences: [96]` 表示它会使用 96 步季节差分，也就是按“前一天同一时刻”做差分建模。

注意：外部传入的 `wind_gust_mph` 在 known covariates 中存在，但最终内部 LGBModel 的 feature list 里没有出现，说明 AutoGluon 内部预处理/特征选择后没有把它送入最终这个 LGBModel。

## 模型参数设置

### LightGBM

模型类：`lightgbm.LGBMRegressor`

主要参数：

```yaml
n_estimators: 300
learning_rate: 0.05
objective: regression_l1
num_leaves: 31
random_state: 42
n_jobs: 1
verbosity: -1
enable_quantiles: true
lower_quantile: 0.1
upper_quantile: 0.9
```

当前会训练三套 `LGBMRegressor`：

- 点预测：`objective=regression_l1`
- P10：`objective=quantile, alpha=0.1`
- P90：`objective=quantile, alpha=0.9`

CSV 字段语义：

- `prediction`：点预测模型输出
- `0.1` / `0.9`：分位数模型输出，用于图中 P10-P90 阴影区间
- `prediction_lower` / `prediction_upper`：点预测模型残差标准差区间，作为兜底诊断区间

完整已加载参数：

```text
boosting_type: gbdt
colsample_bytree: 1.0
importance_type: split
learning_rate: 0.05
max_depth: -1
min_child_samples: 20
min_child_weight: 0.001
min_split_gain: 0.0
n_estimators: 300
n_jobs: 1
num_leaves: 31
objective: regression_l1
random_state: 42
reg_alpha: 0.0
reg_lambda: 0.0
subsample: 1.0
subsample_for_bin: 200000
subsample_freq: 0
verbosity: -1
```

### sklearn HistGradientBoosting

模型类：`sklearn.ensemble.HistGradientBoostingRegressor`

测试中的主要参数：

```yaml
max_iter: 80
loss: absolute_error
random_state: 42
enable_quantiles: true
lower_quantile: 0.1
upper_quantile: 0.9
```

当前会训练三套 `HistGradientBoostingRegressor`：

- 点预测：`loss=absolute_error`
- P10：`loss=quantile, quantile=0.1`
- P90：`loss=quantile, quantile=0.9`

CSV 字段语义同 LightGBM：`prediction` 来自点预测模型，`0.1/0.9` 来自分位数模型，`prediction_lower/prediction_upper` 来自点预测残差区间。

完整已加载参数：

```text
categorical_features: from_dtype
early_stopping: auto
l2_regularization: 0.0
learning_rate: 0.1
loss: absolute_error
max_bins: 255
max_depth: None
max_features: 1.0
max_iter: 80
max_leaf_nodes: 31
min_samples_leaf: 20
n_iter_no_change: 10
quantile: None
random_state: 42
scoring: loss
tol: 1e-07
validation_fraction: 0.1
verbose: 0
warm_start: False
```

### AutoGluon RecursiveTabular GBM

模型类：`autogluon.timeseries.TimeSeriesPredictor`

主要参数：

```yaml
eval_metric: MAE
presets: medium_quality
time_limit: 600
enable_ensemble: false
random_seed: 42
hyperparameters:
  RecursiveTabular:
    model_name: GBM
    model_hyperparameters:
      seed: 42
```

预测器参数：

```text
prediction_length: 96
freq: 15min
known_covariates_names:
  - temperature_f
  - dew_point_f
  - humidity_pct
  - wind_direction
  - wind_speed_mph
  - wind_gust_mph
  - pressure_in
quantile_levels:
  - 0.1
  - 0.2
  - 0.3
  - 0.4
  - 0.5
  - 0.6
  - 0.7
  - 0.8
  - 0.9
```

## 对比结论

LightGBM 和 sklearn 的输入特征是严格一致的 53 维，因此二者结果更适合直接比较模型算法差异。

AutoGluon 虽然底层配置了 `RecursiveTabular -> GBM`，但它接收的是原始时序长表和 7 个 known covariates，并在内部自动构造时序特征、验证窗口和递归预测逻辑。因此 AutoGluon 与项目内 LightGBM 的对比，本质上是两个预测框架的对比，而不是同一份 53 维特征矩阵上的同参数 GBM 对比。
