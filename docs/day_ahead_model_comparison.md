# 日前负荷预测模型对比

本文基于 `tests/pipeline/` 的最新完整运行结果，比较当前代码库中的日前预测模型、输入特征处理方式和离散场景概率生成方式。场景概率部分沿用当前已保存的 scenario 输出结构说明。

## 测试结论

- 本次刷新命令：`python -m pytest -q tests/pipeline`
- 结果：`9 passed in 499.38s`
- 本次运行重新生成了 AutoGluon、LightGBM、GMM、sklearn HistGradientBoosting、MDN、LSTM、BiLSTM 以及 LightGBM short-term 4h 的 model artifact、forecast 和 metrics。
- 当前工程已在 `tests/conftest.py` 中设置 `LOKY_MAX_CPU_COUNT=1`，未再出现 joblib/loky 读取物理 CPU 数的 warning。

## 评估设置

| 项目 | 设置 |
|---|---|
| 数据集 | `inputs/EWELD/eweld_industrial_u141_2018_2019_15min.csv` |
| 测试时间范围 | `2019-01-01 00:00:00` 到 `2019-12-31 23:45:00` |
| 时间粒度 | 15 分钟 |
| 预测任务 | 日前预测 |
| 单次预测长度 | 96 点，也就是 1 天 |
| 训练/测试切分 | 按时间顺序 7:3，并对齐到 96 点 |
| 训练行数 | 24,576 |
| 测试行数 | 10,464 |
| 回测方式 | 测试集按 109 个日窗口滚动预测，每个窗口 96 点 |
| 指标 | MAE、RMSE、MAPE |

## 指标对比

| 排名 | 模型 | MAE | RMSE | MAPE | 评估行数 | Forecast CSV |
|---:|---|---:|---:|---:|---:|---|
| 1 | AutoGluon RecursiveTabular GBM | 166.4757 | 250.8682 | 0.1817 | 10464 | `outputs/autogluon/day_ahead/artifacts/forecast.csv` |
| 2 | LightGBM | 171.5629 | 246.3731 | 0.1753 | 10464 | `outputs/lightgbm/day_ahead/artifacts/forecast.csv` |
| 2 | GMM residual model, LightGBM point estimator | 171.5629 | 246.3731 | 0.1753 | 10464 | `outputs/gmm/day_ahead/e2e_probability_plots/artifacts/forecast.csv` |
| 4 | sklearn HistGradientBoosting | 173.6976 | 246.1750 | 0.1768 | 10464 | `outputs/sklearn_hist_gradient_boosting/day_ahead/artifacts/forecast.csv` |
| 5 | BiLSTM MDN head | 208.8135 | 284.5115 | 0.2210 | 10464 | `outputs/bilstm/day_ahead/artifacts/forecast.csv` |
| 6 | LSTM, last 4096 samples | 220.6440 | 293.0014 | 0.2052 | 10464 | `outputs/lstm/day_ahead/artifacts/forecast.csv` |
| 7 | MDN, last 16384 samples | 254.8283 | 331.2617 | 0.2507 | 10464 | `outputs/mdn/day_ahead/e2e_mdn/artifacts/forecast.csv` |

AutoGluon 的 MAE 最低；LightGBM 和调优后的 GMM 在点预测指标上完全一致，因为 GMM 当前使用同一类 LightGBM 点估计器，差异主要在残差 GMM 概率分布输出。sklearn 的 RMSE 略低，但 MAE/MAPE 略高。BiLSTM 和 LSTM 本次重跑结果明显好于此前记录，但点预测误差仍高于 AutoGluon、LightGBM、GMM 和 sklearn。MDN 改为按时间顺序取最后 16384 个训练样本后，MAE 为 `254.8283`，仍是当前日前模型中点预测误差最高的一组。MDN/LSTM/BiLSTM 的价值主要在原生概率输出；如果只看点预测，当前树模型和 AutoGluon 仍是更强基线。

## Recursive 与 Direct 96 点预测

当前结果只能说明：在这组测试配置、这份数据、这套特征和训练预算下，recursive tabular/AutoGluon 模型的点预测指标优于当前 MDN、LSTM、BiLSTM 这些直接输出 96 点的模型。它不能单独证明“recursive 预测方式天然优于 direct 96 点预测”。

原因是这里同时改变了多件事：

- 模型族不同：LightGBM/sklearn/AutoGluon 是树模型，MDN/LSTM/BiLSTM 是神经网络或 mixture 模型。
- 特征不同：recursive tabular 模型显式使用 calendar、lag、rolling、天气 one-hot 等 53 维特征；LSTM 使用序列窗口和 embedding；AutoGluon 还有内部自动构造的 lag/差分特征。
- 训练目标不同：LightGBM/sklearn 的点预测目标偏 MAE/L1；当前 MDN、LSTM、BiLSTM 训练的是 mixture negative log-likelihood，点预测只是 mixture mean。
- 训练预算不同：当前 MDN 使用训练集最后 16384 个监督样本、35 个 epoch 上限并早停；LSTM 使用最后 4096 个序列样本并早停，BiLSTM 使用最多 4096 个随机序列样本并早停，未做系统调参。
- direct 96 点模型更难：它一次性学习完整 horizon 的联合模式，输出更丰富，但参数量和优化难度也更高。

更严谨的结论应该是：当前数据规模和配置下，带强特征工程的 recursive tabular 方法是更强的点预测基线。要判断 recursive 和 direct horizon 的方法差异，需要做同模型族对照，例如同样的 LightGBM 分别实现 recursive 和 direct multi-output，或者同样的 LSTM 分别实现 recursive one-step 和 direct 96-step。

## 输入特征与特征处理

所有模型使用同一份清洗后的日前数据：目标列 `target`，实体列 `item_id`，时间列 `timestamp`，已知协变量为 `temperature_f`、`dew_point_f`、`humidity_pct`、`wind_direction`、`wind_speed_mph`、`wind_gust_mph`、`pressure_in`。

| 模型 | 输入特征 | 特征处理 |
|---|---|---|
| LightGBM | 日历特征、周期时间特征、item one-hot、已知天气协变量、目标滞后、滚动统计 | `FeatureBuilder` 生成 53 个 tabular 特征；数值协变量直接使用；`wind_direction` one-hot；滞后为 1/2/4/8/16/32/96/192/672；滚动窗口为 4/16/96 |
| sklearn HistGradientBoosting | 同 LightGBM | 同 LightGBM，使用 sklearn HGB 点模型和分位数模型 |
| GMM residual model | 同 LightGBM | 先训练递归 tabular LightGBM 点估计器，再对残差拟合 Gaussian mixture |
| MDN | 同 LightGBM 的 tabular 特征 | `FeatureBuilder` 输出 tabular 特征后送入 PyTorch MDN；训练样本上限为 16384，抽样策略为取训练集最后一段；使用纯 mixture NLL；输出 mixture 参数 |
| AutoGluon | 原始时间序列、目标值、已知天气协变量 | AutoGluon `TimeSeriesPredictor` 处理时间索引和 known covariates；当前配置只训练 `RecursiveTabular` GBM，不启用 ensemble |
| LSTM | 历史目标序列、历史数值协变量、未来已知数值协变量、`wind_direction` 类别编码 | context length 为 672，prediction length 为 96；数值特征标准化；类别协变量 embedding；MDN head 输出每个 horizon 的 mixture；代码支持可选 point head 和 NLL+MAE 点预测辅助损失 |
| BiLSTM | 同 LSTM | LSTM encoder 改为 bidirectional，其他概率输出路径同 LSTM |

### 特征维度总览

| 模型 | 送入模型的主要数据形态 | 显式特征维度 | 说明 |
|---|---|---:|---|
| LightGBM | `FeatureBuilder` 生成的监督学习矩阵 | 53 | recursive one-step tabular 特征，逐点滚动预测 96 点 |
| sklearn HistGradientBoosting | `FeatureBuilder` 生成的监督学习矩阵 | 53 | 与 LightGBM 完全同一套外部特征 |
| GMM residual model | `FeatureBuilder` 生成的监督学习矩阵 + 残差 GMM | 53 | 点估计器改为 LightGBM，使用 53 维 recursive tabular 特征，GMM 拟合残差分布 |
| MDN | `FeatureBuilder` 生成的监督学习矩阵 | 53 | tabular 输入，神经网络输出 mixture 参数 |
| AutoGluon RecursiveTabular GBM | `TimeSeriesDataFrame`，包含 target 和 known covariates | 外部 7 个 known covariates | 内部 RecursiveTabular 会自动构造 lag、日期、差分等特征 |
| LSTM | 序列张量：历史窗口 + 未来 known covariates | 历史数值 7 维，未来数值 6 维，类别 embedding 4 维 | 直接输出 96 点 horizon 的 mixture 参数 |
| BiLSTM | 同 LSTM | 同 LSTM | encoder 为 bidirectional |

### FeatureBuilder 的 53 维特征

LightGBM、sklearn、GMM 和 MDN 使用项目内同一套 `FeatureBuilder` 特征：

- 日历特征：9 维
- 周期时间特征：4 维
- item ID 特征：1 维
- 数值天气协变量：6 维
- `wind_direction` 类别 one-hot：18 维
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

### AutoGluon 输入与内部特征

AutoGluon 外部接收长表时序数据：

- `item_id`
- `timestamp`
- `target`
- 7 个 known covariates：`temperature_f`、`dew_point_f`、`humidity_pct`、`wind_direction`、`wind_speed_mph`、`wind_gust_mph`、`pressure_in`

AutoGluon 日志识别结果：

```text
categorical:        ['wind_direction']
continuous (float): ['temperature_f', 'dew_point_f', 'humidity_pct', 'wind_speed_mph', 'wind_gust_mph', 'pressure_in']
```

保存模型中的 RecursiveTabular 内部结构为：

```text
MultiWindowBacktestingModel
  -> RecursiveTabularModel
    -> MLForecast
      -> AutoGluon Tabular LGBModel
```

参考 `day_ahead_model_features_and_params.md` 中的模型检查结果，内部 LGBModel 最终使用 51 维特征：

- 原始数值 known covariates：5 维
- 缩放后的数值 known covariates：5 维
- target lag：35 维
- 日期时间特征：5 维
- 类别 known covariate：1 维

内部 target lags 为：

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

AutoGluon 当前还启用了 `target_scaler: standard` 和 `differences: [96]`，其中 `96` 表示按前一天同一时刻做季节差分。

### LSTM / BiLSTM 序列特征

LSTM 和 BiLSTM 不使用 `FeatureBuilder` 的 53 维 tabular 矩阵，而是构造直接预测 96 点的序列样本：

- `context_length: 672`，即历史 7 天，每天 96 个 15 分钟点。
- `prediction_length: 96`，即一次输出下一天 96 点。
- history numeric features：历史 `target` + 6 个数值天气协变量，共 7 维。
- future numeric features：未来已知的 6 个数值天气协变量，不含未来 `target`。
- categorical feature：`wind_direction`，类别数 18，embedding dim 为 4。
- 数值特征经过 `StandardScaler`；类别特征映射为整数后进入 embedding。
- 输出层为 MDN head，每个 horizon 输出 `n_components * 3` 个参数，即 weights、means、stds。

## 测试案例算法参数

以下参数来自 `tests/pipeline/` 中实际构造的测试配置；基础数据、清洗、评估和 postprocess 继承 `configs/day_ahead.yaml`，不同算法只覆盖 model 配置。

### LightGBM

```yaml
name: lightgbm
random_state: 42
scale_features: false
scale_target: false
params:
  n_estimators: 300
  learning_rate: 0.05
  objective: regression_l1
  num_leaves: 31
  enable_quantiles: true
  lower_quantile: 0.1
  upper_quantile: 0.9
  quantile_levels: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
```

当前会训练一个 L1 点预测模型和 11 个 quantile 模型。`prediction` 来自点预测模型，`0.05` 到 `0.95` 来自分位数模型。

### sklearn HistGradientBoosting

```yaml
name: sklearn_hist_gradient_boosting
random_state: 42
params:
  max_iter: 80
  enable_quantiles: true
  lower_quantile: 0.1
  upper_quantile: 0.9
  quantile_levels: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
  loss: absolute_error
```

点预测模型使用 `loss=absolute_error`，同时训练 11 个 `loss=quantile` 的分位数模型。

### GMM residual model

```yaml
name: gmm
random_state: 42
params:
  n_components: 3
  point_estimator: lightgbm
  point_params:
    n_estimators: 300
    learning_rate: 0.05
    objective: regression_l1
    num_leaves: 31
    n_jobs: 1
    verbosity: -1
  distribution_grid_size: 80
  distribution_std_width: 4.0
  gmm_max_iter: 120
  n_init: 2
```

点估计器已从 sklearn `HistGradientBoostingRegressor` 调整为 LightGBM `LGBMRegressor`，目标函数为 `regression_l1`。GMM 在点预测残差上拟合 3 个 Gaussian components；当前实现默认 `conditional_distribution: true`，训练样本为 `[fitted_value, residual]`。调优后 GMM 的点预测指标从 MAE `191.8768`、RMSE `265.8045`、MAPE `0.1905` 提升到 MAE `171.5629`、RMSE `246.3731`、MAPE `0.1753`。

### MDN

```yaml
name: mdn
random_state: 42
params:
  n_components: 2
  hidden_size: 32
  num_layers: 1
  dropout: 0.0
  epochs: 35
  batch_size: 128
  learning_rate: 0.001
  nll_loss_weight: 1.0
  mae_loss_weight: 0.0
  max_train_samples: 16384
  train_sample_strategy: last
  validation_fraction: 0.1
  early_stopping_patience: 8
  distribution_grid_size: 25
  device: cpu
```

训练目标保持纯 mixture negative log-likelihood：

```text
loss = 1.0 * mixture_nll
```

`prediction` 为 mixture mean，P10/P90 由 mixture CDF 反推。本次训练使用训练集最后 16384 个监督样本，epoch 上限为 35，实际在 epoch 20 早停；best validation loss 在 epoch 12，`validation_loss=-0.0689`。点预测 MAE 为 `254.8283`，好于最后 8192 样本的 `277.3631` 和随机抽样 8192 samples / 35 epochs 的 `301.5774`，但仍略差于旧的 2048 samples / 5 epochs 配置 `252.2748`。补充试验中，NLL + MAE 混合损失 `mae_loss_weight: 0.5` 的 MAE 为 `274.4228`，`mae_loss_weight: 0.1` 的 MAE 为 `309.7168`，都没有改善到旧配置水平。

### AutoGluon

```yaml
name: autogluon
random_state: 42
params:
  eval_metric: MAE
  presets: medium_quality
  time_limit: 600
  enable_ensemble: false
  quantile_levels: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
  hyperparameters:
    RecursiveTabular:
      model_name: GBM
```

测试中显式把 `time_limit` 设为 600 秒；实际日志显示只训练 `RecursiveTabular` 一个模型，不启用 ensemble。

### LSTM

```yaml
name: lstm
random_state: 42
params:
  context_length: 672
  hidden_size: 64
  num_layers: 1
  dropout: 0.1
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.0
  max_train_samples: 4096
  train_sample_strategy: last
  device: cpu
  validation_fraction: 0.1
  early_stopping_patience: 5
  early_stopping_min_delta: 0.0001
  use_point_head: false
  nll_loss_weight: 1.0
  point_loss_weight: 0.0
```

默认 `n_components: 3`、`distribution_grid_size: 80`、`distribution_std_width: 4.0`。代码已实现 5 项 LSTM 改造：可选 point head、point MAE 辅助损失、`train_sample_strategy` 时序抽样、按训练尾段做 validation split、以及 `dropout`/`weight_decay` 正则配置。

本次保守配置使用训练集最后 4096 个序列样本，保持纯 mixture NLL，早停于 epoch 9，best validation loss 在 epoch 4，点预测 MAE 为 `220.6440`。这是本次完整重跑生成并保存到 `outputs/lstm/day_ahead/artifacts/metrics.json` 的结果。

| LSTM 试验 | MAE | RMSE | MAPE | 说明 |
|---|---:|---:|---:|---|
| 当前保存配置，last 4096，纯 NLL | 220.6440 | 293.0014 | 0.2052 | 本次 `tests/pipeline` 完整重跑结果 |

因此当前代码保留 point head 和混合损失作为可选能力，但默认测试配置不开启 point head；本次重跑下默认 LSTM 已明显优于文档旧记录，不过仍未超过树模型基线。

### BiLSTM

```yaml
name: bilstm
random_state: 42
params:
  context_length: 672
  hidden_size: 64
  num_layers: 1
  dropout: 0.1
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  max_train_samples: 4096
  device: cpu
  validation_fraction: 0.1
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0001
```

BiLSTM 使用同 LSTM 的 MDN head，区别是 encoder 设置为 bidirectional。本次测试早停于 epoch 7，best validation loss 在 epoch 2，点预测 MAE 为 `208.8135`、RMSE 为 `284.5115`、MAPE 为 `0.2210`。

## 概率输出机制

| 模型 | Forecast 输出类型 | `distribution_values/probabilities` 来源 | P10/P90 来源 |
|---|---|---|---|
| LightGBM | 点预测 + 11 个分位数列 | 由分位数函数插值成 value grid，再用 `dP/dX` 近似密度 | 额外 quantile 模型的 `0.1/0.9` |
| sklearn HistGradientBoosting | 点预测 + 11 个分位数列 | 同 LightGBM | 额外 quantile 模型的 `0.1/0.9` |
| AutoGluon | AutoGluon mean + 11 个原生分位数列 | 同 LightGBM | AutoGluon 原生 `0.1/0.9` |
| GMM | 点预测 + `gmm_weights/gmm_means/gmm_stds` | 直接在 value grid 上评估 Gaussian mixture PDF | GMM 分布反推 |
| MDN | 点预测 + `mdn_weights/mdn_means/mdn_stds` | 直接在 value grid 上评估 MDN Gaussian mixture PDF | MDN mixture CDF 反推 |
| LSTM | 点预测 + `lstm_weights/lstm_means/lstm_stds` | 直接在 value grid 上评估 LSTM-MDN Gaussian mixture PDF | LSTM-MDN mixture CDF 反推 |
| BiLSTM | 点预测 + `lstm_weights/lstm_means/lstm_stds` | 同 LSTM | 同 LSTM |

分位数模型包括 LightGBM、sklearn 和 AutoGluon。它们的 forecast CSV 同时保留分位数列和密度网格；密度网格主要用于可视化和兼容输出，离散场景采样优先使用分位数函数。

原生 mixture 模型包括 GMM、MDN、LSTM 和 BiLSTM。它们直接输出 mixture weights、means、stds，因此概率分布和采样都来自模型自身的 mixture 参数。

## 离散场景概率

`tests/scenario/` 验证了 MDN、GMM、LightGBM、sklearn、LSTM 和 AutoGluon 的场景生成；BiLSTM 当前没有单独测试函数，但 forecast 使用同 LSTM 的 `lstm_*` mixture 参数，调用统一生成器时走同一条 mixture 路径。

统一配置如下：

| 参数 | 值 |
|---|---:|
| `num_scenarios` | 8 |
| `num_samples` | 10000 |
| `clip_min` | 0.0 |
| `kmeans_n_init` | 2 |
| `kmeans_max_iter` | 100 |
| `kmeans_fit_samples` | 1000 |

离散概率生成规则：

1. mixture 模型：按每行的 mixture weights 抽 component，再按对应 Gaussian mean/std 抽 10,000 个样本。
2. quantile 模型：抽 `u ~ Uniform(0, 1)`，用 forecast 中的分位数列插值得到 `Q(u)` 样本。
3. fallback：如果没有 mixture 参数和分位数列，但有 `distribution_values/probabilities`，则按 density grid 转 mass 后采样。
4. 对样本做非负裁剪。
5. 用 KMeans 将样本缩减为 8 个场景中心。
6. 每个场景概率等于全部样本分配到该中心的频率，最后按场景值升序输出。

本次 scenario 测试保存的 metadata：

| 模型 | `source_type` | `model_prefix` | random_state | 输出 |
|---|---|---|---:|---|
| MDN | `mixture` | `mdn` | 11 | `outputs/mdn/day_ahead/e2e_mdn/probability/point_scenarios.csv` |
| GMM | `mixture` | `gmm` | 11 | `outputs/gmm/day_ahead/e2e_probability_plots/probability/point_scenarios.csv` |
| LSTM | `mixture` | `lstm` | 19 | `outputs/lstm/day_ahead/probability/point_scenarios.csv` |
| LightGBM | `quantile_function` | null | 13 | `outputs/lightgbm/day_ahead/probability/point_scenarios.csv` |
| sklearn HGB | `quantile_function` | null | 17 | `outputs/sklearn_hist_gradient_boosting/day_ahead/probability/point_scenarios.csv` |
| AutoGluon | `quantile_function` | null | 23 | `outputs/autogluon/day_ahead/probability/point_scenarios.csv` |

## 实用建议

- 当前点预测误差优先级下，AutoGluon、LightGBM 和调优后的 GMM 是前三个候选。
- 如果重点是可解释、稳定、训练快，优先 LightGBM 或 sklearn；两者都已输出密集分位数并使用 quantile-function 采样。
- 如果重点是 AutoML 基线，当前 AutoGluon 只启用了 `RecursiveTabular` GBM；要比较完整 AutoML 能力，需要放开更多模型和 ensemble 后重新评估。
- 如果重点是原生概率分布，调优后的 GMM 目前是最实用的方案：点预测接近 LightGBM，同时保留 native mixture 参数和 mixture scenario 采样。MDN/LSTM/BiLSTM 的概率表达更灵活；本次重跑中 BiLSTM 和 LSTM 的点预测结果明显优于此前文档记录，但仍落后于当前树模型基线。后续若继续优化神经网络路线，可以重新系统搜索网络容量、学习率、抽样策略和损失权重。
- 后续若要把 BiLSTM 纳入 `tests/scenario/`，可新增一个与 LSTM 相同的 mixture scenario 测试，输入 `outputs/bilstm/day_ahead/artifacts/forecast.csv`。
