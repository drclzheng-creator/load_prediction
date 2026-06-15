# Load Prediction

面向虚拟电厂、聚合资源、备用响应和电力交易场景的多时间尺度负荷预测框架。

## 已落地的框架分层

- `data`: CSV 接入、标准长表结构。
- `preprocessing`: 时间对齐、重复时间戳聚合、缺失值填补、异常值裁剪。
- `features`: 日历特征、周期正余弦编码、滞后特征、滑窗特征、协变量入口、相关性筛选。
- `models`: `BaseForecastModel` 抽象接口、sklearn 递归预测器、LightGBM 适配器、可选 AutoGluon 适配器。
- `postprocessing`: 非负裁剪、保留模型预测区间；无模型区间时用残差区间兜底。
- `evaluation`: MAE、RMSE、MAPE。
- `visualization`: 测试集真实值与预测值对比图。
- `pipeline`: 单尺度和多尺度编排。
- `config`: 多尺度预设、模型/清洗/特征/评估配置。

## 多时间尺度预设

- `ultra_short`: 5 分钟粒度，默认预测未来 12 个点。
- `day_ahead`: 15 分钟粒度，默认预测未来 96 个点。
- `hourly`: 1 小时粒度，默认预测未来 24 个点。
- `medium_term`: 1 天粒度，默认预测未来 30 个点。

## 快速运行

```bash
./.venv/bin/python scripts/run_demo.py demo --scale day_ahead --input-path inputs/electricity.csv
```

多尺度 Demo：

```bash
./.venv/bin/python scripts/run_demo.py multi-demo --scales ultra_short,day_ahead,hourly --input-path inputs/electricity.csv
```

按 YAML 场景配置运行：

```bash
./.venv/bin/python scripts/run_demo.py run-config configs/day_ahead.yaml
```

运行产物会按模型和预测尺度统一保存：

```text
outputs/
  lightgbm/
    day_ahead/
      model/
        model.joblib
      logs/
        train.log
      artifacts/
        forecast.csv
        forecast.png
        metrics.json
```

AutoGluon 会使用同样的目录结构，只是 `model/` 下保存的是 AutoGluon 自己的训练产物。

当前提供三份场景配置：

- `configs/ultra_short.yaml`: 超短期预测。
- `configs/day_ahead.yaml`: 日前预测。
- `configs/medium_term.yaml`: 中长期预测。

本地 CSV：

```bash
./.venv/bin/python scripts/run_demo.py forecast data/load.csv \
  --scale day_ahead \
  --timestamp-col timestamp \
  --target-col target \
  --item-id-col item_id \
  --covariates temperature,humidity
```

CSV 至少需要：

- `timestamp`: 时间列。
- `target`: 负荷列。
- `item_id`: 可选，缺省时使用单站点。

输出文件默认写入 `outputs/{model_name}/{scale_name}/`。可以在 YAML 的 `output.root_dir` 或命令行的 `--output-root` 中调整根目录。

## 特征 Registry

外部特征不再靠代码里写死 `["timestamp", "item_id", "target"]` 后手工拼列名，而是通过 feature registry 和配置共同决定：

```yaml
data:
  feature_sets:
    - weather_temperature
  covariate_cols:
    - humidity

feature:
  external_feature_sets:
    - weather_temperature
  known_covariates:
    - humidity
```

- `data.feature_sets`: 控制 CSV loader 从原始文件中读取哪些已注册外部特征列。
- `data.covariate_cols`: 额外显式指定要读取的列。
- `feature.external_feature_sets`: 控制特征工程阶段把哪些已注册外部特征转成模型特征。
- `feature.known_covariates`: 额外显式指定进入模型的协变量。

当前 registry 已包含 `weather_temperature`、`weather_humidity`、`production`、`traffic`，后续新增特征组只需要扩展 `src/load_prediction/features/registry.py`。

## 负荷+天气数据

项目里已经放了一个较小的公开样例：`inputs/load_weather/swiss_load_weather.csv`，列包括 `timestamp,item_id,target` 和多个温度协变量，可用于验证天气特征链路。

手工下载地址：

- Swiss load + weather 原始 CSV：https://raw.githubusercontent.com/dafrie/lstm-load-forecasting/master/data/fulldataset.csv
- Swiss load + weather GitHub 仓库：https://github.com/dafrie/lstm-load-forecasting
- 如果 raw 下载很慢，可用 GitHub blob API：https://api.github.com/repos/dafrie/lstm-load-forecasting/git/blobs/aa3463b4da8c1bf587c4d4bface151778b4e91f3
- OPSD 负荷/发电时间序列：https://data.open-power-system-data.org/time_series/2019-06-05/time_series_60min_singleindex.csv
- OPSD 天气数据：https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv

## 评估集切分

默认按每个 `item_id` 的时间顺序 7:3 切分训练集和测试集，避免随机切分造成时间泄漏：

```yaml
evaluation:
  split_strategy: ratio
  train_ratio: 0.7
  holdout_length:
```

如果配置了 `train_ratio`，优先按比例切分。只有当 `train_ratio` 为空时，才会使用固定测试窗口 `holdout_length`。

## 重采样策略

清洗层会显式区分降采样、同频和升采样，并通过日志输出处理路径。升采样会制造高频数据，默认不会静默处理：

```yaml
cleaning:
  resample: true
  upsample_strategy: warn_interpolate  # error | warn_interpolate | interpolate | ffill
```

- `error`: 发现原始频率低于目标频率时直接报错，适合生产训练。
- `warn_interpolate`: 打 warning，然后按缺失值策略补齐，适合探索和 demo。
- `interpolate`: 明确允许插值补齐。
- `ffill`: 数值列前向填充，适合阶梯型信号，不适合连续负荷功率。

## Scaler 规则

模型配置支持特征和目标缩放：

```yaml
model:
  name: sklearn_random_forest
  scale_features: true
  scale_target: true
```

Scaler 只在训练集 `fit`，测试集和线上递归预测只调用 `transform`；目标缩放会在输出前做 `inverse_transform`。

## LightGBM 模型

LightGBM 已按 `BaseForecastModel` 接口适配：

```yaml
model:
  name: lightgbm
  params:
    n_estimators: 300
    learning_rate: 0.05
    objective: regression_l1
    num_leaves: 31
```

LightGBM 默认使用 `objective: regression_l1`，即按 L1/MAE 方向优化训练目标；最终测试集评估仍同时输出 MAE、RMSE、MAPE。树模型通常不需要特征缩放，默认 `scale_features: false`。

## Sklearn 模型

sklearn 递归模型也默认按 MAE/L1 方向配置：

```yaml
model:
  name: sklearn_random_forest
  params:
    criterion: absolute_error
```

`sklearn_hist_gradient_boosting` 默认使用：

```yaml
model:
  name: sklearn_hist_gradient_boosting
  params:
    loss: absolute_error
```

## 可选 AutoGluon 模型

核心框架默认不强依赖 AutoGluon。需要使用时先安装可选依赖，再把配置中的模型名改为 `autogluon`：

```yaml
model:
  name: autogluon
  params:
    eval_metric: MAE
    presets: medium_quality
    time_limit: 300
```

AutoGluon TimeSeries 自身会集成多类时间序列模型。可以通过 `model.params.hyperparameters` 指定要训练的模型族，通过 `excluded_model_types` 排除模型族：

```yaml
model:
  name: autogluon
  params:
    eval_metric: MAE
    presets: medium_quality
    time_limit: 300
    hyperparameters:
      Naive: {}
      SeasonalNaive: {}
      RecursiveTabular:
        model_name: LGB
      DirectTabular:
        model_name: CAT
    excluded_model_types:
      - Chronos
```

这里的 `eval_metric: MAE` 控制 AutoGluon 内部模型选择/验证偏向 MAE；最终 pipeline 仍会统一输出 MAE、RMSE、MAPE。

本项目会先校验 AutoGluon 模型名，不在白名单内会直接报错。当前支持的模型名包括：

```text
ADIDA, ARIMA, AutoARIMA, AutoCES, AutoETS, Average, Chronos, Chronos2,
Chronos-2, Croston, CrostonSBA, DLinear, DeepAR, DirectTabular,
DynamicOptimizedTheta, ETS, IMAPA, NPTS, Naive, PatchTST, PerStepTabular,
RecursiveTabular, SeasonalAverage, SeasonalNaive, SimpleFeedForward, TFT,
TemporalFusionTransformer, Theta, TiDE, Toto, WaveNet, Zero
```

如果 `hyperparameters` 没有配置，框架默认使用轻量且适合协变量的：

```yaml
hyperparameters:
  RecursiveTabular:
    model_name: GBM
```
