# 日前负荷预测模型对比

本文记录 `tests/pipeline/` 下 end-to-end 测试的最新运行结果。参数调优类测试未运行。

## 测试结论

| 项目 | 结果 |
|---|---|
| 运行命令 | `PYTHONPATH=src python -m pytest tests/pipeline -k "end_to_end and not parameterTuning" -q` |
| 测试结果 | `12 passed, 9 deselected` |
| 总耗时 | `2119.95s`，约 35 分 19 秒 |
| 数据范围 | `2019-01-01 00:00:00` 到 `2019-12-31 23:45:00` |
| 数据集 | `inputs/EWELD/eweld_industrial_u141_2018_2019_15min.csv` |
| 日前预测长度 | 96 点，也就是 1 天 |
| 日前测试集 | 10,464 行，109 个滚动预测窗口 |
| 默认节假日特征 | `add_chinese_calendar=true` |

本轮已重新生成 `outputs/` 下各模型的 model artifact、forecast、plot 和 metrics。`with_parameterTuning` 相关测试被排除。

## 日前指标排序

| 排名 | 模型 | MAE | RMSE | WAPE | Pinball mean | P10-P90 coverage | P10-P90 width | Metrics |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | AutoGluon RecursiveTabular GBM | 166.4757 | 250.8682 | 0.1089 | 83.0600 | 0.9744 | 1412.0624 | `outputs/autogluon/day_ahead/artifacts/metrics.json` |
| 2 | LightGBM + similar time | 170.8192 | 242.2982 | 0.1117 | 69.1520 | 0.3220 | 160.6024 | `outputs/lightgbm/day_ahead/similar_time/artifacts/metrics.json` |
| 3 | LightGBM | 172.3548 | 247.7214 | 0.1127 | 70.1849 | 0.3390 | 165.8517 | `outputs/lightgbm/day_ahead/artifacts/metrics.json` |
| 4 | GMM residual + LightGBM point | 172.3548 | 247.7214 | 0.1127 | 58.4329 | 0.3976 | 188.1955 | `outputs/gmm/day_ahead/e2e_probability_plots/artifacts/metrics.json` |
| 5 | VMD-LightGBM | 176.9815 | 254.7825 | 0.1158 | 70.7154 | 0.3604 | 179.4323 | `outputs/vmd_lightgbm/day_ahead/artifacts/metrics.json` |
| 6 | sklearn HistGradientBoosting | 180.6871 | 255.3362 | 0.1182 | 73.2827 | 0.3084 | 165.1582 | `outputs/sklearn_hist_gradient_boosting/day_ahead/artifacts/metrics.json` |
| 7 | CNN-LSTM-Attention | 190.8574 | 266.8758 | 0.1249 | 48.4729 | 0.7245 | 485.9834 | `outputs/cnn_lstm_attention/day_ahead/artifacts/metrics.json` |
| 8 | CNN-LSTM | 194.1093 | 266.8857 | 0.1270 | 47.0560 | 0.7712 | 576.5685 | `outputs/cnn_lstm/day_ahead/artifacts/metrics.json` |
| 9 | BiLSTM | 206.0485 | 281.9095 | 0.1348 | 53.7519 | 0.6373 | 446.6881 | `outputs/bilstm/day_ahead/artifacts/metrics.json` |
| 10 | LSTM | 231.4616 | 306.1565 | 0.1514 | 53.9204 | 0.7642 | 659.8550 | `outputs/lstm/day_ahead/artifacts/metrics.json` |
| 11 | MDN | 319.6438 | 412.6459 | 0.2091 | 133.9459 | 0.1476 | 145.9977 | `outputs/mdn/day_ahead/e2e_mdn/artifacts/metrics.json` |

补充：`short_term_4h` 的 LightGBM end-to-end 也已通过，MAE 为 `133.0529`，对应 `outputs/lightgbm/short_term_4h/artifacts/metrics.json`。该结果不是日前 96 点任务，不参与上表排序。

## 结果解读

AutoGluon 的 MAE 最低，但 P10-P90 区间非常宽，coverage 高达 `0.9744`，说明区间预测偏保守。若只看点预测 MAE，AutoGluon 是当前最强结果；若同时看区间宽度和概率质量，需要谨慎比较。

LightGBM + similar time 比 LightGBM baseline 的 MAE 从 `172.3548` 降到 `170.8192`，本轮是小幅正收益。这个版本只增加相似时刻差值类特征，LightGBM 参数与 baseline 保持一致，因此这个对比更能反映特征增益。

GMM 的点预测和 LightGBM 完全一致，因为它的 point estimator 就是同参数 LightGBM；差异主要在残差 GMM 概率分布输出，因此 pinball mean、coverage、width 不同。

VMD-LightGBM 低于 LightGBM baseline。它每次在线预测都只使用当前可见历史窗口做 VMD，再预测各分量并叠加，避免了未来泄漏；但 VMD 分解会引入端点效应、分量预测误差叠加和额外计算成本。本轮 MAE `176.9815`，说明当前数据和参数下 VMD 没有带来点预测收益。

CNN-LSTM-Attention、CNN-LSTM、BiLSTM 都优于纯 LSTM。当前 attention 结构带来小幅 MAE 改善：CNN-LSTM `194.1093`，CNN-LSTM-Attention `190.8574`。但它们仍弱于 tabular 树模型，说明现阶段树模型 + 显式滞后/统计特征仍是更强的点预测基线。

## 公共数据与特征

所有 pipeline 使用同一份清洗后的 15 分钟负荷数据。原始已知协变量为：

```text
temperature_f, dew_point_f, humidity_pct, wind_direction,
wind_speed_mph, wind_gust_mph, pressure_in
```

`FeatureBuilder` 路径当前默认加入中文节假日特征，训练日志显示 tabular 特征数从原先 53 增加到 58。新增 5 个特征为：

```text
feature__chinese_calendar__is_holiday
feature__chinese_calendar__is_workday
feature__chinese_calendar__is_in_lieu
feature__chinese_calendar__is_pre_holiday
feature__chinese_calendar__is_post_holiday
```

tabular 模型使用的 58 维特征包括：

| 特征组 | 数量 | 说明 |
|---|---:|---|
| 普通日历特征 | 9 | dayofweek、weekend、year、month、day、quarter、dayofyear、hour、minute |
| 周期时间特征 | 4 | 日周期和周周期 sin/cos |
| 中文节假日特征 | 5 | holiday、workday、调休、节前、节后 |
| item 特征 | 1 | 当前测试数据为 `U141` |
| 数值天气协变量 | 6 | 温度、露点、湿度、风速、阵风、气压 |
| `wind_direction` one-hot | 18 | 风向类别 |
| lag 特征 | 9 | `1,2,4,8,16,32,96,192,672` |
| rolling 特征 | 6 | 窗口 `4,16,96` 的 mean/std |

AutoGluon 当前走自身 `TimeSeriesPredictor` 路径，不经过项目内 `FeatureBuilder`。因此本轮 AutoGluon 使用原始 target 和 known covariates，由 AutoGluon 内部生成递归特征；中文节假日特征没有作为额外 known covariate 显式注入 AutoGluon。

LSTM、BiLSTM、CNN-LSTM、CNN-LSTM-Attention 不使用 tabular 58 维矩阵，而是构造序列窗口。当前 LSTM 族已通过 `add_chinese_calendar_features` 使用中文节假日特征，日志中历史数值维度为 `25`、未来数值维度为 `24`。

## 模型结构与关键参数

| 模型 | 结构/特征 | 关键参数 |
|---|---|---|
| LightGBM | recursive one-step tabular forecaster，逐点滚动预测 96 点 | `n_estimators=300`，`learning_rate=0.05`，`objective=regression_l1`，`num_leaves=31`，`quantile_levels=0.05..0.95` |
| LightGBM + similar time | LightGBM baseline + 相似时刻特征 | 相似候选 `top_k=5`，候选回看 90 天，同 15min slot 附近，lag `96,192,672`，rolling `96,672`，各距离权重为 `1.0` |
| VMD-LightGBM | 先对可见历史窗口做 VMD，分 4 个 mode；每个 mode 训练同参数 LightGBM，预测后叠加 | `num_modes=4`，`alpha=2000`，`max_iter=120`，`tolerance=1e-5`；LightGBM 参数与 baseline 一致 |
| sklearn HGB | recursive tabular forecaster，底层为 HistGradientBoostingRegressor | `max_iter=80`，`loss=absolute_error`，同样输出 0.05 到 0.95 分位数 |
| GMM | LightGBM 点预测 + 残差 Gaussian Mixture | `n_components=3`，`gmm_max_iter=120`，`n_init=2`，`distribution_grid_size=80` |
| AutoGluon | `TimeSeriesPredictor` + `RecursiveTabular` GBM | `eval_metric=MAE`，`presets=medium_quality`，`time_limit=600`，`enable_ensemble=False` |
| MDN | tabular FeatureBuilder 输入，PyTorch mixture density network 输出概率分布 | `n_components=2`，`hidden_size=32`，`epochs=35`，`max_train_samples=16384`，`batch_size=128` |
| LSTM | 历史窗口 + 未来协变量，LSTM encoder，MDN head 一次输出 96 点 | `context_length=672`，`hidden_size=64`，`epochs=50`，`max_train_samples=8192`，`batch_size=128` |
| BiLSTM | LSTM 的 bidirectional 版本，MDN head | 同 LSTM 主结构，但 `max_train_samples=4096`，`bidirectional=True` |
| CNN-LSTM | 历史窗口先过轻量 1D CNN，再进入 LSTM，最后 MDN head | LSTM 参数同当前 LSTM；`cnn_channels=32`，`cnn_kernel_size=5`，`cnn_layers=1` |
| CNN-LSTM-Attention | CNN-LSTM + anchor attention context，再接 MDN head | `attention_hidden_size=64`，`recent_steps=32`，anchor periods `96,192,672`，`anchor_window=4` |

## 结构说明

CNN-LSTM 当前结构可以概括为：

```text
历史负荷窗口 + 历史协变量 + 日历/节假日/lag 特征
  -> 1D CNN 提取局部波动、尖峰、爬坡模式
  -> LSTM 编码长短期时序依赖
  -> final hidden state 作为历史窗口摘要
  -> 拼接未来已知协变量
  -> MDN head 输出 96 点概率预测
```

CNN-LSTM-Attention 当前结构是在 CNN-LSTM 基础上增加 anchor attention：

```text
CNN-LSTM encoder 输出序列 hidden states
  -> 保留 final hidden state
  -> 从关键 anchor 历史点取 attention context
     anchor 包括昨天同一时刻、前天同一时刻、上周同一时刻附近
  -> final hidden state + anchor attention context + future covariates
  -> MDN head
```

这里的 final hidden state 可以理解为 LSTM 对整个历史窗口的压缩摘要；attention context 是从更明确的关键历史位置提取的补充信息。

## 本轮结论

当前点预测主线仍建议以 LightGBM / LightGBM + similar time 作为强 baseline。AutoGluon 点预测最好，但区间过宽且特征路径与项目内 FeatureBuilder 不完全一致。VMD-LightGBM 本轮没有正收益，且在线推理成本显著高于 LightGBM。CNN-LSTM-Attention 相比 CNN-LSTM 有小幅提升，但仍没有超过树模型。
