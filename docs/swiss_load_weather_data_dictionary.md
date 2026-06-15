# Swiss Load Weather 数据字典

文件路径：`inputs/load_weather/swiss_load_weather.csv`

该文件来自公开项目 `dafrie/lstm-load-forecasting`，已转换成本项目统一长表格式。它可用于验证“负荷时序 + 天气协变量”的预测链路。

## 字段说明

| 字段 | 类型 | 含义 | 建模角色 |
| --- | --- | --- | --- |
| `timestamp` | datetime | 观测时间，已转换为无时区时间戳 | 时间索引，用于排序、切分、重采样和时间特征构造 |
| `item_id` | string | 序列 ID，当前固定为 `swiss_load` | 多序列标识，当前样例为单序列 |
| `target` | float | 瑞士系统实际负荷，来自原始字段 `actual` | 预测目标 |
| `bsl_t` | float | Basel 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `brn_t` | float | Bern 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `zrh_t` | float | Zurich 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `lug_t` | float | Lugano 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `lau_t` | float | Lausanne 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `gen_t` | float | Geneva 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `stg_t` | float | St. Gallen 温度特征 | 已注册到 `weather_temperature` 特征组 |
| `luz_t` | float | Luzern 温度特征 | 已注册到 `weather_temperature` 特征组 |

## 配置方式

在 YAML 或代码配置中启用温度特征组：

```yaml
data:
  feature_sets:
    - weather_temperature

feature:
  external_feature_sets:
    - weather_temperature
```

其中：

- `data.feature_sets` 控制 CSV loader 从文件中读取哪些外部特征列。
- `feature.external_feature_sets` 控制特征工程把这些列转换为模型输入特征。

## 注意事项

- 该数据是单序列系统负荷，不是单个建筑或单个电表负荷。
- 温度列是否为摄氏度需以原始数据源说明为准；当前框架只把它们作为数值协变量使用。
- 递归预测测试集时，测试集里的温度列会作为未来已知协变量传入模型；线上预测时也需要提供同一预测窗口的未来天气值。
