# EWELD Industrial U141 2018-2019 15min Dataset

文件：`eweld_industrial_u141_2018_2019_15min.csv`

该文件从 EWELD 原始数据中整理得到，选择制造业用户 `U141`，行业为 `C24 Manufacture of basic metals`，位置为 `CT2`，天气数据匹配 `Weather Data/W2.csv`。原始 `Condition` 列在该窗口全为空，已从标准样例文件中移除。

## 时间范围

- 开始：`2018-01-01 00:00:00`
- 结束：`2019-12-31 23:45:00`
- 频率：15 分钟
- 行数：`70080`

## 字段

| 字段 | 含义 |
| --- | --- |
| `timestamp` | 15 分钟时间戳 |
| `item_id` | 用户 ID，固定为 `U141` |
| `target` | 用户电力负荷，来自原始 `Value` 列 |
| `temperature_f` | 温度，华氏度 |
| `dew_point_f` | 露点温度，华氏度 |
| `humidity_pct` | 相对湿度，百分比 |
| `wind_direction` | 风向/风况文本 |
| `wind_speed_mph` | 风速，mph |
| `wind_gust_mph` | 阵风风速，mph |
| `pressure_in` | 气压，英寸汞柱 |
| `city` | 用户所在城市分组，`CT2` |
| `industry_code` | 行业代码，`C24` |
| `industry_name` | 行业名称 |

## 质量检查

- 期望行数：`70080`
- 实际行数：`70080`
- 负荷缺失值数量：`0`
- 负荷 0 值比例：`0.000000`
- 天气数值列缺失值数量：`{'temperature_f': 0, 'dew_point_f': 0, 'humidity_pct': 0, 'wind_speed_mph': 0, 'wind_gust_mph': 0, 'pressure_in': 0}`
- 天气分类列缺失值数量：`{'wind_direction': 0}`

## 选择原因

- `U141` 属于制造业，满足工业用户优先的要求。
- 2018-2019 两个完整自然年覆盖率为 100%。
- 负荷 0 值比例为 0，适合作为 15 分钟负荷+天气标准样例。
