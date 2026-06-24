"""Feature engineering configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Feature construction options for tabular and sequence forecasters."""

    add_calendar: bool = True
    add_cyclical_time: bool = True
    add_chinese_calendar: bool = True
    add_item_id: bool = True
    lag_steps: tuple[int, ...] = (1, 2, 3, 4, 24, 96)
    rolling_windows: tuple[int, ...] = (4, 12, 24)
    correlation_threshold: float = 0.0
    known_covariates: tuple[str, ...] = ()
    external_feature_sets: tuple[str, ...] = ()
    add_similar_time_features: bool = False
    similar_time_top_k: int = 5
    similar_time_candidate_lookback: int | None = None
    similar_time_slot_tolerance_steps: int = 1
    similar_time_lag_steps: tuple[int, ...] = (96, 192, 672)
    similar_time_rolling_windows: tuple[int, ...] = (96, 672)
    similar_time_weather_weight: float = 1.0
    similar_time_calendar_weight: float = 1.0
    similar_time_lag_weight: float = 1.0
    similar_time_rolling_weight: float = 1.0
    similar_time_restrict_weekend: bool = False
    similar_time_restrict_holiday: bool = False
    similar_time_restrict_extreme_weather: bool = False
    similar_time_candidate_recent_days: int | None = None
