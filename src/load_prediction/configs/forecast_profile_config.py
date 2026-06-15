"""Forecast horizon and frequency profile presets."""

from __future__ import annotations

from dataclasses import dataclass

from load_prediction.configs.feature_engineering_config import FeatureEngineeringConfig
from load_prediction.constants import (
    FORECAST_PROFILE_DAY_AHEAD,
    FORECAST_PROFILE_HOURLY,
    FORECAST_PROFILE_MEDIUM_TERM,
    FORECAST_PROFILE_SHORT_TERM_4H,
    FORECAST_PROFILE_ULTRA_SHORT,
)


@dataclass(frozen=True)
class ForecastProfileConfig:
    """One forecasting profile: horizon, frequency, and default features."""

    name: str
    freq: str
    prediction_length: int
    feature: FeatureEngineeringConfig
    description: str = ""

    @property
    def horizon_steps(self) -> int:
        return self.prediction_length


FORECAST_PROFILE_PRESETS: dict[str, ForecastProfileConfig] = {
    FORECAST_PROFILE_ULTRA_SHORT: ForecastProfileConfig(
        name=FORECAST_PROFILE_ULTRA_SHORT,
        freq="5min",
        prediction_length=12,
        description="5-minute ultra-short-term forecast for real-time dispatch.",
        feature=FeatureEngineeringConfig(
            lag_steps=(1, 2, 3, 6, 12, 24, 48, 288),
            rolling_windows=(3, 6, 12, 24),
        ),
    ),
    FORECAST_PROFILE_SHORT_TERM_4H: ForecastProfileConfig(
        name=FORECAST_PROFILE_SHORT_TERM_4H,
        freq="15min",
        prediction_length=16,
        description="15-minute short-term 4-hour forecast.",
        feature=FeatureEngineeringConfig(
            lag_steps=(1, 2, 4, 8, 16, 32, 96, 192, 672),
            rolling_windows=(4, 16, 96),
        ),
    ),
    FORECAST_PROFILE_DAY_AHEAD: ForecastProfileConfig(
        name=FORECAST_PROFILE_DAY_AHEAD,
        freq="15min",
        prediction_length=96,
        description="15-minute day-ahead forecast.",
        feature=FeatureEngineeringConfig(
            lag_steps=(1, 2, 4, 8, 16, 32, 96, 192, 672),
            rolling_windows=(4, 16, 96),
        ),
    ),
    FORECAST_PROFILE_HOURLY: ForecastProfileConfig(
        name=FORECAST_PROFILE_HOURLY,
        freq="1h",
        prediction_length=24,
        description="Hourly forecast for intraday and next-day decisions.",
        feature=FeatureEngineeringConfig(
            lag_steps=(1, 2, 3, 6, 12, 24, 48, 168),
            rolling_windows=(3, 6, 24),
        ),
    ),
    FORECAST_PROFILE_MEDIUM_TERM: ForecastProfileConfig(
        name=FORECAST_PROFILE_MEDIUM_TERM,
        freq="1D",
        prediction_length=30,
        description="Daily medium-term forecast for weekly/monthly planning.",
        feature=FeatureEngineeringConfig(
            lag_steps=(1, 2, 3, 7, 14, 30, 60),
            rolling_windows=(3, 7, 14, 30),
        ),
    ),
}

SUPPORTED_FORECAST_PROFILES = frozenset(FORECAST_PROFILE_PRESETS)


def forecast_profile_preset(name: str) -> ForecastProfileConfig:
    try:
        return FORECAST_PROFILE_PRESETS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(FORECAST_PROFILE_PRESETS))
        raise ValueError(f"Unsupported forecast profile '{name}'. Supported: {supported}") from exc

