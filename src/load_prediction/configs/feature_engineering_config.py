"""Feature engineering configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Feature construction options for tabular and sequence forecasters."""

    add_calendar: bool = True
    add_cyclical_time: bool = True
    add_item_id: bool = True
    lag_steps: tuple[int, ...] = (1, 2, 3, 4, 24, 96)
    rolling_windows: tuple[int, ...] = (4, 12, 24)
    correlation_threshold: float = 0.0
    known_covariates: tuple[str, ...] = ()
    external_feature_sets: tuple[str, ...] = ()

