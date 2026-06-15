"""Feature registry for external covariates and generated feature groups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    columns: tuple[str, ...]
    description: str = ""


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "weather_temperature": FeatureSpec(
        name="weather_temperature",
        columns=("temperature", "bsl_t", "brn_t", "zrh_t", "lug_t", "lau_t", "gen_t", "stg_t", "luz_t"),
        description="Temperature/weather covariates commonly used by load forecasting datasets.",
    ),
    "weather_humidity": FeatureSpec(
        name="weather_humidity",
        columns=("humidity",),
        description="Humidity covariates.",
    ),
    "production": FeatureSpec(
        name="production",
        columns=("production_intensity", "shift", "line_id"),
        description="Production and shift covariates.",
    ),
    "traffic": FeatureSpec(
        name="traffic",
        columns=("passenger_flow", "traffic", "occupancy"),
        description="Traffic, passenger flow, or occupancy covariates.",
    ),
}


def resolve_external_feature_columns(
    frame: pd.DataFrame,
    feature_sets: tuple[str, ...] = (),
    explicit_columns: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Resolve configured feature sets and explicit columns against a DataFrame."""

    resolved: list[str] = []
    for feature_set in feature_sets:
        spec = FEATURE_REGISTRY.get(feature_set)
        if spec is None:
            raise ValueError(
                f"Unknown feature set '{feature_set}'. "
                f"Available: {', '.join(sorted(FEATURE_REGISTRY))}"
            )
        resolved.extend([column for column in spec.columns if column in frame.columns])

    resolved.extend([column for column in explicit_columns if column in frame.columns])
    return tuple(dict.fromkeys(resolved))
