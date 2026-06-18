"""Config factory helpers for YAML-friendly dictionaries."""

from __future__ import annotations

from typing import Any

from load_prediction.configs.artifacts_config import ArtifactConfig
from load_prediction.configs.data_cleaning_config import DataCleaningConfig
from load_prediction.configs.data_source_config import DataSourceConfig
from load_prediction.configs.feature_engineering_config import FeatureEngineeringConfig
from load_prediction.configs.forecast_postprocessing_config import (
    DistributionGridConfig,
    ForecastPostprocessingConfig,
)
from load_prediction.configs.forecast_profile_config import (
    ForecastProfileConfig,
    forecast_profile_preset,
)
from load_prediction.configs.model_spec_config import ModelSpecConfig
from load_prediction.configs.validation_config import ValidationConfig
from load_prediction.constants import FORECAST_PROFILE_DAY_AHEAD


def pipeline_config_from_dict(raw: dict[str, Any]):
    """Build a strongly-shaped pipeline config from a YAML-friendly dictionary."""

    from load_prediction.configs.pipeline_config import PipelineConfig

    profile_name = (
        raw.get("forecast_profile")
        or raw.get("scale_name")
        or raw.get("scale", {}).get("name")
        or FORECAST_PROFILE_DAY_AHEAD
    )
    profile = forecast_profile_preset(profile_name)
    feature_raw = raw.get("feature") or {}

    if "scale" in raw:
        scale_raw = raw["scale"] or {}
        feature_raw = {**feature_raw, **(scale_raw.get("feature") or {})}
        feature = _feature_config({**profile.feature.__dict__, **feature_raw})
        profile = ForecastProfileConfig(
            name=scale_raw.get("name", profile.name),
            freq=scale_raw.get("freq", profile.freq),
            prediction_length=scale_raw.get(
                "prediction_length",
                scale_raw.get("horizon_steps", profile.prediction_length),
            ),
            feature=feature,
            description=scale_raw.get("description", profile.description),
        )
    elif feature_raw:
        profile = ForecastProfileConfig(
            name=profile.name,
            freq=profile.freq,
            prediction_length=profile.prediction_length,
            feature=_feature_config({**profile.feature.__dict__, **feature_raw}),
            description=profile.description,
        )

    return PipelineConfig(
        data=_data_source_config(raw.get("data", {})),
        cleaning=DataCleaningConfig(**raw.get("cleaning", {})),
        model=ModelSpecConfig(**raw.get("model", {})),
        evaluation=_validation_config(raw.get("evaluation", {})),
        postprocess=_forecast_postprocessing_config(raw.get("postprocess", {})),
        output=ArtifactConfig(**raw.get("output", {})),
        scale=profile,
    )


def _data_source_config(raw: dict[str, Any]) -> DataSourceConfig:
    values = dict(raw)
    for key in ("covariate_cols", "feature_sets"):
        if key in values:
            values[key] = tuple(values[key] or ())
    return DataSourceConfig(**values)


def _feature_config(raw: dict[str, Any]) -> FeatureEngineeringConfig:
    values = dict(raw)
    for key in (
        "lag_steps",
        "rolling_windows",
        "known_covariates",
        "external_feature_sets",
        "similar_time_lag_steps",
        "similar_time_rolling_windows",
    ):
        if key in values:
            values[key] = tuple(values[key] or ())
    return FeatureEngineeringConfig(**values)


def _validation_config(raw: dict[str, Any]) -> ValidationConfig:
    values = dict(raw)
    if "metrics" in values:
        values["metrics"] = tuple(values["metrics"] or ())
    return ValidationConfig(**values)


def _forecast_postprocessing_config(raw: dict[str, Any]) -> ForecastPostprocessingConfig:
    values = dict(raw)
    distribution_grid_raw = values.pop("distribution_grid", None)
    legacy_enabled = values.pop("add_distribution_grid", None)
    legacy_size = values.pop("distribution_grid_size", None)
    legacy_std_width = values.pop("distribution_std_width", None)

    if distribution_grid_raw is None:
        distribution_grid_raw = {}
    distribution_grid_values = dict(distribution_grid_raw)
    if legacy_enabled is not None:
        distribution_grid_values["enabled"] = legacy_enabled
    if legacy_size is not None:
        distribution_grid_values["size"] = legacy_size
    if legacy_std_width is not None:
        distribution_grid_values["std_width"] = legacy_std_width

    if distribution_grid_values:
        values["distribution_grid"] = DistributionGridConfig(**distribution_grid_values)
    return ForecastPostprocessingConfig(**values)
