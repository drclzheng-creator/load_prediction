"""Top-level load-forecast pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from load_prediction.configs.artifacts_config import ArtifactConfig
from load_prediction.configs.pipeline_config_factory import pipeline_config_from_dict
from load_prediction.configs.data_cleaning_config import DataCleaningConfig
from load_prediction.configs.data_source_config import DataSourceConfig
from load_prediction.configs.forecast_postprocessing_config import ForecastPostprocessingConfig
from load_prediction.configs.forecast_profile_config import (
    FORECAST_PROFILE_PRESETS,
    ForecastProfileConfig,
)
from load_prediction.configs.model_spec_config import ModelSpecConfig
from load_prediction.configs.validation_config import ValidationConfig
from load_prediction.constants import FORECAST_PROFILE_DAY_AHEAD


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level load-forecast pipeline configuration."""

    data: DataSourceConfig = field(default_factory=DataSourceConfig)
    cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    model: ModelSpecConfig = field(default_factory=ModelSpecConfig)
    evaluation: ValidationConfig = field(default_factory=ValidationConfig)
    postprocess: ForecastPostprocessingConfig = field(default_factory=ForecastPostprocessingConfig)
    output: ArtifactConfig = field(default_factory=ArtifactConfig)
    scale: ForecastProfileConfig = field(
        default_factory=lambda: FORECAST_PROFILE_PRESETS[FORECAST_PROFILE_DAY_AHEAD]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return pipeline_config_from_dict(raw)

