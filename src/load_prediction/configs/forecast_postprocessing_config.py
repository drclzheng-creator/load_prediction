"""Forecast post-processing configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from load_prediction.constants import (
    DEFAULT_DISTRIBUTION_GRID_SIZE,
    DEFAULT_DISTRIBUTION_STD_WIDTH,
)


@dataclass(frozen=True)
class DistributionGridConfig:
    """Density-grid generation settings for probabilistic forecasts."""

    enabled: bool = True
    size: int = DEFAULT_DISTRIBUTION_GRID_SIZE
    std_width: float = DEFAULT_DISTRIBUTION_STD_WIDTH


@dataclass(frozen=True)
class ForecastPostprocessingConfig:
    """Forecast clipping, residual intervals, and distribution-grid settings."""

    clip_negative: bool = True
    add_residual_interval: bool = True
    interval_width: float = 1.96
    distribution_grid: DistributionGridConfig = field(default_factory=DistributionGridConfig)

    @property
    def add_distribution_grid(self) -> bool:
        return self.distribution_grid.enabled

    @property
    def distribution_grid_size(self) -> int:
        return self.distribution_grid.size

    @property
    def distribution_std_width(self) -> float:
        return self.distribution_grid.std_width

