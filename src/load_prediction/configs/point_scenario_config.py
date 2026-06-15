"""Point scenario generation configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PointScenarioConfig:
    """Configuration for point-wise sampling and scenario reduction."""

    num_scenarios: int = 8
    num_samples: int = 10000
    random_state: int | None = 42
    clip_min: float | None = 0.0
    kmeans_n_init: int = 2
    kmeans_max_iter: int = 100
    kmeans_fit_samples: int = 1000
    values_col: str = "point_scenario_values"
    probabilities_col: str = "point_scenario_probabilities"

