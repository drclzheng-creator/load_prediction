"""Forecast validation and metric configuration."""

from __future__ import annotations

from dataclasses import dataclass

from load_prediction.constants import DEFAULT_POINT_METRICS


@dataclass(frozen=True)
class ValidationConfig:
    """Train/test split policy and requested evaluation metrics."""

    split_strategy: str = "ratio"
    train_ratio: float | None = 0.7
    holdout_length: int | None = None
    metrics: tuple[str, ...] = DEFAULT_POINT_METRICS

