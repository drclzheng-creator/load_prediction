"""Data cleaning policy configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataCleaningConfig:
    """Data cleaning, resampling, missing-value, and outlier policy."""

    resample: bool = True
    upsample_strategy: str = "warn_interpolate"
    missing_strategy: str = "hybrid"
    short_gap_limit: int = 3
    outlier_method: str = "iqr"
    outlier_iqr_multiplier: float = 3.0
    non_negative_target: bool = True

