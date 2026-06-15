"""Data source and schema configuration."""

from __future__ import annotations

from dataclasses import dataclass

from load_prediction.constants import (
    DEFAULT_ITEM_ID_COL,
    DEFAULT_TARGET_COL,
    DEFAULT_TIMESTAMP_COL,
)


@dataclass(frozen=True)
class DataSourceConfig:
    """Input data source, column mapping, and optional time filters."""

    path: str | None = None
    timestamp_col: str = DEFAULT_TIMESTAMP_COL
    target_col: str = DEFAULT_TARGET_COL
    item_id_col: str = DEFAULT_ITEM_ID_COL
    freq: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    covariate_cols: tuple[str, ...] = ()
    feature_sets: tuple[str, ...] = ()

