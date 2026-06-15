"""Data loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import pandas as pd

from load_prediction.configs import DataSourceConfig
from load_prediction.constants import DEFAULT_ITEM_ID
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features import resolve_external_feature_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CSVLoadDataLoader:
    """Load long-format CSV data into the normalized dataset shape."""

    config: DataSourceConfig

    def load(self, path: str | Path) -> TimeSeriesDataset:
        input_path = Path(path)
        logger.info("Loading CSV dataset from %s", input_path)
        frame = pd.read_csv(input_path)
        logger.info("Loaded CSV rows=%s columns=%s", len(frame), list(frame.columns))
        rename_map = {
            self.config.timestamp_col: "timestamp",
            self.config.target_col: "target",
        }
        if self.config.item_id_col in frame.columns:
            rename_map[self.config.item_id_col] = "item_id"

        normalized = frame.rename(columns=rename_map)
        if "item_id" not in normalized.columns:
            normalized["item_id"] = DEFAULT_ITEM_ID
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
        if self.config.start_time is not None:
            start_time = pd.Timestamp(self.config.start_time)
            before_count = len(normalized)
            normalized = normalized[normalized["timestamp"] >= start_time].copy()
            logger.info(
                "Filtered dataset by start_time=%s rows_before=%s rows_after=%s",
                start_time,
                before_count,
                len(normalized),
            )
            if normalized.empty:
                raise ValueError(f"No rows remain after applying start_time={start_time}")
        if self.config.end_time is not None:
            end_time = pd.Timestamp(self.config.end_time)
            before_count = len(normalized)
            normalized = normalized[normalized["timestamp"] <= end_time].copy()
            logger.info(
                "Filtered dataset by end_time=%s rows_before=%s rows_after=%s",
                end_time,
                before_count,
                len(normalized),
            )
            if normalized.empty:
                raise ValueError(f"No rows remain after applying end_time={end_time}")

        external_columns = resolve_external_feature_columns(
            normalized,
            feature_sets=self.config.feature_sets,
            explicit_columns=self.config.covariate_cols,
        )
        selected = ["timestamp", "item_id", "target", *external_columns]
        missing_requested = [
            col for col in self.config.covariate_cols if col not in normalized.columns
        ]
        if missing_requested:
            logger.warning("Requested covariate columns not found and skipped: %s", missing_requested)
        logger.info("Selected dataset columns=%s", selected)
        return TimeSeriesDataset(
            normalized[selected],
            timestamp_col="timestamp",
            target_col="target",
            item_id_col="item_id",
            freq=self.config.freq,
        )
