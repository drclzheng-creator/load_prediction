"""Domain dataset object for load forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from load_prediction.constants import DEFAULT_ITEM_ID, DEFAULT_ITEM_ID_COL


@dataclass
class TimeSeriesDataset:
    """Normalized long-format time-series data.

    Required columns are timestamp, item_id, and target. Extra columns are
    treated as covariates by config when selected.
    """

    frame: pd.DataFrame
    timestamp_col: str
    target_col: str
    item_id_col: str = DEFAULT_ITEM_ID_COL
    freq: str | None = None

    def __post_init__(self) -> None:
        missing = {
            self.timestamp_col,
            self.target_col,
            self.item_id_col,
        } - set(self.frame.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

        normalized = self.frame.copy()
        normalized[self.timestamp_col] = pd.to_datetime(normalized[self.timestamp_col])
        normalized[self.item_id_col] = normalized[self.item_id_col].fillna(DEFAULT_ITEM_ID)
        normalized = normalized.sort_values([self.item_id_col, self.timestamp_col])
        self.frame = normalized.reset_index(drop=True)

    @property
    def item_ids(self) -> list[str]:
        return sorted(self.frame[self.item_id_col].astype(str).unique().tolist())

    @property
    def covariate_columns(self) -> list[str]:
        required = {self.timestamp_col, self.item_id_col, self.target_col}
        return [col for col in self.frame.columns if col not in required]

    def copy(self) -> "TimeSeriesDataset":
        return TimeSeriesDataset(
            frame=self.frame.copy(),
            timestamp_col=self.timestamp_col,
            target_col=self.target_col,
            item_id_col=self.item_id_col,
            freq=self.freq,
        )

    def with_frame(self, frame: pd.DataFrame) -> "TimeSeriesDataset":
        return TimeSeriesDataset(
            frame=frame,
            timestamp_col=self.timestamp_col,
            target_col=self.target_col,
            item_id_col=self.item_id_col,
            freq=self.freq,
        )

    def split_holdout(self, holdout_length: int) -> tuple["TimeSeriesDataset", "TimeSeriesDataset"]:
        if holdout_length <= 0:
            raise ValueError("holdout_length must be positive")

        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        for _, group in self.frame.groupby(self.item_id_col, sort=False):
            if len(group) <= holdout_length:
                raise ValueError(
                    f"Not enough rows for holdout_length={holdout_length}; "
                    f"item has only {len(group)} rows."
                )
            train_parts.append(group.iloc[:-holdout_length])
            test_parts.append(group.iloc[-holdout_length:])

        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        return self.with_frame(train), self.with_frame(test)

    def split_by_ratio(self, train_ratio: float) -> tuple["TimeSeriesDataset", "TimeSeriesDataset"]:
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")

        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        for _, group in self.frame.groupby(self.item_id_col, sort=False):
            split_index = int(len(group) * train_ratio)
            if split_index <= 0 or split_index >= len(group):
                raise ValueError(
                    f"Not enough rows for train_ratio={train_ratio}; "
                    f"item has only {len(group)} rows."
                )
            train_parts.append(group.iloc[:split_index])
            test_parts.append(group.iloc[split_index:])

        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        return self.with_frame(train), self.with_frame(test)
