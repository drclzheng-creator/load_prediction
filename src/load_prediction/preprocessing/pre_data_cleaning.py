"""Cleaning and regularization for load time series."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from load_prediction.configs import DataCleaningConfig
from load_prediction.data.data_schema import TimeSeriesDataset

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesCleaner:
    config: DataCleaningConfig

    def fit_transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        frame = data.frame.copy()
        ts_col = data.timestamp_col
        item_col = data.item_id_col
        target_col = data.target_col

        cleaned_parts: list[pd.DataFrame] = []
        for item_id, group in frame.groupby(item_col, sort=False):
            item_frame = group.sort_values(ts_col).copy()
            logger.info("Cleaning item_id=%s rows=%s", item_id, len(item_frame))
            item_frame = self._collapse_duplicate_timestamps(item_frame, ts_col, target_col)
            if self.config.resample:
                item_frame = self._resample_item(item_frame, data, str(item_id))
            item_frame[target_col] = self._clip_outliers(item_frame[target_col])
            item_frame[target_col] = self._fill_missing_target(item_frame, ts_col, target_col)
            if self.config.non_negative_target:
                item_frame[target_col] = item_frame[target_col].clip(lower=0)
            cleaned_parts.append(item_frame)

        cleaned = pd.concat(cleaned_parts, ignore_index=True)
        return data.with_frame(cleaned)

    def _collapse_duplicate_timestamps(
        self,
        frame: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        if not frame.duplicated(timestamp_col).any():
            return frame

        logger.warning("Found duplicate timestamps; numeric columns will be averaged.")
        numeric_cols = frame.select_dtypes(include=np.number).columns.tolist()
        aggregations = {col: "mean" for col in numeric_cols}
        for col in frame.columns:
            if col not in aggregations and col != timestamp_col:
                aggregations[col] = "first"
        return frame.groupby(timestamp_col, as_index=False).agg(aggregations)

    def _resample_item(
        self,
        frame: pd.DataFrame,
        data: TimeSeriesDataset,
        item_id: str,
    ) -> pd.DataFrame:
        freq = data.freq or pd.infer_freq(frame[data.timestamp_col])
        if not freq:
            logger.warning("Cannot infer frequency; skipping resample for item_id=%s", item_id)
            return frame

        direction = self._resample_direction(frame, data.timestamp_col, freq)
        logger.info(
            "Resampling item_id=%s to freq=%s direction=%s",
            item_id,
            freq,
            direction,
        )

        if direction == "upsample":
            self._handle_upsample_policy(item_id, freq)

        item_col = data.item_id_col
        numeric_cols = frame.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = [
            col
            for col in frame.columns
            if col not in numeric_cols and col not in {data.timestamp_col, item_col}
        ]

        indexed = frame.set_index(data.timestamp_col).sort_index()
        resampled = indexed[numeric_cols].resample(freq).mean()
        for col in categorical_cols:
            resampled[col] = indexed[col].resample(freq).ffill()
        if direction == "upsample" and self.config.upsample_strategy == "ffill":
            resampled[numeric_cols] = resampled[numeric_cols].ffill()
        resampled[item_col] = item_id
        return resampled.reset_index()

    def _clip_outliers(self, series: pd.Series) -> pd.Series:
        if self.config.outlier_method != "iqr":
            logger.info("Skipping outlier clipping because method=%s", self.config.outlier_method)
            return series

        valid = series.dropna()
        if valid.empty:
            return series
        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return series
        lower = q1 - self.config.outlier_iqr_multiplier * iqr
        upper = q3 + self.config.outlier_iqr_multiplier * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count:
            logger.warning("Clipping %s outliers by IQR bounds [%.4f, %.4f]", outlier_count, lower, upper)
        return series.clip(lower=lower, upper=upper)

    def _fill_missing_target(
        self,
        frame: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
    ) -> pd.Series:
        target = frame[target_col].copy()
        if not target.isna().any():
            return target

        missing_count = int(target.isna().sum())
        logger.warning("Filling %s missing target values with strategy=%s", missing_count, self.config.missing_strategy)
        if self.config.missing_strategy == "mean":
            return target.fillna(target.mean())

        interpolated = target.interpolate(
            method="linear",
            limit=self.config.short_gap_limit,
            limit_direction="both",
        )
        if not interpolated.isna().any():
            return interpolated

        helper = frame[[timestamp_col]].copy()
        helper[target_col] = interpolated
        helper["_slot"] = pd.to_datetime(helper[timestamp_col]).dt.strftime("%H:%M")
        slot_median = helper.groupby("_slot")[target_col].transform("median")
        filled = helper[target_col].fillna(slot_median)
        return filled.fillna(target.median()).fillna(0)

    def _resample_direction(
        self,
        frame: pd.DataFrame,
        timestamp_col: str,
        target_freq: str,
    ) -> str:
        source_freq = pd.infer_freq(frame[timestamp_col])
        if not source_freq:
            return "unknown"

        try:
            source_delta = pd.Timedelta(to_offset(source_freq))
            target_delta = pd.Timedelta(to_offset(target_freq))
        except ValueError:
            logger.warning(
                "Cannot compare source_freq=%s and target_freq=%s; treating as unknown.",
                source_freq,
                target_freq,
            )
            return "unknown"

        if target_delta < source_delta:
            return "upsample"
        if target_delta > source_delta:
            return "downsample"
        return "same"

    def _handle_upsample_policy(self, item_id: str, freq: str) -> None:
        strategy = self.config.upsample_strategy
        message = (
            f"Upsampling item_id={item_id} to freq={freq}. This creates synthetic "
            "higher-frequency values from lower-frequency observations."
        )
        if strategy == "error":
            raise ValueError(message)
        if strategy == "warn_interpolate":
            logger.warning("%s Missing values will be filled by missing_strategy=%s.", message, self.config.missing_strategy)
            return
        if strategy in {"interpolate", "ffill"}:
            logger.info("Upsampling allowed by strategy=%s for item_id=%s", strategy, item_id)
            return
        raise ValueError(
            "Unsupported upsample_strategy "
            f"'{strategy}'. Use error, warn_interpolate, interpolate, or ffill."
        )
