"""Similar-time prior features for load forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

US_FIXED_HOLIDAYS = {(1, 1), (7, 4), (12, 25)}


SIMILAR_TIME_FEATURE_COLUMNS = (
    "feature__similar_time__load_mean",
    "feature__similar_time__load_weighted_mean",
    "feature__similar_time__load_mean_minus_lag_96",
    "feature__similar_time__load_weighted_mean_minus_lag_96",
    "feature__similar_time__load_mean_minus_rolling_mean_96",
    "feature__similar_time__load_weighted_mean_minus_rolling_mean_96",
)


@dataclass
class SimilarTimeFeatureBuilder:
    """Build nearest historical similar-time aggregate load features.

    Similarity is computed from features available before the predicted point:
    calendar/weather covariates plus lagged and rolling historical load state.
    The target load of a candidate timestamp is only used after nearest
    neighbors are selected, as an aggregate prior feature.
    """

    top_k: int = 5
    lag_steps: tuple[int, ...] = (96, 192, 672)
    rolling_windows: tuple[int, ...] = (96, 672)
    weather_weight: float = 1.0
    calendar_weight: float = 1.0
    lag_weight: float = 1.0
    rolling_weight: float = 1.0
    candidate_lookback: int | None = None
    slot_tolerance_steps: int = 1
    restrict_weekend: bool = False
    restrict_holiday: bool = False
    restrict_extreme_weather: bool = False
    candidate_recent_days: int | None = None
    query_columns_: list[str] = field(default_factory=list)
    query_weights_: np.ndarray | None = None
    query_mean_: pd.Series | None = None
    query_std_: pd.Series | None = None
    item_reference_: dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit_transform(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_col: str,
        item_id_col: str,
        target_col: str,
        numeric_covariates: tuple[str, ...],
    ) -> pd.DataFrame:
        prepared = self._prepare_frame(
            frame,
            timestamp_col=timestamp_col,
            item_id_col=item_id_col,
            target_col=target_col,
            numeric_covariates=numeric_covariates,
        )
        query_columns = self._query_columns(numeric_covariates)
        self.query_columns_ = query_columns
        complete = prepared.dropna(subset=[*query_columns, target_col])
        if complete.empty:
            logger.warning("No complete rows available for similar-time feature fitting")
            self.query_mean_ = pd.Series(0.0, index=query_columns)
            self.query_std_ = pd.Series(1.0, index=query_columns)
            return self._with_default_features(frame.copy())

        self.query_mean_ = complete[query_columns].mean()
        std = complete[query_columns].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.query_std_ = std
        self.query_weights_ = self._query_weights(query_columns, numeric_covariates)
        self.item_reference_ = {
            str(item_id): item.sort_values(timestamp_col).reset_index(drop=True)
            for item_id, item in prepared.groupby(item_id_col, sort=False)
        }
        parts = []
        for _, item in prepared.groupby(item_id_col, sort=False):
            parts.append(self._transform_item_frame(item, timestamp_col, target_col))
        return pd.concat(parts, ignore_index=True)

    def transform_future_row(
        self,
        history: pd.DataFrame,
        *,
        timestamp: pd.Timestamp,
        item_id: str,
        timestamp_col: str,
        item_id_col: str,
        target_col: str,
        numeric_covariates: tuple[str, ...],
        known_covariates: dict[str, object],
    ) -> dict[str, float]:
        if not self.query_columns_ or self.query_mean_ is None or self.query_std_ is None:
            return self.default_features()
        row = self._future_query_row(
            history,
            timestamp=pd.Timestamp(timestamp),
            item_id=item_id,
            timestamp_col=timestamp_col,
            item_id_col=item_id_col,
            target_col=target_col,
            numeric_covariates=numeric_covariates,
            known_covariates=known_covariates,
        )
        reference = self.item_reference_.get(str(item_id))
        if reference is None or reference.empty:
            return self.default_features()
        candidates = reference[pd.to_datetime(reference[timestamp_col]) < pd.Timestamp(timestamp)]
        candidates = self._filter_candidates(row, candidates)
        return self._nearest_features(row, candidates, target_col)

    def default_features(self) -> dict[str, float]:
        return {column: 0.0 for column in SIMILAR_TIME_FEATURE_COLUMNS}

    def _with_default_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for column in SIMILAR_TIME_FEATURE_COLUMNS:
            prepared[column] = 0.0
        return prepared

    def _prepare_frame(
        self,
        frame: pd.DataFrame,
        *,
        timestamp_col: str,
        item_id_col: str,
        target_col: str,
        numeric_covariates: tuple[str, ...],
    ) -> pd.DataFrame:
        parts = []
        for item_id, group in frame.groupby(item_id_col, sort=False):
            item = group.sort_values(timestamp_col).copy()
            item[timestamp_col] = pd.to_datetime(item[timestamp_col])
            item["__similar_timestamp"] = item[timestamp_col]
            self._add_calendar_columns(item, timestamp_col)
            for covariate in numeric_covariates:
                item[f"__similar_weather__{covariate}"] = pd.to_numeric(
                    item.get(covariate, 0.0),
                    errors="coerce",
                )
            target = pd.to_numeric(item[target_col], errors="coerce")
            for lag in self.lag_steps:
                item[f"__similar_lag__{int(lag)}"] = target.shift(int(lag))
            shifted = target.shift(1)
            for window in self.rolling_windows:
                item[f"__similar_rolling_mean__{int(window)}"] = shifted.rolling(int(window)).mean()
            parts.append(item)
        return pd.concat(parts, ignore_index=True)

    def _transform_item_frame(
        self,
        item: pd.DataFrame,
        timestamp_col: str,
        target_col: str,
    ) -> pd.DataFrame:
        item = item.sort_values(timestamp_col).reset_index(drop=True).copy()
        feature_rows = []
        for index, row in item.iterrows():
            candidates = item.iloc[:index]
            candidates = self._filter_candidates(row, candidates)
            feature_rows.append(self._nearest_features(row, candidates, target_col))
        features = pd.DataFrame(feature_rows, index=item.index)
        for column in SIMILAR_TIME_FEATURE_COLUMNS:
            item[column] = features[column].to_numpy(dtype=float)
        return item

    def _nearest_features(
        self,
        query_row: pd.Series,
        candidates: pd.DataFrame,
        target_col: str,
    ) -> dict[str, float]:
        if candidates.empty:
            return self.default_features()
        query = query_row.reindex(self.query_columns_).astype(float)
        if query.isna().any():
            return self.default_features()
        usable = candidates.dropna(subset=[*self.query_columns_, target_col])
        if usable.empty:
            return self.default_features()
        candidate_values = usable[self.query_columns_].astype(float)
        scaled_query = ((query - self.query_mean_) / self.query_std_).to_numpy(dtype=float)
        scaled_candidates = ((candidate_values - self.query_mean_) / self.query_std_).to_numpy(dtype=float)
        weights = self.query_weights_
        if weights is None:
            weights = np.ones(len(self.query_columns_), dtype=float)
        distances = np.sqrt(np.sum(weights * (scaled_candidates - scaled_query) ** 2, axis=1))
        if len(distances) == 0:
            return self.default_features()
        top_k = max(1, min(int(self.top_k), len(distances)))
        nearest = np.argpartition(distances, top_k - 1)[:top_k]
        nearest = nearest[np.argsort(distances[nearest])]
        top_distances = distances[nearest]
        top_loads = pd.to_numeric(usable.iloc[nearest][target_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(top_loads)
        if not valid.any():
            return self.default_features()
        top_loads = top_loads[valid]
        top_distances = top_distances[valid]
        similarity_weights = np.exp(-0.5 * top_distances**2)
        if float(similarity_weights.sum()) <= 0:
            weighted_mean = float(np.mean(top_loads))
        else:
            weighted_mean = float(np.average(top_loads, weights=similarity_weights))
        load_mean = float(np.mean(top_loads))
        lag_96 = self._finite_value(query_row.get("__similar_lag__96"))
        rolling_mean_96 = self._finite_value(query_row.get("__similar_rolling_mean__96"))
        return {
            "feature__similar_time__load_mean": load_mean,
            "feature__similar_time__load_weighted_mean": weighted_mean,
            "feature__similar_time__load_mean_minus_lag_96": load_mean - lag_96,
            "feature__similar_time__load_weighted_mean_minus_lag_96": weighted_mean - lag_96,
            "feature__similar_time__load_mean_minus_rolling_mean_96": load_mean - rolling_mean_96,
            "feature__similar_time__load_weighted_mean_minus_rolling_mean_96": weighted_mean - rolling_mean_96,
        }

    def _finite_value(self, value: object) -> float:
        numeric_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if not np.isfinite(float(numeric_value)):
            return 0.0
        return float(numeric_value)

    def _future_query_row(
        self,
        history: pd.DataFrame,
        *,
        timestamp: pd.Timestamp,
        item_id: str,
        timestamp_col: str,
        item_id_col: str,
        target_col: str,
        numeric_covariates: tuple[str, ...],
        known_covariates: dict[str, object],
    ) -> pd.Series:
        row: dict[str, float | object] = {
            timestamp_col: pd.Timestamp(timestamp),
            item_id_col: item_id,
            "__similar_timestamp": pd.Timestamp(timestamp),
        }
        self._add_calendar_values(row, pd.Timestamp(timestamp))
        for covariate in numeric_covariates:
            row[f"__similar_weather__{covariate}"] = float(known_covariates.get(covariate, 0.0) or 0.0)
        item_history = history.sort_values(timestamp_col).copy()
        target = pd.to_numeric(item_history[target_col], errors="coerce").dropna()
        for lag in self.lag_steps:
            row[f"__similar_lag__{int(lag)}"] = float(target.iloc[-int(lag)]) if len(target) >= int(lag) else np.nan
        for window in self.rolling_windows:
            values = target.iloc[-int(window):]
            row[f"__similar_rolling_mean__{int(window)}"] = float(values.mean()) if not values.empty else np.nan
        return pd.Series(row)

    def _query_columns(self, numeric_covariates: tuple[str, ...]) -> list[str]:
        return [
            "__similar_time__sin_day",
            "__similar_time__cos_day",
            "__similar_time__sin_week",
            "__similar_time__cos_week",
            "__similar_calendar__is_weekend",
            *[f"__similar_weather__{covariate}" for covariate in numeric_covariates],
            *[f"__similar_lag__{int(lag)}" for lag in self.lag_steps],
            *[f"__similar_rolling_mean__{int(window)}" for window in self.rolling_windows],
        ]

    def _query_weights(self, query_columns: list[str], numeric_covariates: tuple[str, ...]) -> np.ndarray:
        weather_columns = {f"__similar_weather__{covariate}" for covariate in numeric_covariates}
        weights = []
        for column in query_columns:
            if column in weather_columns:
                weights.append(float(self.weather_weight))
            elif column.startswith("__similar_lag__"):
                weights.append(float(self.lag_weight))
            elif column.startswith("__similar_rolling_mean__"):
                weights.append(float(self.rolling_weight))
            else:
                weights.append(float(self.calendar_weight))
        return np.asarray(weights, dtype=float)

    def _add_calendar_columns(self, frame: pd.DataFrame, timestamp_col: str) -> None:
        ts = pd.to_datetime(frame[timestamp_col])
        minute_of_day = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy()
        frame["__similar_slot__minute_of_day"] = minute_of_day.astype(float)
        frame["__similar_time__sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440)
        frame["__similar_time__cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440)
        frame["__similar_time__sin_week"] = np.sin(
            2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
        )
        frame["__similar_time__cos_week"] = np.cos(
            2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
        )
        frame["__similar_calendar__is_weekend"] = (ts.dt.dayofweek >= 5).astype(float).to_numpy()
        frame["__similar_calendar__is_holiday"] = self._holiday_flags(ts)
        self._add_extreme_weather_column(frame)

    def _add_calendar_values(self, row: dict[str, float | object], timestamp: pd.Timestamp) -> None:
        minute_of_day = timestamp.hour * 60 + timestamp.minute
        row["__similar_slot__minute_of_day"] = float(minute_of_day)
        row["__similar_time__sin_day"] = float(np.sin(2 * np.pi * minute_of_day / 1440))
        row["__similar_time__cos_day"] = float(np.cos(2 * np.pi * minute_of_day / 1440))
        row["__similar_time__sin_week"] = float(
            np.sin(2 * np.pi * (timestamp.dayofweek * 1440 + minute_of_day) / (7 * 1440))
        )
        row["__similar_time__cos_week"] = float(
            np.cos(2 * np.pi * (timestamp.dayofweek * 1440 + minute_of_day) / (7 * 1440))
        )
        row["__similar_calendar__is_weekend"] = float(timestamp.dayofweek >= 5)
        row["__similar_calendar__is_holiday"] = float(self._is_us_holiday(timestamp))
        self._add_extreme_weather_value(row)

    def _filter_candidates(
        self,
        query_row: pd.Series,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        if candidates.empty:
            return candidates
        filtered = candidates
        if self.candidate_lookback is not None and len(filtered) > int(self.candidate_lookback):
            filtered = filtered.tail(int(self.candidate_lookback))
        filtered = self._filter_same_slot_candidates(query_row, filtered)
        filtered = self._filter_recent_candidates(query_row, filtered)
        filtered = self._filter_same_flag_candidates(
            query_row,
            filtered,
            "__similar_calendar__is_weekend",
            enabled=self.restrict_weekend,
        )
        filtered = self._filter_same_flag_candidates(
            query_row,
            filtered,
            "__similar_calendar__is_holiday",
            enabled=self.restrict_holiday,
        )
        filtered = self._filter_same_flag_candidates(
            query_row,
            filtered,
            "__similar_weather__is_extreme",
            enabled=self.restrict_extreme_weather,
        )
        return filtered

    def _filter_same_slot_candidates(
        self,
        query_row: pd.Series,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        if candidates.empty or "__similar_slot__minute_of_day" not in candidates.columns:
            return candidates
        query_slot = query_row.get("__similar_slot__minute_of_day")
        if pd.isna(query_slot):
            return candidates
        tolerance_minutes = max(0, int(self.slot_tolerance_steps)) * 15
        candidate_slots = pd.to_numeric(candidates["__similar_slot__minute_of_day"], errors="coerce")
        direct_distance = (candidate_slots - float(query_slot)).abs()
        circular_distance = np.minimum(direct_distance, 1440 - direct_distance)
        return candidates[circular_distance <= tolerance_minutes]

    def _filter_recent_candidates(
        self,
        query_row: pd.Series,
        candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.candidate_recent_days is None or candidates.empty:
            return candidates
        if "__similar_timestamp" not in candidates.columns:
            return candidates
        query_timestamp = query_row.get("__similar_timestamp")
        if pd.isna(query_timestamp):
            return candidates
        cutoff = pd.Timestamp(query_timestamp) - pd.Timedelta(days=int(self.candidate_recent_days))
        candidate_timestamps = pd.to_datetime(candidates["__similar_timestamp"], errors="coerce")
        recent = candidates[candidate_timestamps >= cutoff]
        return recent if not recent.empty else candidates

    def _filter_same_flag_candidates(
        self,
        query_row: pd.Series,
        candidates: pd.DataFrame,
        column: str,
        *,
        enabled: bool,
    ) -> pd.DataFrame:
        if not enabled or candidates.empty or column not in candidates.columns:
            return candidates
        query_flag = query_row.get(column)
        if pd.isna(query_flag):
            return candidates
        candidate_flags = pd.to_numeric(candidates[column], errors="coerce")
        filtered = candidates[candidate_flags == float(query_flag)]
        return filtered if not filtered.empty else candidates

    def _holiday_flags(self, timestamps: pd.Series) -> np.ndarray:
        return np.asarray([float(self._is_us_holiday(timestamp)) for timestamp in timestamps], dtype=float)

    def _is_us_holiday(self, timestamp: pd.Timestamp) -> bool:
        ts = pd.Timestamp(timestamp)
        if (ts.month, ts.day) in US_FIXED_HOLIDAYS:
            return True
        if ts.month == 11 and ts.dayofweek == 3 and 22 <= ts.day <= 28:
            return True
        return False

    def _add_extreme_weather_column(self, frame: pd.DataFrame) -> None:
        frame["__similar_weather__is_extreme"] = 0.0
        if "temperature_f" in frame.columns:
            temperature = pd.to_numeric(frame["temperature_f"], errors="coerce")
            frame["__similar_weather__is_extreme"] = (
                (temperature <= 32.0) | (temperature >= 90.0)
            ).astype(float)

    def _add_extreme_weather_value(self, row: dict[str, float | object]) -> None:
        temperature = pd.to_numeric(pd.Series([row.get("__similar_weather__temperature_f")]), errors="coerce").iloc[0]
        row["__similar_weather__is_extreme"] = float(
            np.isfinite(float(temperature)) and (float(temperature) <= 32.0 or float(temperature) >= 90.0)
        )
