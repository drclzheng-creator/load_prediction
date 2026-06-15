"""Feature engineering for recursive load forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np
import pandas as pd

from load_prediction.configs import FeatureEngineeringConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features.feature_registry import resolve_external_feature_columns

logger = logging.getLogger(__name__)


@dataclass
class FeatureBuilder:
    """Build train and future inference features.

    The builder is stateful because one-hot item columns and final feature
    ordering must stay stable between training and recursive prediction.
    """

    config: FeatureEngineeringConfig
    feature_columns_: list[str] = field(default_factory=list)
    selected_columns_: list[str] = field(default_factory=list)
    item_ids_: list[str] = field(default_factory=list)
    covariate_names_: tuple[str, ...] = ()
    numeric_covariates_: tuple[str, ...] = ()
    categorical_covariates_: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def fit_transform(self, data: TimeSeriesDataset) -> tuple[pd.DataFrame, pd.Series]:
        logger.info(
            "Building features for rows=%s items=%s freq=%s",
            len(data.frame),
            len(data.item_ids),
            data.freq,
        )
        self.item_ids_ = data.item_ids
        self.covariate_names_ = ()
        self.numeric_covariates_ = ()
        self.categorical_covariates_ = {}
        self._fit_covariate_types(data.frame)
        frame = self._build_lagged_frame(data.frame, data)
        frame = frame.dropna(subset=[data.target_col])

        feature_columns = [col for col in frame.columns if col.startswith("feature__")]
        feature_columns = [
            col
            for col in feature_columns
            if pd.to_numeric(frame[col], errors="coerce").notna().any()
        ]
        supervised = frame.dropna(subset=feature_columns + [data.target_col]).copy()
        if supervised.empty:
            raise ValueError(
                "No supervised rows remain after feature construction. "
                "Reduce lag_steps/rolling_windows or provide more history."
            )

        self.feature_columns_ = feature_columns
        self.selected_columns_ = self._select_by_correlation(
            supervised,
            feature_columns,
            data.target_col,
        )
        logger.info(
            "Feature matrix ready supervised_rows=%s total_features=%s selected_features=%s",
            len(supervised),
            len(feature_columns),
            len(self.selected_columns_),
        )
        return supervised[self.selected_columns_], supervised[data.target_col]

    def transform_future_row(
        self,
        history: pd.DataFrame,
        data: TimeSeriesDataset,
        timestamp: pd.Timestamp,
        item_id: str,
        known_covariates: dict[str, object] | None = None,
    ) -> pd.DataFrame:
        if not self.selected_columns_:
            raise RuntimeError("FeatureBuilder must be fitted before transform_future_row")

        row: dict[str, float] = {}
        row.update(self._calendar_features(pd.Timestamp(timestamp)))
        row.update(self._item_features(str(item_id)))
        row.update(self._known_covariate_features(known_covariates or {}))
        row.update(self._history_features(history, data.target_col))
        return pd.DataFrame([{col: row.get(col, 0.0) for col in self.selected_columns_}])

    def _build_lagged_frame(self, frame: pd.DataFrame, data: TimeSeriesDataset) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for item_id, group in frame.groupby(data.item_id_col, sort=False):
            item = group.sort_values(data.timestamp_col).copy()
            item_features = self._calendar_feature_frame(pd.to_datetime(item[data.timestamp_col]))

            if self.config.add_item_id:
                for known_item in self.item_ids_:
                    item_features[f"feature__item__{known_item}"] = float(str(item_id) == known_item)

            covariates = resolve_external_feature_columns(
                item,
                feature_sets=self.config.external_feature_sets,
                explicit_columns=self.config.known_covariates,
            )
            self.covariate_names_ = tuple(dict.fromkeys((*self.covariate_names_, *covariates)))
            for covariate in covariates:
                if covariate in self.numeric_covariates_ and covariate in item.columns:
                    item_features[f"feature__cov__{covariate}"] = pd.to_numeric(
                        item[covariate],
                        errors="coerce",
                    )
                elif covariate in self.categorical_covariates_ and covariate in item.columns:
                    normalized = item[covariate].astype("string").fillna("__missing__")
                    for category in self.categorical_covariates_[covariate]:
                        item_features[
                            f"feature__cov__{covariate}__category__{category}"
                        ] = (normalized == category).astype(float).to_numpy()

            target = pd.to_numeric(item[data.target_col], errors="coerce")
            for lag in self.config.lag_steps:
                item_features[f"feature__lag__{lag}"] = target.shift(lag)
            for window in self.config.rolling_windows:
                shifted = target.shift(1)
                item_features[f"feature__rolling_mean__{window}"] = shifted.rolling(window).mean()
                item_features[f"feature__rolling_std__{window}"] = (
                    shifted.rolling(window).std().fillna(0)
                )

            item_features[data.target_col] = target.to_numpy()
            item_features[data.timestamp_col] = item[data.timestamp_col].to_numpy()
            item_features[data.item_id_col] = item[data.item_id_col].to_numpy()
            parts.append(item_features)

        return pd.concat(parts, ignore_index=True)

    def _calendar_feature_frame(self, timestamps: pd.Series) -> pd.DataFrame:
        ts = pd.to_datetime(timestamps)
        features = pd.DataFrame(index=range(len(ts)))
        if not self.config.add_calendar:
            return features

        features["feature__calendar__dayofweek"] = ts.dt.dayofweek.to_numpy()
        features["feature__calendar__is_weekend"] = (ts.dt.dayofweek >= 5).astype(int).to_numpy()
        features["feature__calendar__year"] = ts.dt.year.to_numpy()
        features["feature__calendar__month"] = ts.dt.month.to_numpy()
        features["feature__calendar__day"] = ts.dt.day.to_numpy()
        features["feature__calendar__quarter"] = ts.dt.quarter.to_numpy()
        features["feature__calendar__dayofyear"] = ts.dt.dayofyear.to_numpy()
        features["feature__calendar__hour"] = ts.dt.hour.to_numpy()
        features["feature__calendar__minute"] = ts.dt.minute.to_numpy()

        if self.config.add_cyclical_time:
            minute_of_day = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy()
            features["feature__time__sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440)
            features["feature__time__cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440)
            features["feature__time__sin_week"] = np.sin(
                2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
            )
            features["feature__time__cos_week"] = np.cos(
                2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
            )

        return features

    def _calendar_features(self, timestamp: pd.Timestamp) -> dict[str, float]:
        if not self.config.add_calendar:
            return {}

        minute_of_day = timestamp.hour * 60 + timestamp.minute
        features = {
            "feature__calendar__dayofweek": float(timestamp.dayofweek),
            "feature__calendar__is_weekend": float(timestamp.dayofweek >= 5),
            "feature__calendar__year": float(timestamp.year),
            "feature__calendar__month": float(timestamp.month),
            "feature__calendar__day": float(timestamp.day),
            "feature__calendar__quarter": float(timestamp.quarter),
            "feature__calendar__dayofyear": float(timestamp.dayofyear),
            "feature__calendar__hour": float(timestamp.hour),
            "feature__calendar__minute": float(timestamp.minute),
        }
        if self.config.add_cyclical_time:
            features.update(
                {
                    "feature__time__sin_day": float(np.sin(2 * np.pi * minute_of_day / 1440)),
                    "feature__time__cos_day": float(np.cos(2 * np.pi * minute_of_day / 1440)),
                    "feature__time__sin_week": float(
                        np.sin(2 * np.pi * (timestamp.dayofweek * 1440 + minute_of_day) / (7 * 1440))
                    ),
                    "feature__time__cos_week": float(
                        np.cos(2 * np.pi * (timestamp.dayofweek * 1440 + minute_of_day) / (7 * 1440))
                    ),
                }
            )
        return features

    def _item_features(self, item_id: str) -> dict[str, float]:
        if not self.config.add_item_id:
            return {}
        return {f"feature__item__{known_item}": float(item_id == known_item) for known_item in self.item_ids_}

    def _known_covariate_features(self, known_covariates: dict[str, object]) -> dict[str, float]:
        features = {
            f"feature__cov__{name}": float(known_covariates.get(name, 0.0) or 0.0)
            for name in self.numeric_covariates_
        }
        for name, categories in self.categorical_covariates_.items():
            value = known_covariates.get(name, "__missing__")
            category_value = "__missing__" if pd.isna(value) else str(value)
            for category in categories:
                features[f"feature__cov__{name}__category__{category}"] = float(
                    category_value == category
                )
        return features

    def _fit_covariate_types(self, frame: pd.DataFrame) -> None:
        covariates = resolve_external_feature_columns(
            frame,
            feature_sets=self.config.external_feature_sets,
            explicit_columns=self.config.known_covariates,
        )
        numeric_covariates: list[str] = []
        categorical_covariates: dict[str, tuple[str, ...]] = {}

        for covariate in covariates:
            values = frame[covariate]
            numeric = pd.to_numeric(values, errors="coerce")
            if numeric.notna().any() and numeric.notna().mean() >= 0.95:
                numeric_covariates.append(covariate)
                continue

            categories = (
                values.astype("string")
                .fillna("__missing__")
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            categorical_covariates[covariate] = tuple(str(category) for category in categories)

        self.numeric_covariates_ = tuple(numeric_covariates)
        self.categorical_covariates_ = categorical_covariates
        logger.info(
            "Resolved covariates numeric=%s categorical=%s",
            self.numeric_covariates_,
            {name: len(categories) for name, categories in self.categorical_covariates_.items()},
        )

    def _history_features(self, history: pd.DataFrame, target_col: str) -> dict[str, float]:
        target = pd.to_numeric(history[target_col], errors="coerce").dropna()
        features: dict[str, float] = {}
        for lag in self.config.lag_steps:
            features[f"feature__lag__{lag}"] = float(target.iloc[-lag]) if len(target) >= lag else 0.0
        for window in self.config.rolling_windows:
            window_values = target.iloc[-window:]
            features[f"feature__rolling_mean__{window}"] = (
                float(window_values.mean()) if not window_values.empty else 0.0
            )
            features[f"feature__rolling_std__{window}"] = (
                float(window_values.std(ddof=1)) if len(window_values) > 1 else 0.0
            )
        return features

    def _select_by_correlation(
        self,
        frame: pd.DataFrame,
        feature_columns: list[str],
        target_col: str,
    ) -> list[str]:
        threshold = self.config.correlation_threshold
        if threshold <= 0:
            logger.info("Correlation feature selection disabled")
            return feature_columns

        selected: list[str] = []
        target = pd.to_numeric(frame[target_col], errors="coerce")
        for col in feature_columns:
            feature = pd.to_numeric(frame[col], errors="coerce")
            if feature.nunique(dropna=True) <= 1:
                continue
            corr = feature.corr(target)
            if pd.notna(corr) and abs(corr) >= threshold:
                selected.append(col)
        logger.info(
            "Correlation feature selection threshold=%.4f selected=%s/%s",
            threshold,
            len(selected or feature_columns),
            len(feature_columns),
        )
        return selected or feature_columns
