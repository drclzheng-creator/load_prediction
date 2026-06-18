"""Feature engineering for recursive load forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import pandas as pd

from load_prediction.configs import FeatureEngineeringConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features.calendar_features import (
    build_calendar_feature_frame,
    build_calendar_features,
)
from load_prediction.features.feature_registry import resolve_external_feature_columns
from load_prediction.features.history_features import build_history_feature_frame, build_history_features
from load_prediction.features.item_features import build_item_feature_frame, build_item_features
from load_prediction.features.similar_time_features import (
    SIMILAR_TIME_FEATURE_COLUMNS,
    SimilarTimeFeatureBuilder,
)

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
    similar_time_builder_: SimilarTimeFeatureBuilder | None = None

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
        self.similar_time_builder_ = None
        self._fit_covariate_types(data.frame)
        frame = self._build_lagged_frame(data.frame, data)
        if self.config.add_similar_time_features:
            self.similar_time_builder_ = SimilarTimeFeatureBuilder(
                top_k=self.config.similar_time_top_k,
                lag_steps=self.config.similar_time_lag_steps,
                rolling_windows=self.config.similar_time_rolling_windows,
                weather_weight=self.config.similar_time_weather_weight,
                calendar_weight=self.config.similar_time_calendar_weight,
                lag_weight=self.config.similar_time_lag_weight,
                rolling_weight=self.config.similar_time_rolling_weight,
                candidate_lookback=self.config.similar_time_candidate_lookback,
                slot_tolerance_steps=self.config.similar_time_slot_tolerance_steps,
                restrict_weekend=self.config.similar_time_restrict_weekend,
                restrict_holiday=self.config.similar_time_restrict_holiday,
                restrict_extreme_weather=self.config.similar_time_restrict_extreme_weather,
                candidate_recent_days=self.config.similar_time_candidate_recent_days,
            )
            frame = self.similar_time_builder_.fit_transform(
                frame,
                timestamp_col=data.timestamp_col,
                item_id_col=data.item_id_col,
                target_col=data.target_col,
                numeric_covariates=self.numeric_covariates_,
            )
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
        row.update(
            build_calendar_features(
                pd.Timestamp(timestamp),
                add_calendar=self.config.add_calendar,
                add_cyclical_time=self.config.add_cyclical_time,
                add_chinese_calendar=self.config.add_chinese_calendar,
            )
        )
        row.update(
            build_item_features(
                str(item_id),
                self.item_ids_,
                enabled=self.config.add_item_id,
            )
        )
        row.update(self._known_covariate_features(known_covariates or {}))
        row.update(
            build_history_features(
                history,
                data.target_col,
                lag_steps=self.config.lag_steps,
                rolling_windows=self.config.rolling_windows,
            )
        )
        row.update(
            self._similar_time_features(
                history=history,
                data=data,
                timestamp=pd.Timestamp(timestamp),
                item_id=str(item_id),
                known_covariates=known_covariates or {},
            )
        )
        return pd.DataFrame([{col: row.get(col, 0.0) for col in self.selected_columns_}])

    def _build_lagged_frame(self, frame: pd.DataFrame, data: TimeSeriesDataset) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for item_id, group in frame.groupby(data.item_id_col, sort=False):
            item = group.sort_values(data.timestamp_col).copy()
            item_features = build_calendar_feature_frame(
                pd.to_datetime(item[data.timestamp_col]),
                add_calendar=self.config.add_calendar,
                add_cyclical_time=self.config.add_cyclical_time,
                add_chinese_calendar=self.config.add_chinese_calendar,
            )
            item_features = pd.concat(
                [
                    item_features,
                    build_item_feature_frame(
                        str(item_id),
                        self.item_ids_,
                        length=len(item),
                        enabled=self.config.add_item_id,
                    ),
                ],
                axis=1,
            )

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
            item_features = pd.concat(
                [
                    item_features,
                    build_history_feature_frame(
                        target,
                        lag_steps=self.config.lag_steps,
                        rolling_windows=self.config.rolling_windows,
                    ),
                ],
                axis=1,
            )

            item_features[data.target_col] = target.to_numpy()
            item_features[data.timestamp_col] = item[data.timestamp_col].to_numpy()
            item_features[data.item_id_col] = item[data.item_id_col].to_numpy()
            parts.append(item_features)

        return pd.concat(parts, ignore_index=True)

    def _similar_time_features(
        self,
        *,
        history: pd.DataFrame,
        data: TimeSeriesDataset,
        timestamp: pd.Timestamp,
        item_id: str,
        known_covariates: dict[str, object],
    ) -> dict[str, float]:
        if not self.config.add_similar_time_features:
            return {}
        if self.similar_time_builder_ is None:
            return {column: 0.0 for column in SIMILAR_TIME_FEATURE_COLUMNS}
        return self.similar_time_builder_.transform_future_row(
            history,
            timestamp=timestamp,
            item_id=item_id,
            timestamp_col=data.timestamp_col,
            item_id_col=data.item_id_col,
            target_col=data.target_col,
            numeric_covariates=self.numeric_covariates_,
            known_covariates=known_covariates,
        )

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
