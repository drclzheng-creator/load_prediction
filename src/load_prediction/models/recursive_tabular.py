"""Shared recursive tabular forecasting mechanics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features import FeatureBuilder
from load_prediction.models.base_forecaster import BaseForecastModel, ForecastFrame

logger = logging.getLogger(__name__)


@dataclass
class RecursiveTabularForecaster(BaseForecastModel):
    """One-step tabular estimator used recursively for multi-step forecasts."""

    feature_builder: FeatureBuilder
    estimator: object
    scale_features: bool = False
    scale_target: bool = False
    feature_scaler: StandardScaler | None = None
    target_scaler: StandardScaler | None = None
    residual_std_: float = 0.0
    fitted_: bool = False
    target_col_: str = "target"

    def fit(self, data: TimeSeriesDataset) -> "RecursiveTabularForecaster":
        logger.info(
            "Fitting %s on %s rows and %s item(s)",
            self.__class__.__name__,
            len(data.frame),
            len(data.item_ids),
        )
        x_train, y_train = self.feature_builder.fit_transform(data)
        logger.info("Built training feature matrix shape=%s", x_train.shape)
        x_model = self._fit_transform_features(x_train)
        y_model = self._fit_transform_target(y_train)
        self.estimator.fit(x_model, y_model)
        fitted_values = self._inverse_transform_target(self.estimator.predict(x_model)).to_numpy()
        fitted = pd.Series(fitted_values, index=y_train.index)
        residual = y_train.to_numpy() - fitted.to_numpy()
        self.residual_std_ = float(np.nan_to_num(np.std(residual, ddof=1), nan=0.0))
        self.fitted_ = True
        self.target_col_ = data.target_col
        logger.info("Finished fitting %s residual_std=%.6f", self.__class__.__name__, self.residual_std_)
        return self

    def predict(
        self,
        history: TimeSeriesDataset,
        prediction_length: int,
        freq: str,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before predict")

        logger.info(
            "Predicting %s steps at freq=%s for %s item(s)",
            prediction_length,
            freq,
            len(history.item_ids),
        )
        history_frame = history.frame.copy()
        forecast_rows: list[dict[str, object]] = []
        known_lookup = self._build_known_covariate_lookup(known_covariates, history)

        for item_id in history.item_ids:
            item_history = history_frame[history_frame[history.item_id_col] == item_id].copy()
            item_history = item_history.sort_values(history.timestamp_col)
            last_timestamp = pd.Timestamp(item_history[history.timestamp_col].max())
            future_index = pd.date_range(
                last_timestamp,
                periods=prediction_length + 1,
                freq=freq,
            )[1:]

            for step, timestamp in enumerate(future_index, 1):
                known = known_lookup.get((str(item_id), pd.Timestamp(timestamp)), {})
                features = self.feature_builder.transform_future_row(
                    item_history,
                    history,
                    pd.Timestamp(timestamp),
                    str(item_id),
                    known,
                )
                model_features = self._transform_features(features)
                prediction = float(
                    self._inverse_transform_target(self.estimator.predict(model_features))[0]
                )
                forecast_rows.append(
                    {
                        history.item_id_col: item_id,
                        history.timestamp_col: pd.Timestamp(timestamp),
                        "horizon_step": step,
                        "prediction": prediction,
                    }
                )
                new_row = {
                    history.item_id_col: item_id,
                    history.timestamp_col: pd.Timestamp(timestamp),
                    history.target_col: prediction,
                }
                for covariate in history.covariate_columns:
                    new_row[covariate] = known.get(covariate, 0.0)
                item_history = pd.concat([item_history, pd.DataFrame([new_row])], ignore_index=True)

        return ForecastFrame(
            frame=pd.DataFrame(forecast_rows),
            timestamp_col=history.timestamp_col,
            item_id_col=history.item_id_col,
            prediction_col="prediction",
        )

    def _build_known_covariate_lookup(
        self,
        known_covariates: pd.DataFrame | None,
        history: TimeSeriesDataset,
    ) -> dict[tuple[str, pd.Timestamp], dict[str, object]]:
        if known_covariates is None or known_covariates.empty:
            return {}

        lookup: dict[tuple[str, pd.Timestamp], dict[str, object]] = {}
        covariate_columns = [
            col
            for col in known_covariates.columns
            if col not in {history.timestamp_col, history.item_id_col}
        ]
        for row in known_covariates.to_dict("records"):
            item_id = str(row.get(history.item_id_col, history.item_ids[0]))
            timestamp = pd.Timestamp(row[history.timestamp_col])
            lookup[(item_id, timestamp)] = {
                col: row[col] for col in covariate_columns if pd.notna(row.get(col))
            }
        return lookup

    def _fit_transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_features:
            logger.info("Feature scaling disabled")
            return features
        logger.info("Fitting feature StandardScaler on training features only")
        self.feature_scaler = StandardScaler()
        values = self.feature_scaler.fit_transform(features)
        return pd.DataFrame(values, columns=features.columns, index=features.index)

    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.scale_features:
            return features
        if self.feature_scaler is None:
            raise RuntimeError("Feature scaler has not been fitted")
        values = self.feature_scaler.transform(features)
        return pd.DataFrame(values, columns=features.columns, index=features.index)

    def _fit_transform_target(self, target: pd.Series) -> pd.Series:
        if not self.scale_target:
            logger.info("Target scaling disabled")
            return target
        logger.info("Fitting target StandardScaler on training target only")
        self.target_scaler = StandardScaler()
        values = self.target_scaler.fit_transform(target.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(values, index=target.index)

    def _inverse_transform_target(self, values: object) -> pd.Series:
        series = pd.Series(values)
        if not self.scale_target:
            return series
        if self.target_scaler is None:
            raise RuntimeError("Target scaler has not been fitted")
        restored = self.target_scaler.inverse_transform(series.to_numpy().reshape(-1, 1)).ravel()
        return pd.Series(restored, index=series.index)


@dataclass
class QuantileRecursiveTabularForecaster(RecursiveTabularForecaster):
    """Recursive tabular forecaster with point model plus quantile models."""

    lower_estimator: object | None = None
    upper_estimator: object | None = None
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    quantile_estimator_factory: Callable[[float], object] | None = field(default=None, repr=False)
    quantile_levels: tuple[float, ...] | None = None
    quantile_estimators: dict[float, object] | None = field(default=None, repr=False)

    def save(self, path: str | Path) -> Path:
        self.quantile_estimator_factory = None
        return super().save(path)

    def fit(self, data: TimeSeriesDataset) -> "QuantileRecursiveTabularForecaster":
        quantile_levels = self._resolved_quantile_levels()
        logger.info(
            "Fitting %s point model and quantiles=%s on %s rows and %s item(s)",
            self.__class__.__name__,
            quantile_levels,
            len(data.frame),
            len(data.item_ids),
        )
        if self.quantile_estimator_factory is not None:
            self.quantile_estimators = {
                level: self.quantile_estimator_factory(level)
                for level in quantile_levels
            }
            self.lower_estimator = self.quantile_estimators.get(self.lower_quantile)
            self.upper_estimator = self.quantile_estimators.get(self.upper_quantile)
        elif self.quantile_estimators is None:
            self.quantile_estimators = {
                self.lower_quantile: self.lower_estimator,
                self.upper_quantile: self.upper_estimator,
            }
        missing_levels = [
            level
            for level in quantile_levels
            if self.quantile_estimators is None or self.quantile_estimators.get(level) is None
        ]
        if missing_levels:
            raise RuntimeError(f"Quantile forecaster is missing estimators for levels={missing_levels}")

        x_train, y_train = self.feature_builder.fit_transform(data)
        logger.info("Built training feature matrix shape=%s", x_train.shape)
        x_model = self._fit_transform_features(x_train)
        y_model = self._fit_transform_target(y_train)
        for level in quantile_levels:
            self.quantile_estimators[level].fit(x_model, y_model)
        self.estimator.fit(x_model, y_model)
        fitted_values = self._inverse_transform_target(self.estimator.predict(x_model)).to_numpy()
        fitted = pd.Series(fitted_values, index=y_train.index)
        residual = y_train.to_numpy() - fitted.to_numpy()
        self.residual_std_ = float(np.nan_to_num(np.std(residual, ddof=1), nan=0.0))
        self.fitted_ = True
        self.target_col_ = data.target_col
        logger.info("Finished fitting %s residual_std=%.6f", self.__class__.__name__, self.residual_std_)
        return self

    def predict(
        self,
        history: TimeSeriesDataset,
        prediction_length: int,
        freq: str,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before predict")
        quantile_levels = self._resolved_quantile_levels()
        if self.quantile_estimators is None:
            raise RuntimeError("Quantile forecaster requires fitted quantile_estimators")

        logger.info(
            "Predicting %s quantile steps at freq=%s for %s item(s)",
            prediction_length,
            freq,
            len(history.item_ids),
        )
        history_frame = history.frame.copy()
        forecast_rows: list[dict[str, object]] = []
        known_lookup = self._build_known_covariate_lookup(known_covariates, history)

        for item_id in history.item_ids:
            item_history = history_frame[history_frame[history.item_id_col] == item_id].copy()
            item_history = item_history.sort_values(history.timestamp_col)
            last_timestamp = pd.Timestamp(item_history[history.timestamp_col].max())
            future_index = pd.date_range(
                last_timestamp,
                periods=prediction_length + 1,
                freq=freq,
            )[1:]

            for step, timestamp in enumerate(future_index, 1):
                known = known_lookup.get((str(item_id), pd.Timestamp(timestamp)), {})
                features = self.feature_builder.transform_future_row(
                    item_history,
                    history,
                    pd.Timestamp(timestamp),
                    str(item_id),
                    known,
                )
                model_features = self._transform_features(features)
                prediction = float(
                    self._inverse_transform_target(self.estimator.predict(model_features))[0]
                )
                quantile_predictions = {
                    level: float(
                        self._inverse_transform_target(
                            self.quantile_estimators[level].predict(model_features)
                        )[0]
                    )
                    for level in quantile_levels
                }
                quantile_predictions = self._monotonic_quantile_predictions(
                    quantile_predictions,
                    prediction,
                )
                lower = quantile_predictions[self.lower_quantile]
                upper = quantile_predictions[self.upper_quantile]
                lower = min(lower, prediction)
                upper = max(upper, prediction)
                row = {
                    history.item_id_col: item_id,
                    history.timestamp_col: pd.Timestamp(timestamp),
                    "horizon_step": step,
                    "prediction": prediction,
                }
                row.update(
                    {
                        self._quantile_column(level): value
                        for level, value in quantile_predictions.items()
                    }
                )
                row[self._quantile_column(self.lower_quantile)] = lower
                row[self._quantile_column(self.upper_quantile)] = upper
                forecast_rows.append(row)
                new_row = {
                    history.item_id_col: item_id,
                    history.timestamp_col: pd.Timestamp(timestamp),
                    history.target_col: prediction,
                }
                for covariate in history.covariate_columns:
                    new_row[covariate] = known.get(covariate, 0.0)
                item_history = pd.concat([item_history, pd.DataFrame([new_row])], ignore_index=True)

        return ForecastFrame(
            frame=pd.DataFrame(forecast_rows),
            timestamp_col=history.timestamp_col,
            item_id_col=history.item_id_col,
            prediction_col="prediction",
        )

    def _resolved_quantile_levels(self) -> tuple[float, ...]:
        levels = self.quantile_levels or (self.lower_quantile, self.upper_quantile)
        levels = tuple(sorted({float(level) for level in levels}))
        if not levels:
            raise RuntimeError("Quantile forecaster requires at least one quantile level")
        if self.lower_quantile not in levels or self.upper_quantile not in levels:
            raise RuntimeError("Quantile levels must include lower_quantile and upper_quantile")
        for level in levels:
            if not 0.0 < level < 1.0:
                raise ValueError(f"Quantile level must be between 0 and 1, got {level}")
        return levels

    @staticmethod
    def _quantile_column(level: float) -> str:
        return f"{float(level):g}"

    def _monotonic_quantile_predictions(
        self,
        quantile_predictions: dict[float, float],
        prediction: float,
    ) -> dict[float, float]:
        ordered_levels = sorted(quantile_predictions)
        values = np.asarray([quantile_predictions[level] for level in ordered_levels], dtype=float)
        values = np.maximum.accumulate(values)
        for index, level in enumerate(ordered_levels):
            if level < 0.5:
                values[index] = min(values[index], prediction)
            elif level > 0.5:
                values[index] = max(values[index], prediction)
        values = np.maximum.accumulate(values)
        return {
            level: float(value)
            for level, value in zip(ordered_levels, values, strict=True)
        }
