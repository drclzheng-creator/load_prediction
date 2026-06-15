"""Gaussian mixture residual distribution forecaster."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.mixture import GaussianMixture

from load_prediction.configs import FeatureEngineeringConfig, ModelSpecConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features import FeatureBuilder
from load_prediction.models.base_forecaster import ForecastFrame
from load_prediction.models.recursive_tabular import RecursiveTabularForecaster

logger = logging.getLogger(__name__)


@dataclass
class GMMForecaster(RecursiveTabularForecaster):
    """Point forecaster with a Gaussian mixture model over prediction residuals.

    The point estimator predicts the conditional load mean/median from recursive
    tabular features. The GMM is fitted on in-sample residuals and is shifted by
    each point forecast to describe the predicted load distribution.
    """

    gmm: GaussianMixture | None = None
    n_components: int = 3
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    distribution_grid_size: int = 0
    distribution_std_width: float = 4.0
    residual_floor: float = 1e-6
    conditional_distribution: bool = True
    component_weights_: np.ndarray | None = None
    component_means_: np.ndarray | None = None
    component_stds_: np.ndarray | None = None

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        feature_config: FeatureEngineeringConfig,
    ) -> "GMMForecaster":
        params = dict(model_config.params)
        n_components = int(params.pop("n_components", 3))
        lower_quantile = float(params.pop("lower_quantile", 0.1))
        upper_quantile = float(params.pop("upper_quantile", 0.9))
        distribution_grid_size = int(params.pop("distribution_grid_size", 0))
        distribution_std_width = float(params.pop("distribution_std_width", 4.0))
        residual_floor = float(params.pop("residual_floor", 1e-6))
        conditional_distribution = bool(params.pop("conditional_distribution", True))
        reg_covar = float(params.pop("reg_covar", 1e-6))
        gmm_max_iter = int(params.pop("gmm_max_iter", params.pop("max_iter_gmm", 200)))
        n_init = int(params.pop("n_init", 3))
        init_params = str(params.pop("init_params", "random"))
        point_estimator_name = params.pop("point_estimator", "hist_gradient_boosting")
        point_params = dict(params.pop("point_params", params))

        if point_estimator_name in {"hist_gradient_boosting", "hgb", "sklearn_hgb"}:
            point_params.setdefault("loss", "absolute_error")
            estimator = HistGradientBoostingRegressor(
                random_state=model_config.random_state,
                **point_params,
            )
        elif point_estimator_name in {"lightgbm", "lgbm"}:
            try:
                from lightgbm import LGBMRegressor
            except ImportError as exc:
                raise ImportError("GMM lightgbm point_estimator requires: pip install lightgbm") from exc

            point_params.setdefault("n_estimators", 300)
            point_params.setdefault("learning_rate", 0.05)
            point_params.setdefault("objective", "regression_l1")
            point_params.setdefault("num_leaves", 31)
            point_params.setdefault("n_jobs", 1)
            point_params.setdefault("verbosity", -1)
            estimator = LGBMRegressor(
                random_state=model_config.random_state,
                **point_params,
            )
        elif point_estimator_name in {"random_forest", "rf", "sklearn_random_forest"}:
            criterion = point_params.pop("criterion", "absolute_error")
            estimator = RandomForestRegressor(
                n_estimators=point_params.pop("n_estimators", 160),
                criterion=criterion,
                min_samples_leaf=point_params.pop("min_samples_leaf", 2),
                n_jobs=point_params.pop("n_jobs", -1),
                random_state=model_config.random_state,
                **point_params,
            )
        else:
            raise ValueError(
                "Unsupported GMM point_estimator "
                f"'{point_estimator_name}'. Use hist_gradient_boosting, lightgbm, or random_forest."
            )

        logger.info(
            "Creating GMM forecaster point_estimator=%s n_components=%s quantiles=(%s, %s)",
            point_estimator_name,
            n_components,
            lower_quantile,
            upper_quantile,
        )
        return cls(
            feature_builder=FeatureBuilder(feature_config),
            estimator=estimator,
            gmm=GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=model_config.random_state,
                reg_covar=reg_covar,
                max_iter=gmm_max_iter,
                n_init=n_init,
                init_params=init_params,
            ),
            n_components=n_components,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            distribution_grid_size=distribution_grid_size,
            distribution_std_width=distribution_std_width,
            residual_floor=residual_floor,
            conditional_distribution=conditional_distribution,
            scale_features=model_config.scale_features,
            scale_target=model_config.scale_target,
        )

    def fit(self, data: TimeSeriesDataset) -> "GMMForecaster":
        logger.info(
            "Fitting %s point model and residual GMM components=%s on %s rows and %s item(s)",
            self.__class__.__name__,
            self.n_components,
            len(data.frame),
            len(data.item_ids),
        )
        if self.gmm is None:
            self.gmm = GaussianMixture(n_components=self.n_components, covariance_type="full")

        x_train, y_train = self.feature_builder.fit_transform(data)
        logger.info("Built training feature matrix shape=%s", x_train.shape)
        x_model = self._fit_transform_features(x_train)
        y_model = self._fit_transform_target(y_train)
        self.estimator.fit(x_model, y_model)
        fitted_values = self._inverse_transform_target(self.estimator.predict(x_model)).to_numpy()
        residual = y_train.to_numpy(dtype=float) - fitted_values
        self.residual_std_ = float(np.nan_to_num(np.std(residual, ddof=1), nan=0.0))

        if self.conditional_distribution:
            gmm_samples = np.column_stack(
                [
                    np.nan_to_num(fitted_values, nan=0.0),
                    np.nan_to_num(residual, nan=0.0),
                ]
            )
        else:
            gmm_samples = np.nan_to_num(residual, nan=0.0).reshape(-1, 1)
        if len(gmm_samples) < self.n_components:
            raise ValueError(
                f"GMM requires at least n_components={self.n_components} supervised rows; "
                f"got {len(gmm_samples)}."
            )
        for column_index in range(gmm_samples.shape[1]):
            if float(np.std(gmm_samples[:, column_index])) < self.residual_floor:
                gmm_samples[:, column_index] = gmm_samples[:, column_index] + np.linspace(
                    -self.residual_floor,
                    self.residual_floor,
                    len(gmm_samples),
                )

        self.gmm.fit(gmm_samples)
        self.component_weights_ = self.gmm.weights_.astype(float)
        residual_mean_index = 1 if self.conditional_distribution else 0
        self.component_means_ = self.gmm.means_[:, residual_mean_index].astype(float)
        covariances = np.asarray(self.gmm.covariances_, dtype=float)
        if self.conditional_distribution:
            variances = covariances[:, residual_mean_index, residual_mean_index]
        else:
            variances = covariances.reshape(self.n_components, -1)[:, 0]
        self.component_stds_ = np.sqrt(np.maximum(variances, self.residual_floor**2))
        self.fitted_ = True
        self.target_col_ = data.target_col
        logger.info(
            "Finished fitting %s residual_std=%.6f component_weights=%s",
            self.__class__.__name__,
            self.residual_std_,
            self.component_weights_.round(4).tolist(),
        )
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
        if self.gmm is None or self.component_weights_ is None:
            raise RuntimeError("GMM must be fitted before predict")

        logger.info(
            "Predicting %s GMM distribution steps at freq=%s for %s item(s)",
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
                row = self._distribution_row(prediction)
                row.update(
                    {
                        history.item_id_col: item_id,
                        history.timestamp_col: pd.Timestamp(timestamp),
                        "horizon_step": step,
                        "prediction": prediction,
                    }
                )
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

    def _distribution_row(self, prediction: float) -> dict[str, object]:
        weights, means, stds = self._components_for_prediction(prediction)
        lower, upper = self._mixture_quantiles(weights, means, stds)
        row: dict[str, object] = {
            "0.1": float(lower),
            "0.9": float(upper),
            "prediction_lower": float(lower),
            "prediction_upper": float(upper),
            "prediction_density": float(self._mixture_pdf(np.array([prediction]), weights, means, stds)[0]),
            "gmm_weights": json.dumps(weights.tolist()),
            "gmm_means": json.dumps(means.tolist()),
            "gmm_stds": json.dumps(stds.tolist()),
        }
        for index, (weight, mean, std) in enumerate(
            zip(weights, means, stds),
            1,
        ):
            row[f"gmm_weight_{index}"] = float(weight)
            row[f"gmm_mean_{index}"] = float(mean)
            row[f"gmm_std_{index}"] = float(std)

        if self.distribution_grid_size > 1:
            grid = self._distribution_grid(prediction, means, stds)
            row["distribution_values"] = json.dumps(grid.tolist())
            row["distribution_probabilities"] = json.dumps(
                self._mixture_pdf(grid, weights, means, stds).tolist()
            )
        return row

    def _components_for_prediction(self, prediction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.conditional_distribution:
            return (
                self.component_weights_,
                prediction + self.component_means_,
                self.component_stds_,
            )
        if self.gmm is None:
            raise RuntimeError("GMM must be fitted before predict")

        component_weights = []
        component_means = []
        component_stds = []
        log_weights = []
        covariances = np.asarray(self.gmm.covariances_, dtype=float)
        for index in range(self.n_components):
            mean_prediction = float(self.gmm.means_[index, 0])
            mean_residual = float(self.gmm.means_[index, 1])
            covariance = covariances[index]
            prediction_variance = max(float(covariance[0, 0]), self.residual_floor**2)
            residual_variance = max(float(covariance[1, 1]), self.residual_floor**2)
            cross_covariance = float(covariance[1, 0])
            conditional_residual_mean = mean_residual + (
                cross_covariance / prediction_variance
            ) * (prediction - mean_prediction)
            conditional_residual_variance = residual_variance - (
                cross_covariance * cross_covariance / prediction_variance
            )
            conditional_residual_std = float(
                np.sqrt(max(conditional_residual_variance, self.residual_floor**2))
            )

            prediction_std = float(np.sqrt(prediction_variance))
            z = (prediction - mean_prediction) / prediction_std
            log_likelihood = -0.5 * z * z - np.log(prediction_std) - 0.5 * np.log(2 * np.pi)
            log_weights.append(float(np.log(max(self.gmm.weights_[index], self.residual_floor)) + log_likelihood))
            component_means.append(float(prediction + conditional_residual_mean))
            component_stds.append(conditional_residual_std)

        log_weights_array = np.asarray(log_weights, dtype=float)
        log_weights_array = log_weights_array - np.max(log_weights_array)
        component_weights = np.exp(log_weights_array)
        component_weights = component_weights / component_weights.sum()
        return (
            component_weights.astype(float),
            np.asarray(component_means, dtype=float),
            np.asarray(component_stds, dtype=float),
        )

    def _mixture_quantiles(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> tuple[float, float]:
        quantiles = self._mixture_inverse_cdf(
            weights,
            means,
            stds,
            np.array([self.lower_quantile, self.upper_quantile], dtype=float),
        )
        return max(0.0, float(quantiles[0])), float(quantiles[1])

    def _distribution_grid(
        self,
        prediction: float,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        low = np.min(means - self.distribution_std_width * stds)
        high = np.max(means + self.distribution_std_width * stds)
        low = min(low, prediction)
        high = max(high, prediction)
        return np.linspace(max(0.0, low), high, self.distribution_grid_size)

    def _mixture_inverse_cdf(
        self,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        low = np.min(means - 8.0 * stds)
        high = np.max(means + 8.0 * stds)
        xs = np.linspace(low, high, 2048)
        cdf = self._mixture_cdf(xs, weights, means, stds)
        return np.interp(probabilities, cdf, xs)

    def _mixture_pdf(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        means = means.reshape(1, -1)
        stds = stds.reshape(1, -1)
        z = (values - means) / stds
        component_pdf = np.exp(-0.5 * z**2) / (stds * np.sqrt(2 * np.pi))
        return np.sum(component_pdf * weights.reshape(1, -1), axis=1)

    def _mixture_cdf(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        means = means.reshape(1, -1)
        stds = stds.reshape(1, -1)
        z = (values - means) / (stds * np.sqrt(2.0))
        component_cdf = 0.5 * (1.0 + np.vectorize(_erf)(z))
        return np.sum(component_cdf * weights.reshape(1, -1), axis=1)


def _erf(value: float) -> float:
    import math

    return math.erf(float(value))
