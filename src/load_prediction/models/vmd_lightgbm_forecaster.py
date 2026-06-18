"""VMD decomposition plus LightGBM recursive forecasting."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from load_prediction.configs import FeatureEngineeringConfig, ModelSpecConfig, VMDConfig
from load_prediction.constants import (
    DEFAULT_INTERVAL_LOWER_QUANTILE,
    DEFAULT_INTERVAL_UPPER_QUANTILE,
    DEFAULT_QUANTILE_LEVELS,
)
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.features import FeatureBuilder
from load_prediction.models.base_forecaster import BaseForecastModel, ForecastFrame
from load_prediction.models.recursive_tabular import QuantileRecursiveTabularForecaster

logger = logging.getLogger(__name__)


@dataclass
class VMDLightGBMForecaster(BaseForecastModel):
    """Train one LightGBM model per VMD mode and sum mode forecasts."""

    feature_config: FeatureEngineeringConfig
    model_config: ModelSpecConfig
    vmd_config: VMDConfig = field(default_factory=VMDConfig)
    component_models_: list[QuantileRecursiveTabularForecaster] = field(default_factory=list)
    residual_std_: float = 0.0
    fitted_: bool = False
    target_col_: str = "target"

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        feature_config: FeatureEngineeringConfig,
    ) -> "VMDLightGBMForecaster":
        params = dict(model_config.params)
        vmd_config = VMDConfig(
            num_modes=int(params.pop("num_modes", params.pop("k", 4))),
            alpha=float(params.pop("alpha", 2000.0)),
            tau=float(params.pop("tau", 0.0)),
            max_iter=int(params.pop("max_iter", 300)),
            tolerance=float(params.pop("tolerance", 1e-6)),
            init=str(params.pop("init", "uniform")),
            dc_mode=bool(params.pop("dc_mode", False)),
        )
        component_model_config = ModelSpecConfig(
            name="lightgbm",
            random_state=model_config.random_state,
            params=params,
            scale_features=model_config.scale_features,
            scale_target=model_config.scale_target,
        )
        return cls(
            feature_config=feature_config,
            model_config=component_model_config,
            vmd_config=vmd_config,
        )

    def fit(self, data: TimeSeriesDataset) -> "VMDLightGBMForecaster":
        logger.info(
            "Fitting VMDLightGBM modes=%s alpha=%s rows=%s items=%s",
            self.vmd_config.num_modes,
            self.vmd_config.alpha,
            len(data.frame),
            len(data.item_ids),
        )
        component_frames = self._component_frames(data.frame, data, fit=True)
        self.component_models_ = []
        for mode_index, component_frame in enumerate(component_frames):
            logger.info("Fitting VMD LightGBM component mode=%s rows=%s", mode_index + 1, len(component_frame))
            component_data = data.with_frame(component_frame)
            model = self._build_lightgbm_component()
            model.fit(component_data)
            self.component_models_.append(model)

        fitted = self._predict_components_on_history(data)
        actual = data.frame[[data.item_id_col, data.timestamp_col, data.target_col]]
        joined = actual.merge(
            fitted.frame,
            on=[data.item_id_col, data.timestamp_col],
            how="inner",
        )
        if not joined.empty:
            residual = joined[data.target_col].astype(float) - joined["prediction"].astype(float)
            self.residual_std_ = float(np.nan_to_num(np.std(residual, ddof=1), nan=0.0))
        self.fitted_ = True
        self.target_col_ = data.target_col
        logger.info("Finished fitting VMDLightGBM residual_std=%.6f", self.residual_std_)
        return self

    def predict(
        self,
        history: TimeSeriesDataset,
        prediction_length: int,
        freq: str,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        if not self.fitted_ or not self.component_models_:
            raise RuntimeError("Model must be fitted before predict")

        logger.info(
            "Predicting VMDLightGBM steps=%s freq=%s visible_history_rows=%s",
            prediction_length,
            freq,
            len(history.frame),
        )
        component_frames = self._component_frames(history.frame, history, fit=False)
        component_forecasts = []
        for mode_index, (model, component_frame) in enumerate(zip(self.component_models_, component_frames, strict=True)):
            logger.info("Predicting VMD LightGBM component mode=%s", mode_index + 1)
            component_history = history.with_frame(component_frame)
            component_forecasts.append(
                model.predict(
                    component_history,
                    prediction_length=prediction_length,
                    freq=freq,
                    known_covariates=known_covariates,
                ).frame
            )

        combined = self._combine_component_forecasts(component_forecasts, history)
        return ForecastFrame(
            frame=combined,
            timestamp_col=history.timestamp_col,
            item_id_col=history.item_id_col,
            prediction_col="prediction",
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.joblib"
        for component_model in self.component_models_:
            if hasattr(component_model, "quantile_estimator_factory"):
                component_model.quantile_estimator_factory = None
        joblib.dump(self, model_path)
        return model_path

    def training_log(self) -> dict[str, Any]:
        return {
            "metadata": {
                "model_type": "vmd_lightgbm",
                "num_modes": self.vmd_config.num_modes,
                "alpha": self.vmd_config.alpha,
                "tau": self.vmd_config.tau,
                "max_iter": self.vmd_config.max_iter,
                "tolerance": self.vmd_config.tolerance,
                "init": self.vmd_config.init,
                "dc_mode": self.vmd_config.dc_mode,
                "component_model": "lightgbm",
            }
        }

    def _component_frames(
        self,
        frame: pd.DataFrame,
        data: TimeSeriesDataset,
        fit: bool,
    ) -> list[pd.DataFrame]:
        mode_parts = [[] for _ in range(self.vmd_config.num_modes)]
        for item_id, group in frame.groupby(data.item_id_col, sort=False):
            ordered = group.sort_values(data.timestamp_col).copy()
            values = pd.to_numeric(ordered[data.target_col], errors="coerce").interpolate().ffill().bfill()
            modes = vmd_decompose(values.to_numpy(dtype=float), self.vmd_config)
            for mode_index in range(self.vmd_config.num_modes):
                component = ordered.copy()
                component[data.target_col] = modes[mode_index]
                mode_parts[mode_index].append(component)
            logger.info(
                "VMD decomposed item=%s rows=%s fit=%s modes=%s",
                item_id,
                len(ordered),
                fit,
                self.vmd_config.num_modes,
            )
        return [pd.concat(parts, ignore_index=True) for parts in mode_parts]

    def _build_lightgbm_component(self) -> QuantileRecursiveTabularForecaster:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("VMD LightGBM model requires: pip install lightgbm") from exc

        params = dict(self.model_config.params)
        enable_quantiles = params.pop("enable_quantiles", True)
        if not enable_quantiles:
            logger.warning("VMDLightGBM currently enables quantile component models for interval aggregation")
        lower_quantile = params.pop("lower_quantile", DEFAULT_INTERVAL_LOWER_QUANTILE)
        upper_quantile = params.pop("upper_quantile", DEFAULT_INTERVAL_UPPER_QUANTILE)
        quantile_levels = tuple(
            float(level)
            for level in params.pop("quantile_levels", DEFAULT_QUANTILE_LEVELS)
        )
        quantile_levels = tuple(sorted({*quantile_levels, float(lower_quantile), float(upper_quantile)}))
        objective = params.pop("objective", "regression_l1")
        base_params = {
            "n_estimators": params.pop("n_estimators", 300),
            "learning_rate": params.pop("learning_rate", 0.05),
            "num_leaves": params.pop("num_leaves", 31),
            "n_jobs": params.pop("n_jobs", 1),
            "verbosity": params.pop("verbosity", -1),
            **params,
        }
        point_estimator = LGBMRegressor(
            **base_params,
            objective=objective,
            random_state=self.model_config.random_state,
        )

        def quantile_estimator_factory(alpha: float):
            return LGBMRegressor(
                **base_params,
                objective="quantile",
                alpha=alpha,
                random_state=self.model_config.random_state,
            )

        return QuantileRecursiveTabularForecaster(
            feature_builder=FeatureBuilder(self.feature_config),
            estimator=point_estimator,
            lower_estimator=quantile_estimator_factory(lower_quantile),
            upper_estimator=quantile_estimator_factory(upper_quantile),
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            quantile_levels=quantile_levels,
            quantile_estimator_factory=quantile_estimator_factory,
            scale_features=self.model_config.scale_features,
            scale_target=self.model_config.scale_target,
        )

    def _combine_component_forecasts(
        self,
        component_forecasts: list[pd.DataFrame],
        history: TimeSeriesDataset,
    ) -> pd.DataFrame:
        key_cols = [history.item_id_col, history.timestamp_col, "horizon_step"]
        combined = component_forecasts[0][key_cols].copy()
        combined["prediction"] = 0.0

        quantile_cols = sorted(
            {
                col
                for forecast in component_forecasts
                for col in forecast.columns
                if _is_quantile_column(col)
            },
            key=float,
        )
        for col in quantile_cols:
            combined[col] = 0.0

        for forecast in component_forecasts:
            component = forecast[key_cols + ["prediction", *quantile_cols]].copy()
            combined["prediction"] += component["prediction"].to_numpy(dtype=float)
            for col in quantile_cols:
                if col in component:
                    combined[col] += component[col].to_numpy(dtype=float)

        if "0.1" in combined and "0.9" in combined:
            combined["0.1"] = np.minimum(combined["0.1"], combined["prediction"])
            combined["0.9"] = np.maximum(combined["0.9"], combined["prediction"])
        return combined

    def _predict_components_on_history(self, data: TimeSeriesDataset) -> ForecastFrame:
        frame = data.frame.copy()
        min_required = max(self.feature_config.lag_steps or (1,)) + 1
        if len(frame) <= min_required:
            return ForecastFrame(
                frame=pd.DataFrame(columns=[data.item_id_col, data.timestamp_col, "prediction"]),
                timestamp_col=data.timestamp_col,
                item_id_col=data.item_id_col,
            )
        holdout = min(max(24, min_required), max(1, len(frame) // 10))
        train_data, test_data = data.split_holdout(holdout)
        forecasts = []
        component_train_frames = self._component_frames(train_data.frame, train_data, fit=False)
        for model, component_frame in zip(self.component_models_, component_train_frames, strict=True):
            forecasts.append(
                model.predict(
                    train_data.with_frame(component_frame),
                    prediction_length=holdout,
                    freq=data.freq or "15min",
                ).frame
            )
        combined = self._combine_component_forecasts(forecasts, data)
        expected = test_data.frame[[data.item_id_col, data.timestamp_col]]
        combined = combined.merge(expected, on=[data.item_id_col, data.timestamp_col], how="inner")
        return ForecastFrame(combined, timestamp_col=data.timestamp_col, item_id_col=data.item_id_col)


def vmd_decompose(signal: np.ndarray, config: VMDConfig) -> np.ndarray:
    """Decompose a one-dimensional signal into band-limited modes.

    This follows the standard ADMM update form used by VMD implementations.
    The caller is responsible for using only visible history in forecasting
    workflows to avoid leakage from future target values.
    """

    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("VMD input signal must be one-dimensional")
    if len(values) < 3:
        return np.tile(values, (config.num_modes, 1)) / max(config.num_modes, 1)

    original_length = len(values)
    original_values = values.copy()
    centered_values = values - np.mean(values)
    mirrored = np.concatenate([
        centered_values[: original_length // 2][::-1],
        centered_values,
        centered_values[-original_length // 2 :][::-1],
    ])
    length = len(mirrored)
    freqs = np.fft.fftshift(np.fft.fftfreq(length))
    f_hat = np.fft.fftshift(np.fft.fft(mirrored))
    f_hat_plus = f_hat.copy()
    f_hat_plus[: length // 2] = 0

    alpha = np.full(config.num_modes, float(config.alpha))
    u_hat_plus = np.zeros((config.num_modes, length), dtype=complex)
    lambda_hat = np.zeros(length, dtype=complex)
    omega = _initial_omega(config, freqs)

    previous = np.zeros_like(u_hat_plus)
    for _ in range(config.max_iter):
        sum_uk = np.sum(u_hat_plus, axis=0)
        for mode in range(config.num_modes):
            sum_uk = sum_uk - u_hat_plus[mode]
            denominator = 1.0 + alpha[mode] * (freqs - omega[mode]) ** 2
            u_hat_plus[mode] = (f_hat_plus - sum_uk - lambda_hat / 2.0) / denominator
            if not (config.dc_mode and mode == 0):
                positive = slice(length // 2, length)
                power = np.abs(u_hat_plus[mode, positive]) ** 2
                power_sum = np.sum(power)
                if power_sum > 0:
                    omega[mode] = float(np.sum(freqs[positive] * power) / power_sum)
            sum_uk = sum_uk + u_hat_plus[mode]
        lambda_hat = lambda_hat + config.tau * (np.sum(u_hat_plus, axis=0) - f_hat_plus)
        diff = np.sum(np.abs(u_hat_plus - previous) ** 2) / max(np.sum(np.abs(previous) ** 2), 1e-12)
        if diff < config.tolerance:
            break
        previous = u_hat_plus.copy()

    modes = np.zeros((config.num_modes, length))
    for mode in range(config.num_modes):
        u_hat = np.zeros(length, dtype=complex)
        u_hat[length // 2 :] = u_hat_plus[mode, length // 2 :]
        u_hat[: length // 2] = np.conj(u_hat_plus[mode, : length // 2][::-1])
        modes[mode] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat)))

    start = original_length // 2
    trimmed = modes[:, start : start + original_length]
    reconstruction_error = original_values - np.sum(trimmed, axis=0)
    trimmed[-1] += reconstruction_error
    return trimmed


def _initial_omega(config: VMDConfig, freqs: np.ndarray) -> np.ndarray:
    if config.init == "zero":
        omega = np.zeros(config.num_modes)
    elif config.init == "random":
        rng = np.random.default_rng(42)
        omega = np.sort(rng.uniform(0, 0.5, config.num_modes))
    else:
        omega = np.linspace(0, 0.5, config.num_modes, endpoint=False)
    if config.dc_mode:
        omega[0] = 0.0
    return omega


def _is_quantile_column(column: str) -> bool:
    try:
        value = float(column)
    except (TypeError, ValueError):
        return False
    return 0.0 < value < 1.0
