"""Optional AutoGluon TimeSeries adapter.

This adapter is intentionally lazy-imported so the core framework remains light.
Install the optional dependencies before using model.name=autogluon.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from load_prediction.configs import ModelSpecConfig, ForecastProfileConfig
from load_prediction.constants import DEFAULT_QUANTILE_LEVELS
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.models.base_forecaster import BaseForecastModel, ForecastFrame

logger = logging.getLogger(__name__)

SUPPORTED_AUTOGLUON_MODELS: tuple[str, ...] = (
    "ADIDA",
    "ARIMA",
    "AutoARIMA",
    "AutoCES",
    "AutoETS",
    "Average",
    "Chronos",
    "Chronos2",
    "Chronos-2",
    "Croston",
    "CrostonSBA",
    "DLinear",
    "DeepAR",
    "DirectTabular",
    "DynamicOptimizedTheta",
    "ETS",
    "IMAPA",
    "NPTS",
    "Naive",
    "PatchTST",
    "PerStepTabular",
    "RecursiveTabular",
    "SeasonalAverage",
    "SeasonalNaive",
    "SimpleFeedForward",
    "TFT",
    "TemporalFusionTransformer",
    "Theta",
    "TiDE",
    "Toto",
    "WaveNet",
    "Zero",
)
DEFAULT_AUTOGLUON_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "RecursiveTabular": {"model_name": "GBM"}
}


@dataclass
class AutoGluonTimeSeriesForecaster(BaseForecastModel):
    prediction_length: int
    freq: str
    known_covariates_names: list[str]
    eval_metric: str = "MAE"
    presets: str = "medium_quality"
    time_limit: int | None = 300
    path: str | None = None
    hyperparameters: str | dict[str, Any] | None = None
    hyperparameter_tune_kwargs: str | dict[str, Any] | None = None
    excluded_model_types: list[str] | None = None
    enable_ensemble: bool = True
    random_seed: int = 42
    quantile_levels: list[float] | None = None
    predictor: object | None = None
    residual_std_: float = 0.0

    @classmethod
    def from_config(
        cls,
        model_config: ModelSpecConfig,
        scale_config: ForecastProfileConfig,
    ) -> "AutoGluonTimeSeriesForecaster":
        params = dict(model_config.params)
        hyperparameters = params.pop("hyperparameters", None)
        if hyperparameters is None:
            hyperparameters = DEFAULT_AUTOGLUON_HYPERPARAMETERS
            logger.info("AutoGluon hyperparameters not provided; using default=%s", hyperparameters)
        hyperparameters = _with_seeded_autogluon_hyperparameters(
            hyperparameters,
            model_config.random_state,
        )
        excluded_model_types = params.pop("excluded_model_types", None)
        _validate_autogluon_model_selection(hyperparameters, excluded_model_types)
        return cls(
            prediction_length=scale_config.prediction_length,
            freq=scale_config.freq,
            known_covariates_names=list(scale_config.feature.known_covariates),
            eval_metric=params.pop("eval_metric", "MAE"),
            presets=params.pop("presets", "medium_quality"),
            time_limit=params.pop("time_limit", 300),
            path=params.pop("path", None),
            hyperparameters=hyperparameters,
            hyperparameter_tune_kwargs=params.pop("hyperparameter_tune_kwargs", None),
            excluded_model_types=excluded_model_types,
            enable_ensemble=params.pop("enable_ensemble", True),
            random_seed=model_config.random_state,
            quantile_levels=params.pop(
                "quantile_levels",
                list(DEFAULT_QUANTILE_LEVELS),
            ),
        )

    def fit(self, data: TimeSeriesDataset) -> "AutoGluonTimeSeriesForecaster":
        _set_global_random_seed(self.random_seed)
        mpl_config_dir = Path("tmp/matplotlib")
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        except ImportError as exc:
            raise ImportError(
                "AutoGluon model requires optional dependencies. "
                "Install with: pip install autogluon.timeseries"
            ) from exc

        frame = self._to_autogluon_frame(data)
        ts_frame = TimeSeriesDataFrame.from_data_frame(frame)
        available_covariates = [
            name for name in self.known_covariates_names if name in frame.columns
        ]
        self.predictor = TimeSeriesPredictor(
            target=data.target_col,
            prediction_length=self.prediction_length,
            freq=self.freq,
            eval_metric=self.eval_metric,
            known_covariates_names=available_covariates or None,
            path=self.path,
            quantile_levels=self.quantile_levels,
        )
        fit_kwargs: dict[str, Any] = {
            "presets": self.presets,
            "time_limit": self.time_limit,
            "enable_ensemble": self.enable_ensemble,
            "random_seed": self.random_seed,
        }
        if self.hyperparameters is not None:
            fit_kwargs["hyperparameters"] = self.hyperparameters
        if self.hyperparameter_tune_kwargs is not None:
            fit_kwargs["hyperparameter_tune_kwargs"] = self.hyperparameter_tune_kwargs
        if self.excluded_model_types is not None:
            fit_kwargs["excluded_model_types"] = self.excluded_model_types

        logger.info(
            "Fitting AutoGluon TimeSeries eval_metric=%s presets=%s random_seed=%s hyperparameters=%s excluded_model_types=%s",
            self.eval_metric,
            self.presets,
            self.random_seed,
            self.hyperparameters,
            self.excluded_model_types,
        )
        self.predictor.fit(ts_frame, **fit_kwargs)
        return self

    def predict(
        self,
        history: TimeSeriesDataset,
        prediction_length: int,
        freq: str,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        if self.predictor is None:
            raise RuntimeError("Model must be fitted before predict")
        if prediction_length != self.prediction_length:
            raise ValueError(
                "AutoGluon predictor was trained for prediction_length="
                f"{self.prediction_length}, got {prediction_length}."
            )

        try:
            from autogluon.timeseries import TimeSeriesDataFrame
        except ImportError as exc:
            raise ImportError(
                "AutoGluon model requires optional dependencies. "
                "Install with: pip install autogluon.timeseries"
            ) from exc

        history_frame = TimeSeriesDataFrame.from_data_frame(self._to_autogluon_frame(history))
        predict_kwargs = {}
        if known_covariates is not None and not known_covariates.empty:
            predict_kwargs["known_covariates"] = TimeSeriesDataFrame.from_data_frame(
                known_covariates
            )
        raw_predictions = self.predictor.predict(history_frame, **predict_kwargs)
        forecast = raw_predictions.reset_index()
        prediction_col = "mean" if "mean" in forecast.columns else forecast.select_dtypes("number").columns[0]
        forecast = forecast.rename(columns={prediction_col: "prediction"})
        keep = [history.item_id_col, history.timestamp_col, "prediction"]
        quantile_cols = [
            col
            for col in forecast.columns
            if _is_quantile_column(col) and col not in keep
        ]
        forecast = forecast.rename(columns={col: str(float(col)) for col in quantile_cols})
        quantile_cols = sorted(
            [str(float(col)) for col in quantile_cols],
            key=float,
        )
        output = forecast[keep + quantile_cols].copy()
        if {"0.1", "0.9"}.issubset(output.columns):
            output["prediction_lower"] = output["0.1"]
            output["prediction_upper"] = output["0.9"]
            logger.info("Mapped AutoGluon native quantiles 0.1/0.9 to prediction interval columns")
        return ForecastFrame(
            frame=output,
            timestamp_col=history.timestamp_col,
            item_id_col=history.item_id_col,
            prediction_col="prediction",
        )

    def save(self, path: str | Path) -> Path:
        if self.predictor is None:
            raise RuntimeError("Model must be fitted before save")
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        predictor_path = getattr(self.predictor, "path", None)
        return Path(predictor_path) if predictor_path else output

    def _to_autogluon_frame(self, data: TimeSeriesDataset) -> pd.DataFrame:
        frame = data.frame.rename(
            columns={
                data.item_id_col: "item_id",
                data.timestamp_col: "timestamp",
            }
        )
        if data.item_id_col != "item_id" or data.timestamp_col != "timestamp":
            frame = frame.copy()
        return frame


def _set_global_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        logger.debug("PyTorch is not installed; skipping torch random seed setup.")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Set global random seed for AutoGluon training seed=%s", seed)


def _with_seeded_autogluon_hyperparameters(
    hyperparameters: str | dict[str, Any] | None,
    seed: int,
) -> str | dict[str, Any] | None:
    if not isinstance(hyperparameters, dict):
        return hyperparameters

    seeded = copy.deepcopy(hyperparameters)
    for model_name, model_config in seeded.items():
        configs = model_config if isinstance(model_config, list) else [model_config]
        for config in configs:
            if isinstance(config, dict):
                _set_autogluon_model_seed(model_name, config, seed)
    return seeded


def _set_autogluon_model_seed(model_name: str, config: dict[str, Any], seed: int) -> None:
    if model_name not in {"RecursiveTabular", "DirectTabular", "PerStepTabular"}:
        return

    tabular_model_name = config.get("model_name")
    if tabular_model_name not in {"GBM", "CAT", "RF", "XT"}:
        return

    model_hyperparameters = config.setdefault("model_hyperparameters", {})
    if not isinstance(model_hyperparameters, dict):
        logger.warning(
            "AutoGluon %s model_hyperparameters is not a dict; cannot inject random seed.",
            model_name,
        )
        return

    seed_key = "random_seed" if tabular_model_name == "CAT" else "random_state"
    if tabular_model_name == "GBM":
        seed_key = "seed"
    model_hyperparameters.setdefault(seed_key, seed)
    logger.info(
        "Configured AutoGluon %s inner model=%s seed via %s=%s",
        model_name,
        tabular_model_name,
        seed_key,
        model_hyperparameters[seed_key],
    )


def _is_quantile_column(column: object) -> bool:
    if not isinstance(column, str):
        return False
    try:
        value = float(column)
    except ValueError:
        return False
    return 0.0 < value < 1.0


def _validate_autogluon_model_selection(
    hyperparameters: str | dict[str, Any] | None,
    excluded_model_types: list[str] | None,
) -> None:
    supported = set(SUPPORTED_AUTOGLUON_MODELS)
    invalid: list[str] = []

    if isinstance(hyperparameters, dict):
        invalid.extend([name for name in hyperparameters if name not in supported])
    elif hyperparameters is not None and not isinstance(hyperparameters, str):
        raise TypeError(
            "AutoGluon hyperparameters must be a preset string, dict, or None; "
            f"got {type(hyperparameters).__name__}"
        )

    if excluded_model_types:
        invalid.extend([name for name in excluded_model_types if name not in supported])

    if invalid:
        available = ", ".join(SUPPORTED_AUTOGLUON_MODELS)
        raise ValueError(
            "Unsupported AutoGluon TimeSeries model(s): "
            f"{sorted(set(invalid))}. Supported models: {available}"
        )
