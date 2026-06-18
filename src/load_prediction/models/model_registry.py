"""Model registry."""



from __future__ import annotations

import logging

from load_prediction.configs import ModelSpecConfig, ForecastProfileConfig
from load_prediction.models.base_forecaster import BaseForecastModel
from load_prediction.models.sklearn_forecaster import RecursiveSklearnForecaster

logger = logging.getLogger(__name__)


def build_model(model_config: ModelSpecConfig, scale_config: ForecastProfileConfig) -> BaseForecastModel:
    logger.info("Building model name=%s scale=%s", model_config.name, scale_config.name)
    if model_config.name in {"sklearn_random_forest", "random_forest", "sklearn"}:
        return RecursiveSklearnForecaster.from_config(model_config, scale_config.feature)
    if model_config.name in {"sklearn_hist_gradient_boosting", "hist_gradient_boosting"}:
        return RecursiveSklearnForecaster.from_config(
            model_config,
            scale_config.feature,
            estimator_name="hist_gradient_boosting",
        )
    if model_config.name in {"lightgbm", "lgbm"}:
        from load_prediction.models.lightgbm_forecaster import LightGBMForecaster

        return LightGBMForecaster.from_config(model_config, scale_config.feature)
    if model_config.name in {"vmd_lightgbm", "lightgbm_vmd", "vmd_lgbm"}:
        from load_prediction.models.vmd_lightgbm_forecaster import VMDLightGBMForecaster

        return VMDLightGBMForecaster.from_config(model_config, scale_config.feature)
    if model_config.name in {"gmm", "gaussian_mixture", "gaussian_mixture_model"}:
        from load_prediction.models.gmm_forecaster import GMMForecaster

        return GMMForecaster.from_config(model_config, scale_config.feature)
    if model_config.name in {"mdn", "mixture_density_network"}:
        from load_prediction.models.mdn_forecaster import MDNForecaster

        return MDNForecaster.from_config(model_config, scale_config.feature)
    if model_config.name in {"autogluon", "autogluon_timeseries"}:
        from load_prediction.models.autogluon_forecaster import AutoGluonTimeSeriesForecaster

        return AutoGluonTimeSeriesForecaster.from_config(model_config, scale_config)
    if model_config.name in {"lstm", "torch_lstm", "bilstm", "bi_lstm"}:
        from load_prediction.models.lstm_forecaster import LSTMForecaster

        return LSTMForecaster.from_config(model_config, scale_config)
    if model_config.name in {"cnn_lstm", "cnnlstm", "conv_lstm"}:
        from load_prediction.models.lstm_forecaster import CNNLSTMForecaster

        return CNNLSTMForecaster.from_config(model_config, scale_config)
    if model_config.name in {"cnn_lstm_attention", "cnn_lstm_attn", "attention_cnn_lstm"}:
        from load_prediction.models.lstm_forecaster import CNNLSTMAttentionForecaster

        return CNNLSTMAttentionForecaster.from_config(model_config, scale_config)
    logger.warning("Unsupported model requested: %s", model_config.name)
    raise ValueError(f"Unsupported model '{model_config.name}'")
