from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_prediction.configs import ModelSpecConfig, PipelineConfig
from load_prediction.data import CSVLoadDataLoader
from load_prediction.inference import OnlineInferenceRequest, OnlineInferenceTask, run_online_inference
from load_prediction.preprocessing import TimeSeriesCleaner


DEFAULT_CONFIG = Path("configs/day_ahead.yaml")
SHORT_TERM_4H_CONFIG = Path("configs/short_term_4h.yaml")
REPLAY_WINDOW_COUNT = 10


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records = []
    for record in frame.to_dict("records"):
        records.append(
            {
                key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                for key, value in record.items()
            }
        )
    return records


def _compact_replay_config(
    tmp_path: Path,
    model: ModelSpecConfig | None = None,
    config_path: Path = DEFAULT_CONFIG,
) -> PipelineConfig:
    base = PipelineConfig.from_yaml(config_path)
    return PipelineConfig(
        data=base.data,
        cleaning=base.cleaning,
        model=model or _lightgbm_replay_model(),
        evaluation=base.evaluation,
        postprocess=base.postprocess,
        output=base.output,
        scale=base.scale,
    )


def _replay_artifacts(config: PipelineConfig) -> tuple[Path, Path]:
    model_name = _model_type(config.model.name)
    scale_name = config.scale.name
    if model_name == "lightgbm" and scale_name == "day_ahead":
        return Path("outputs/lightgbm/day_ahead/model/model.joblib"), Path(
            "outputs/lightgbm/day_ahead/artifacts/forecast.csv"
        )
    if model_name == "lightgbm" and scale_name == "short_term_4h":
        return Path("outputs/lightgbm/short_term_4h/model/model.joblib"), Path(
            "outputs/lightgbm/short_term_4h/artifacts/forecast.csv"
        )
    if model_name == "gmm" and scale_name == "day_ahead":
        return Path("outputs/gmm/day_ahead/e2e_probability_plots/model/model.joblib"), Path(
            "outputs/gmm/day_ahead/e2e_probability_plots/artifacts/forecast.csv"
        )
    if model_name == "lstm" and scale_name == "day_ahead":
        return Path("outputs/lstm/day_ahead/model/model.pt"), Path("outputs/lstm/day_ahead/artifacts/forecast.csv")
    if model_name == "autogluon" and scale_name == "day_ahead":
        return Path("outputs/autogluon/day_ahead/model"), Path("outputs/autogluon/day_ahead/artifacts/forecast.csv")
    raise ValueError(f"Unsupported replay artifacts for model_name={model_name} scale_name={scale_name}")


def _lightgbm_replay_model() -> ModelSpecConfig:
    return ModelSpecConfig(
        name="lightgbm",
        random_state=42,
        params={
            "n_estimators": 20,
            "learning_rate": 0.05,
            "objective": "regression_l1",
            "num_leaves": 15,
            "enable_quantiles": True,
            "lower_quantile": 0.1,
            "upper_quantile": 0.9,
        },
    )


def _gmm_replay_model() -> ModelSpecConfig:
    return ModelSpecConfig(
        name="gmm",
        random_state=42,
        params={
            "n_components": 2,
            "point_estimator": "hist_gradient_boosting",
            "point_params": {
                "max_iter": 20,
                "max_leaf_nodes": 15,
                "learning_rate": 0.08,
            },
            "distribution_grid_size": 25,
            "distribution_std_width": 4.0,
            "gmm_max_iter": 80,
            "n_init": 1,
        },
    )


def _model_type(model_name: str) -> str:
    if model_name in {"lightgbm", "lgbm"}:
        return "lightgbm"
    if model_name in {"gmm", "gaussian_mixture", "gaussian_mixture_model"}:
        return "gmm"
    if model_name in {"lstm", "torch_lstm", "bilstm", "bi_lstm"}:
        return "lstm"
    if model_name in {"autogluon", "autogluon_timeseries"}:
        return "autogluon"
    return model_name


def _distribution_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    return np.asarray([json.loads(value) for value in frame[column]], dtype=float)


def _assert_forecast_frames_allclose(offline: pd.DataFrame, online: pd.DataFrame) -> None:
    numeric_columns = [
        "prediction",
        "0.1",
        "0.9",
        "prediction_lower",
        "prediction_upper",
        "prediction_density",
    ]
    for column in numeric_columns:
        np.testing.assert_allclose(
            online[column].to_numpy(dtype=float),
            offline[column].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Mismatch in forecast column={column}",
        )
    for column in ["distribution_values", "distribution_probabilities"]:
        np.testing.assert_allclose(
            _distribution_matrix(online, column),
            _distribution_matrix(offline, column),
            rtol=1e-9,
            atol=1e-9,
            err_msg=f"Mismatch in distribution grid column={column}",
        )


def _history_for_replay_window(
    cleaned_data,
    window_index: int,
    offline_forecast: pd.DataFrame,
) -> pd.DataFrame:
    window = offline_forecast[offline_forecast["evaluation_window"] == window_index].sort_values(
        ["item_id", "timestamp"]
    )
    start_timestamp = pd.Timestamp(window["timestamp"].min())
    item_id = str(window["item_id"].iloc[0])
    history = cleaned_data.frame[
        (cleaned_data.frame[cleaned_data.item_id_col].astype(str) == item_id)
        & (cleaned_data.frame[cleaned_data.timestamp_col] < start_timestamp)
    ].copy()
    return history


def _known_covariates_for_replay_window(
    config: PipelineConfig,
    cleaned_data,
    offline_window: pd.DataFrame,
    window_index: int,
) -> pd.DataFrame:
    item_id = str(offline_window["item_id"].iloc[0])
    timestamps = pd.to_datetime(offline_window["timestamp"])
    window = cleaned_data.frame[
        (cleaned_data.frame[cleaned_data.item_id_col].astype(str) == item_id)
        & (cleaned_data.frame[cleaned_data.timestamp_col].isin(timestamps))
    ].copy()
    return window[
        [
            cleaned_data.timestamp_col,
            cleaned_data.item_id_col,
            *config.scale.feature.known_covariates,
        ]
    ]


def _online_forecast_for_replay_window(
    config: PipelineConfig,
    cleaned_data,
    offline_forecast: pd.DataFrame,
    window_index: int,
    model_path: Path,
) -> pd.DataFrame:
    offline_window = offline_forecast[offline_forecast["evaluation_window"] == window_index].sort_values(
        ["item_id", "timestamp"]
    )
    prediction_length = int(offline_window["timestamp"].nunique())
    history = _history_for_replay_window(cleaned_data, window_index, offline_forecast)
    known_covariates = _known_covariates_for_replay_window(
        config,
        cleaned_data,
        offline_window,
        window_index,
    )
    request = OnlineInferenceRequest(
        request_id=None,
        request_time=None,
        item_id=str(offline_window["item_id"].iloc[0]),
        task=OnlineInferenceTask(
            task_type="load_forecast",
            forecast_type="point",
            forecast_scale=config.scale.name,
            prediction_length=prediction_length,
            freq=config.scale.freq,
        ),
        history_load=_history_load_records(
            history,
            timestamp_col=cleaned_data.timestamp_col,
            target_col=cleaned_data.target_col,
            item_id_col=cleaned_data.item_id_col,
        ),
        future_covariates=_future_covariate_records(
            known_covariates,
            timestamp_col=cleaned_data.timestamp_col,
            target_col=cleaned_data.target_col,
            item_id_col=cleaned_data.item_id_col,
        ),
        forecast_options={},
        trace={},
    )

    response = run_online_inference(request, model_path=model_path)
    online = pd.DataFrame(response.forecast)
    online["timestamp"] = pd.to_datetime(online["timestamp"])
    return online.sort_values(["item_id", "timestamp"])


def _history_load_records(
    frame: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    item_id_col: str,
) -> list[dict[str, object]]:
    payload = frame.drop(columns=[item_id_col]).rename(columns={timestamp_col: "timestamp", target_col: "actual_load"})
    return _json_records(payload)


def _future_covariate_records(
    frame: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    item_id_col: str,
) -> list[dict[str, object]]:
    payload = frame.drop(columns=[target_col, item_id_col], errors="ignore").rename(
        columns={timestamp_col: "timestamp"}
    )
    return _json_records(payload)


def _assert_replay_ten_rolling_windows_allclose(config: PipelineConfig) -> None:
    model_path, forecast_csv_path = _replay_artifacts(config)
    assert model_path.exists()
    assert forecast_csv_path.exists()

    data = CSVLoadDataLoader(config.data).load(Path(config.data.path))
    cleaned = TimeSeriesCleaner(config.cleaning).fit_transform(data)
    offline_forecast = pd.read_csv(forecast_csv_path)
    offline_forecast["timestamp"] = pd.to_datetime(offline_forecast["timestamp"])

    available_windows = sorted(int(value) for value in offline_forecast["evaluation_window"].dropna().unique())
    assert len(available_windows) >= REPLAY_WINDOW_COUNT
    sampled_windows = np.random.default_rng(42).choice(available_windows, size=REPLAY_WINDOW_COUNT, replace=False)
    sampled_windows = sorted(int(value) for value in sampled_windows)

    for window_index in sampled_windows:
        offline_window = offline_forecast[offline_forecast["evaluation_window"] == window_index].sort_values(
            ["item_id", "timestamp"]
        )

        online = _online_forecast_for_replay_window(
            config,
            cleaned,
            offline_forecast,
            window_index,
            model_path,
        )
        assert online[["item_id", "timestamp"]].reset_index(drop=True).equals(
            offline_window[["item_id", "timestamp"]].reset_index(drop=True)
        )
        _assert_forecast_frames_allclose(
            offline_window.reset_index(drop=True),
            online.reset_index(drop=True),
        )


def run_lightgbm_replay(tmp_path):
    config = _compact_replay_config(tmp_path, model=_lightgbm_replay_model())
    _assert_replay_ten_rolling_windows_allclose(config)


def run_short_term_4h_lightgbm_replay(tmp_path):
    config = _compact_replay_config(
        tmp_path,
        model=_lightgbm_replay_model(),
        config_path=SHORT_TERM_4H_CONFIG,
    )
    _assert_replay_ten_rolling_windows_allclose(config)


def run_gmm_replay(tmp_path):
    config = _compact_replay_config(tmp_path, model=_gmm_replay_model())
    _assert_replay_ten_rolling_windows_allclose(config)


def _lstm_replay_model() -> ModelSpecConfig:
    return ModelSpecConfig(
        name="lstm",
        random_state=42,
        params={
            "context_length": 672,
            "hidden_size": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 20,
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_train_samples": 4096,
            "train_sample_strategy": "last",
            "device": "cpu",
            "validation_fraction": 0.1,
            "early_stopping_patience": 3,
            "early_stopping_min_delta": 1e-4,
            "use_point_head": False,
            "nll_loss_weight": 1.0,
            "point_loss_weight": 0.0,
        },
    )


def _autogluon_replay_model() -> ModelSpecConfig:
    return ModelSpecConfig(
        name="autogluon",
        random_state=42,
        params={
            "eval_metric": "MAE",
            "presets": "medium_quality",
            "time_limit": 300,
            "enable_ensemble": False,
            "quantile_levels": [0.1, 0.9],
            "hyperparameters": {
                "RecursiveTabular": {
                    "model_name": "GBM",
                },
            },
        },
    )


def run_lstm_replay(tmp_path):
    config = _compact_replay_config(tmp_path, model=_lstm_replay_model())
    _assert_replay_ten_rolling_windows_allclose(config)


def run_autogluon_replay(tmp_path):
    config = _compact_replay_config(tmp_path, model=_autogluon_replay_model())
    _assert_replay_ten_rolling_windows_allclose(config)
