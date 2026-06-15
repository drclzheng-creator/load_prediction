from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_prediction.configs import ModelSpecConfig, ArtifactConfig, PipelineConfig
from load_prediction.data import CSVLoadDataLoader
from load_prediction.inference import OnlineInferenceRequest, run_online_inference
from load_prediction.pipeline.forecast_pipeline import LoadForecastPipeline


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
        data=base.data.__class__(
            **{
                **base.data.__dict__,
                "start_time": "2019-01-01 00:00:00",
                "end_time": "2019-01-31 23:45:00",
            }
        ),
        cleaning=base.cleaning,
        model=model or _lightgbm_replay_model(),
        evaluation=base.evaluation.__class__(
            **{
                **base.evaluation.__dict__,
                "split_strategy": "ratio",
                "train_ratio": None,
                "holdout_length": base.scale.prediction_length * REPLAY_WINDOW_COUNT,
            }
        ),
        postprocess=base.postprocess,
        output=ArtifactConfig(
            root_dir=str(tmp_path),
            save_model=True,
            save_artifacts=True,
            save_plot=False,
            save_metrics=True,
        ),
        scale=base.scale,
    )


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


def _history_for_replay_window(result, window_index: int, prediction_length: int) -> pd.DataFrame:
    end = (window_index - 1) * prediction_length
    if end <= 0:
        return result.train_data.frame.copy()
    prior_test = result.test_data.frame.groupby(
        result.test_data.item_id_col,
        sort=False,
    ).head(end)
    return pd.concat([result.train_data.frame, prior_test], ignore_index=True)


def _known_covariates_for_replay_window(
    result,
    config: PipelineConfig,
    window_index: int,
    prediction_length: int,
) -> pd.DataFrame:
    start = (window_index - 1) * prediction_length
    end = start + prediction_length
    window = result.test_data.frame.iloc[start:end]
    return window[
        [
            result.test_data.timestamp_col,
            result.test_data.item_id_col,
            *config.scale.feature.known_covariates,
        ]
    ]


def _online_forecast_for_replay_window(
    result,
    config: PipelineConfig,
    window_index: int,
    prediction_length: int,
) -> pd.DataFrame:
    history = _history_for_replay_window(result, window_index, prediction_length)
    known_covariates = _known_covariates_for_replay_window(
        result,
        config,
        window_index,
        prediction_length,
    )
    request = OnlineInferenceRequest(
        model_path=str(result.model_path),
        model_type=_model_type(config.model.name),
        history=_json_records(history),
        known_covariates=_json_records(known_covariates),
        prediction_length=prediction_length,
        freq=config.scale.freq,
        timestamp_col=result.train_data.timestamp_col,
        target_col=result.train_data.target_col,
        item_id_col=result.train_data.item_id_col,
        postprocess=config.postprocess,
    )

    response = run_online_inference(request)
    online = pd.DataFrame(response.forecast)
    online[request.timestamp_col] = pd.to_datetime(online[request.timestamp_col])
    return online.sort_values([request.item_id_col, request.timestamp_col])


def _assert_replay_ten_rolling_windows_allclose(config: PipelineConfig) -> None:
    data = CSVLoadDataLoader(config.data).load(Path(config.data.path))
    result = LoadForecastPipeline(config).run(data)
    assert result.model_path is not None

    prediction_length = config.scale.prediction_length
    assert result.forecast.frame["evaluation_window"].nunique() == REPLAY_WINDOW_COUNT

    for window_index in range(1, REPLAY_WINDOW_COUNT + 1):
        offline_window = result.forecast.frame[
            result.forecast.frame["evaluation_window"] == window_index
        ].sort_values([result.forecast.item_id_col, result.forecast.timestamp_col])
        assert len(offline_window) == prediction_length

        online = _online_forecast_for_replay_window(
            result,
            config,
            window_index,
            prediction_length,
        )
        assert online[
            [result.forecast.item_id_col, result.forecast.timestamp_col]
        ].reset_index(drop=True).equals(
            offline_window[
                [result.forecast.item_id_col, result.forecast.timestamp_col]
            ].reset_index(drop=True)
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
