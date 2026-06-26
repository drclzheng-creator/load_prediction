from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import pytest

from load_prediction.configs import ModelSpecConfig, PipelineConfig
from load_prediction.inference import (
    OnlineInferenceResponse,
    load_online_inference_request,
    run_online_inference_from_json,
)
from load_prediction.inference.model_manifest import build_model_manifest
from load_prediction.utils.build_mock_request_from_csv import DEFAULT_KNOWN_COVARIATES

logger = logging.getLogger(__name__)


DEFAULT_INPUT_CSV = Path("inputs/EWELD/eweld_industrial_u141_2018_2019_15min.csv")
DEFAULT_CONFIG = Path("configs/day_ahead.yaml")
SHORT_TERM_4H_CONFIG = Path("configs/short_term_4h.yaml")
EXPECTED_KNOWN_COVARIATES = DEFAULT_KNOWN_COVARIATES
REQUEST_INPUT_DIR = Path("inputs/request")
INFERENCE_OUTPUT_DIR = Path("outputs/inference")
LSTM_REQUIRED_HISTORY_LENGTH = 672 + 672
LIGHTGBM_MODEL = Path("outputs/lightgbm/day_ahead/model/model.joblib")
SHORT_TERM_4H_LIGHTGBM_MODEL = Path("outputs/lightgbm/short_term_4h/model/model.joblib")
SKLEARN_MODEL = Path("outputs/sklearn_hist_gradient_boosting/day_ahead/model/model.joblib")
GMM_MODEL = Path("outputs/gmm/day_ahead/e2e_probability_plots/model/model.joblib")
AUTOGLUON_MODEL = Path("outputs/autogluon/day_ahead/model")
LSTM_MODEL = Path("outputs/lstm/day_ahead/model/model.pt")
BILSTM_MODEL = Path("outputs/bilstm/day_ahead/model/model.pt")


def _with_model(config: PipelineConfig, model: ModelSpecConfig) -> PipelineConfig:
    return PipelineConfig(
        data=config.data,
        cleaning=config.cleaning,
        model=model,
        evaluation=config.evaluation,
        postprocess=config.postprocess,
        output=config.output,
        scale=config.scale,
    )


def _run_request_json_inference_case(model_type: str, model_path: Path) -> None:
    _run_request_json_inference_case_with_horizon(
        model_type=model_type,
        model_path=model_path,
        prediction_length=96,
        freq="15min",
        history_length=LSTM_REQUIRED_HISTORY_LENGTH if model_type in {"lstm", "bilstm"} else 900,
    )


def _run_request_json_inference_case_with_horizon(
    model_type: str,
    model_path: Path,
    prediction_length: int,
    freq: str,
    history_length: int | None,
) -> None:
    if not model_path.exists():
        pytest.skip(f"Saved model is not available for model_type={model_type}: {model_path}")

    total_started = perf_counter()
    request_started = perf_counter()
    request_path = REQUEST_INPUT_DIR / f"{model_type}_{prediction_length}_{freq}_prediction_request.json"
    if not request_path.exists():
        pytest.skip(f"Online request JSON is not available: {request_path}")
    request = load_online_inference_request(request_path)
    request_seconds = perf_counter() - request_started

    response_path = INFERENCE_OUTPUT_DIR / f"{model_type}_{prediction_length}_{freq}_response.json"
    response = run_online_inference_from_json(request_path, model_path=model_path, output_path=response_path)
    total_seconds = perf_counter() - total_started
    logger.info(
        "online_inference_timing model_type=%s request_seconds=%.6f total_seconds=%.6f rows=%s request_json=%s response_json=%s",
        model_type,
        request_seconds,
        total_seconds,
        len(response.forecast),
        request_path,
        response_path,
    )
    _log_inference_response(response)
    _assert_inference_response(request, response, model_type)
    _assert_response_json(response_path, response)


def _assert_inference_response(
    request,
    response: OnlineInferenceResponse,
    model_type: str,
) -> None:
    assert response.model_type == model_type
    assert response.prediction_length == request.task.prediction_length
    assert response.freq == request.task.freq
    assert response.request_id == request.request_id
    assert len(response.forecast) == request.task.prediction_length
    first = response.forecast[0]
    first_history = request.history_load[0]
    first_known_covariates = request.future_covariates[0]
    assert EXPECTED_KNOWN_COVARIATES.issubset(first_history)
    assert EXPECTED_KNOWN_COVARIATES.issubset(first_known_covariates)
    assert "actual_load" in first_history
    assert "actual_load" not in first_known_covariates
    expected_columns = {
        "item_id",
        "timestamp",
        "horizon_step",
        "prediction",
        "prediction_lower",
        "prediction_upper",
        "distribution_values",
        "distribution_probabilities",
        "prediction_density",
    }
    assert expected_columns.issubset(first)
    values = json.loads(first["distribution_values"])
    probabilities = json.loads(first["distribution_probabilities"])
    assert len(values) == len(probabilities)
    assert first["prediction"] >= 0


def _assert_response_json(path: Path, response: OnlineInferenceResponse) -> None:
    assert path.exists()
    values = json.loads(path.read_text(encoding="utf-8"))
    assert values["code"] == 0
    assert values["request_id"] == response.request_id
    assert values["response_id"] == response.response_id
    assert values["response_id"] == values["request_id"]
    assert values["task_status"] == "COMPLETED"
    assert values["item_id"] == response.item_id
    assert values["model"]["model_type"] == response.model_type
    assert values["model"]["model_name"] == response.model_name
    assert values["model"]["model_version"] == response.model_version
    assert values["task"]["task_type"] == "load_forecast"
    assert values["task"]["forecast_type"] == response.forecast_type
    assert values["task"]["forecast_scale"] == response.forecast_scale
    assert values["task"]["prediction_length"] == response.prediction_length
    assert values["task"]["freq"] == response.freq
    assert isinstance(values["summary"]["model_inference_time_ms"], float)
    assert values["summary"]["model_inference_time_ms"] >= 0
    assert len(values["results"]) == len(response.forecast)
    assert "item_id" not in values["results"][0]


def _log_inference_response(response: OnlineInferenceResponse) -> None:
    sample = response.forecast[0] if response.forecast else {}
    logger.info(
        "online_inference_response model_type=%s prediction_length=%s freq=%s rows=%s sample=%s",
        response.model_type,
        response.prediction_length,
        response.freq,
        len(response.forecast),
        sample,
    )


def _load_request_for_model(model_type: str) -> tuple[object, object]:
    request_path = REQUEST_INPUT_DIR / f"{model_type}_96_15min_prediction_request.json"
    if not request_path.exists():
        pytest.skip(f"Online request JSON is not available: {request_path}")
    return load_online_inference_request(request_path), request_path


def run_lightgbm_request_json_inference():
    _run_request_json_inference_case("lightgbm", LIGHTGBM_MODEL)


def run_short_term_4h_lightgbm_request_json_inference():
    _run_request_json_inference_case_with_horizon(
        model_type="lightgbm",
        model_path=SHORT_TERM_4H_LIGHTGBM_MODEL,
        prediction_length=16,
        freq="15min",
        history_length=900,
    )


def run_sklearn_request_json_inference():
    _run_request_json_inference_case("sklearn", SKLEARN_MODEL)


def run_gmm_request_json_inference():
    _run_request_json_inference_case("gmm", GMM_MODEL)


def run_autogluon_request_json_inference():
    _run_request_json_inference_case("autogluon", AUTOGLUON_MODEL)


def run_lstm_manifest_requires_context_plus_lag_history():
    config = _with_model(
        PipelineConfig.from_yaml(DEFAULT_CONFIG),
        ModelSpecConfig(
            name="lstm",
            params={
                "context_length": 672,
                "add_lag_features": True,
                "lag_feature_steps": (96, 192, 672),
            },
        ),
    )

    manifest = build_model_manifest(config)

    assert manifest.required_history_length == LSTM_REQUIRED_HISTORY_LENGTH


def run_lstm_request_json_inference():
    _run_request_json_inference_case("lstm", LSTM_MODEL)


def run_bilstm_request_json_inference():
    _run_request_json_inference_case("bilstm", BILSTM_MODEL)
