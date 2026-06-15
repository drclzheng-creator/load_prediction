from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest

from load_prediction.configs import ModelSpecConfig, PipelineConfig
from load_prediction.data import CSVLoadDataLoader
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.inference import (
    OnlineInferenceRequest,
    OnlineInferenceResponse,
    load_forecast_model,
    normalize_forecast_frame,
    validate_online_request,
)
from load_prediction.inference.model_manifest import ModelManifest, build_model_manifest, load_model_manifest
from load_prediction.postprocessing import ForecastPostProcessor

logger = logging.getLogger(__name__)


DEFAULT_INPUT_CSV = Path("inputs/EWELD/eweld_industrial_u141_2018_2019_15min.csv")
DEFAULT_CONFIG = Path("configs/day_ahead.yaml")
SHORT_TERM_4H_CONFIG = Path("configs/short_term_4h.yaml")
EXPECTED_KNOWN_COVARIATES = {
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_direction",
    "wind_speed_mph",
    "wind_gust_mph",
    "pressure_in",
}
LSTM_REQUIRED_HISTORY_LENGTH = 672 + 672
LIGHTGBM_MODEL = Path("outputs/lightgbm/day_ahead/model/model.joblib")
SHORT_TERM_4H_LIGHTGBM_MODEL = Path("outputs/lightgbm/short_term_4h/model/model.joblib")
SKLEARN_MODEL = Path("outputs/sklearn_hist_gradient_boosting/day_ahead/model/model.joblib")
GMM_MODEL = Path("outputs/gmm/day_ahead/e2e_probability_plots/model/model.joblib")
AUTOGLUON_MODEL = Path("outputs/autogluon/day_ahead/model")
LSTM_MODEL = Path("outputs/lstm/day_ahead/model/model.pt")
BILSTM_MODEL = Path("outputs/bilstm/day_ahead/model/model.pt")


def _build_mock_request_from_csv(
    csv_path: Path,
    model_path: Path,
    model_type: str = "lightgbm",
    prediction_length: int = 96,
    freq: str = "15min",
    history_length: int | None = 900,
) -> OnlineInferenceRequest:
    config = PipelineConfig.from_yaml(DEFAULT_CONFIG)
    data_config = config.data.__class__(
        **{
            **config.data.__dict__,
            "path": str(csv_path),
            "freq": freq,
            "start_time": None,
            "end_time": None,
        }
    )
    data = CSVLoadDataLoader(data_config).load(csv_path)
    frame = data.frame.sort_values([data.item_id_col, data.timestamp_col]).copy()
    if len(frame) <= prediction_length:
        raise AssertionError("Input CSV is too short for mock online inference")
    missing_covariates = EXPECTED_KNOWN_COVARIATES - set(frame.columns)
    if missing_covariates:
        raise AssertionError(
            f"Mock request source frame missing covariates: {sorted(missing_covariates)}"
        )

    history_end = len(frame) - prediction_length
    history_start = 0 if history_length is None else max(0, history_end - history_length)
    history_frame = frame.iloc[history_start:history_end].copy()
    known_frame = frame.iloc[history_end : history_end + prediction_length].copy()
    known_frame = known_frame.drop(columns=[data.target_col])

    return OnlineInferenceRequest(
        model_path=str(model_path),
        model_type=model_type,
        history=_json_records(history_frame),
        known_covariates=_json_records(known_frame),
        prediction_length=prediction_length,
        freq=freq,
        timestamp_col=data.timestamp_col,
        target_col=data.target_col,
        item_id_col=data.item_id_col,
    )


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


def _history_dataset_from_request(request: OnlineInferenceRequest) -> TimeSeriesDataset:
    return TimeSeriesDataset(
        frame=pd.DataFrame(request.history),
        timestamp_col=request.timestamp_col,
        target_col=request.target_col,
        item_id_col=request.item_id_col,
        freq=request.freq,
    )


def _known_covariates_from_request(request: OnlineInferenceRequest) -> pd.DataFrame:
    frame = pd.DataFrame(request.known_covariates)
    frame[request.timestamp_col] = pd.to_datetime(frame[request.timestamp_col])
    return frame


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


def _run_saved_model_inference_case(model_type: str, model_path: Path) -> None:
    _run_saved_model_inference_case_with_horizon(
        model_type=model_type,
        model_path=model_path,
        prediction_length=96,
        freq="15min",
        history_length=LSTM_REQUIRED_HISTORY_LENGTH if model_type in {"lstm", "bilstm"} else 900,
    )


def _run_saved_model_inference_case_with_horizon(
    model_type: str,
    model_path: Path,
    prediction_length: int,
    freq: str,
    history_length: int | None,
) -> None:
    if not DEFAULT_INPUT_CSV.exists() or not model_path.exists():
        pytest.skip(f"Saved model or input CSV is not available for model_type={model_type}")

    total_started = perf_counter()
    request_started = perf_counter()
    request = _build_mock_request_from_csv(
        csv_path=DEFAULT_INPUT_CSV,
        model_path=model_path,
        model_type=model_type,
        prediction_length=prediction_length,
        freq=freq,
        history_length=history_length,
    )
    request_seconds = perf_counter() - request_started

    load_started = perf_counter()
    model = load_forecast_model(request.model_path, request.model_type)
    manifest = load_model_manifest(request.model_path)
    validate_online_request(request, manifest)
    load_seconds = perf_counter() - load_started

    history = _history_dataset_from_request(request)
    known_covariates = _known_covariates_from_request(request)
    predict_started = perf_counter()
    raw_forecast = model.predict(
        history,
        prediction_length=request.prediction_length,
        freq=request.freq,
        known_covariates=known_covariates,
    )
    raw_forecast = normalize_forecast_frame(raw_forecast)
    processed_forecast = ForecastPostProcessor(request.postprocess).transform(
        raw_forecast,
        residual_std=float(getattr(model, "residual_std_", 0.0)),
    )
    predict_seconds = perf_counter() - predict_started
    total_seconds = perf_counter() - total_started
    logger.info(
        "online_inference_timing model_type=%s request_seconds=%.6f load_seconds=%.6f predict_seconds=%.6f total_seconds=%.6f rows=%s",
        model_type,
        request_seconds,
        load_seconds,
        predict_seconds,
        total_seconds,
        len(processed_forecast.frame),
    )
    response = OnlineInferenceResponse(
        forecast=_json_records(processed_forecast.frame),
        prediction_length=request.prediction_length,
        freq=request.freq,
        model_type=request.model_type,
    )
    _log_inference_response(response)
    _assert_inference_response(request, response, model_type)



def _assert_inference_response(
    request: OnlineInferenceRequest,
    response,
    model_type: str,
) -> None:
    assert response.model_type == model_type
    assert response.prediction_length == request.prediction_length
    assert response.freq == request.freq
    assert len(response.forecast) == request.prediction_length
    first = response.forecast[0]
    first_history = request.history[0]
    first_known_covariates = request.known_covariates[0]
    assert EXPECTED_KNOWN_COVARIATES.issubset(first_history)
    assert EXPECTED_KNOWN_COVARIATES.issubset(first_known_covariates)
    assert "target" in first_history
    assert "target" not in first_known_covariates
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


def run_lightgbm_saved_model_inference():
    _run_saved_model_inference_case("lightgbm", LIGHTGBM_MODEL)


def run_short_term_4h_lightgbm_saved_model_inference():
    _run_saved_model_inference_case_with_horizon(
        model_type="lightgbm",
        model_path=SHORT_TERM_4H_LIGHTGBM_MODEL,
        prediction_length=16,
        freq="15min",
        history_length=900,
    )


def run_sklearn_saved_model_inference():
    _run_saved_model_inference_case("sklearn", SKLEARN_MODEL)


def run_gmm_saved_model_inference():
    _run_saved_model_inference_case("gmm", GMM_MODEL)


def run_autogluon_saved_model_inference():
    _run_saved_model_inference_case("autogluon", AUTOGLUON_MODEL)


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


def run_lstm_saved_model_inference():
    _run_saved_model_inference_case("lstm", LSTM_MODEL)


def run_bilstm_saved_model_inference():
    _run_saved_model_inference_case("bilstm", BILSTM_MODEL)
