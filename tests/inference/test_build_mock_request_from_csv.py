import json
from pathlib import Path

import pytest

from load_prediction.inference import load_online_inference_request
from load_prediction.utils.build_mock_request_from_csv import (
    DEFAULT_KNOWN_COVARIATES,
    build_and_save_mock_request_from_csv,
)
from tests.inference.inference_helpers import (
    AUTOGLUON_MODEL,
    BILSTM_MODEL,
    DEFAULT_CONFIG,
    DEFAULT_INPUT_CSV,
    GMM_MODEL,
    LIGHTGBM_MODEL,
    LSTM_MODEL,
    REQUEST_INPUT_DIR,
    SHORT_TERM_4H_CONFIG,
    SHORT_TERM_4H_LIGHTGBM_MODEL,
    SKLEARN_MODEL,
)


REQUEST_CASES = [
    pytest.param(
        "lightgbm",
        LIGHTGBM_MODEL,
        DEFAULT_CONFIG,
        96,
        "15min",
        900,
        id="lightgbm_day_ahead",
    ),
    pytest.param(
        "lightgbm",
        SHORT_TERM_4H_LIGHTGBM_MODEL,
        SHORT_TERM_4H_CONFIG,
        16,
        "15min",
        900,
        id="lightgbm_short_term_4h",
    ),
    pytest.param(
        "sklearn",
        SKLEARN_MODEL,
        DEFAULT_CONFIG,
        96,
        "15min",
        900,
        id="sklearn_day_ahead",
    ),
    pytest.param(
        "gmm",
        GMM_MODEL,
        DEFAULT_CONFIG,
        96,
        "15min",
        900,
        id="gmm_day_ahead",
    ),
    pytest.param(
        "autogluon",
        AUTOGLUON_MODEL,
        DEFAULT_CONFIG,
        96,
        "15min",
        900,
        id="autogluon_day_ahead",
    ),
    pytest.param(
        "lstm",
        LSTM_MODEL,
        DEFAULT_CONFIG,
        96,
        "15min",
        1344,
        id="lstm_day_ahead",
    ),
    pytest.param(
        "bilstm",
        BILSTM_MODEL,
        DEFAULT_CONFIG,
        96,
        "15min",
        1344,
        id="bilstm_day_ahead",
    ),
]


@pytest.mark.parametrize(
    "model_type,model_path,config_path,prediction_length,freq,history_length",
    REQUEST_CASES,
)
def test_build_mock_request_from_csv_generates_prediction_request_json(
    model_type: str,
    model_path: Path,
    config_path: Path,
    prediction_length: int,
    freq: str,
    history_length: int,
):
    if not DEFAULT_INPUT_CSV.exists() or not model_path.exists():
        pytest.skip(f"Saved model or input CSV is not available for model_type={model_type}")

    output_path = REQUEST_INPUT_DIR / f"{model_type}_{prediction_length}_{freq}_prediction_request.json"
    request_path = build_and_save_mock_request_from_csv(
        csv_path=DEFAULT_INPUT_CSV,
        output_path=output_path,
        config_path=config_path,
        model_type=model_type,
        prediction_length=prediction_length,
        freq=freq,
        history_length=history_length,
        expected_known_covariates=DEFAULT_KNOWN_COVARIATES,
    )

    assert request_path == output_path
    assert request_path.exists()
    values = json.loads(request_path.read_text(encoding="utf-8"))
    assert values["task"]["prediction_length"] == prediction_length
    assert values["task"]["freq"] == freq
    assert values["task"]["task_type"] == "load_forecast"
    assert values["request_id"]
    assert values["request_time"]
    assert len(values["history_load"]) == history_length
    assert len(values["future_covariates"]) == prediction_length
    assert DEFAULT_KNOWN_COVARIATES.issubset(values["history_load"][0])
    assert DEFAULT_KNOWN_COVARIATES.issubset(values["future_covariates"][0])
    assert "actual_load" in values["history_load"][0]
    assert "target" not in values["history_load"][0]
    assert "target" not in values["future_covariates"][0]
    assert "item_id" not in values["history_load"][0]
    assert "item_id" not in values["future_covariates"][0]
    assert values["trace"]["schema_version"] == "load_forecast.v2"
    assert "model_hint" not in values["trace"]

    request = load_online_inference_request(request_path)
    assert request.item_id
    assert request.task.forecast_scale == values["task"]["forecast_scale"]
    assert request.task.prediction_length == prediction_length
