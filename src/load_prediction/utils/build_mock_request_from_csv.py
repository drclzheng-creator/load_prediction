"""Build online inference request JSON files from historical CSV data."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from load_prediction.configs import PipelineConfig
from load_prediction.data import CSVLoadDataLoader
from load_prediction.inference import OnlineInferenceRequest


DEFAULT_KNOWN_COVARIATES = {
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_direction",
    "wind_speed_mph",
    "wind_gust_mph",
    "pressure_in",
}


def build_mock_request_from_csv(
    csv_path: str | Path,
    model_path: str | Path,
    config_path: str | Path = "configs/day_ahead.yaml",
    model_type: str = "lightgbm",
    prediction_length: int = 96,
    freq: str = "15min",
    history_length: int | None = 900,
    expected_known_covariates: Iterable[str] = DEFAULT_KNOWN_COVARIATES,
) -> OnlineInferenceRequest:
    """Build an online inference request from the tail of a CSV dataset."""

    config = PipelineConfig.from_yaml(config_path)
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
        raise ValueError("Input CSV is too short for mock online inference")

    missing_covariates = set(expected_known_covariates).difference(frame.columns)
    if missing_covariates:
        raise ValueError(f"Source frame missing covariates: {sorted(missing_covariates)}")

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


def save_prediction_request_json(
    request: OnlineInferenceRequest,
    path: str | Path,
) -> Path:
    """Save an online inference request as JSON."""

    request_path = Path(path)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        json.dumps(asdict(request), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return request_path


def build_and_save_mock_request_from_csv(
    csv_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    config_path: str | Path = "configs/day_ahead.yaml",
    model_type: str = "lightgbm",
    prediction_length: int = 96,
    freq: str = "15min",
    history_length: int | None = 900,
    expected_known_covariates: Iterable[str] = DEFAULT_KNOWN_COVARIATES,
) -> Path:
    """Build and save a mock online inference request JSON from CSV data."""

    request = build_mock_request_from_csv(
        csv_path=csv_path,
        model_path=model_path,
        config_path=config_path,
        model_type=model_type,
        prediction_length=prediction_length,
        freq=freq,
        history_length=history_length,
        expected_known_covariates=expected_known_covariates,
    )
    return save_prediction_request_json(request, output_path)


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
