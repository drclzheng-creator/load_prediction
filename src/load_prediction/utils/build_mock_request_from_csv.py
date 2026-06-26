"""Build online inference request JSON files from historical CSV data."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import NAMESPACE_URL, uuid5

import pandas as pd

from load_prediction.configs import PipelineConfig
from load_prediction.data import CSVLoadDataLoader

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
    config_path: str | Path = "configs/day_ahead.yaml",
    model_type: str = "lightgbm",
    prediction_length: int = 96,
    freq: str = "15min",
    history_length: int | None = 900,
    expected_known_covariates: Iterable[str] = DEFAULT_KNOWN_COVARIATES,
) -> dict[str, Any]:
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

    item_ids = set(history_frame[data.item_id_col].dropna().astype(str)).union(
        set(known_frame[data.item_id_col].dropna().astype(str))
    )
    if len(item_ids) != 1:
        raise ValueError(f"Mock online inference request requires one item_id, got {sorted(item_ids)}")
    item_id = next(iter(item_ids))
    request_id = _request_id(
        csv_path=csv_path,
        model_type=model_type,
        item_id=item_id,
        prediction_length=prediction_length,
        freq=freq,
        forecast_start=known_frame[data.timestamp_col].iloc[0],
    )

    return {
        "request_id": request_id,
        "request_time": _timestamp_to_iso(history_frame[data.timestamp_col].iloc[-1]),
        "item_id": item_id,
        "task": {
            "task_type": "load_forecast",
            "forecast_type": "probability",
            "forecast_scale": config.scale.name,
            "prediction_length": prediction_length,
            "freq": freq,
        },
        "history_load": _history_load_records(
            history_frame,
            timestamp_col=data.timestamp_col,
            target_col=data.target_col,
            item_id_col=data.item_id_col,
        ),
        "future_covariates": _future_covariate_records(
            known_frame,
            timestamp_col=data.timestamp_col,
            target_col=data.target_col,
            item_id_col=data.item_id_col,
        ),
        "forecast_options": {
            "enable_probabilistic": True,
            "quantile_levels": list(config.model.params.get("quantile_levels", [])),
        },
        "trace": {
            "source_system": "load_prediction_mock",
            "trace_id": request_id,
            "parent_trace_id": None,
            "feature_version": config.scale.name,
            "schema_version": "load_forecast.v2",
        },
    }


def save_prediction_request_json(
    request: Any,
    path: str | Path,
) -> Path:
    """Save an online inference request as JSON."""

    request_path = Path(path)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(request) if is_dataclass(request) else request
    request_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return request_path


def build_and_save_mock_request_from_csv(
    csv_path: str | Path,
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


def _history_load_records(
    frame: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    item_id_col: str,
) -> list[dict[str, object]]:
    payload = frame.drop(columns=[item_id_col]).rename(
        columns={timestamp_col: "timestamp", target_col: "actual_load"}
    )
    return _json_records(payload)


def _future_covariate_records(
    frame: pd.DataFrame,
    timestamp_col: str,
    target_col: str,
    item_id_col: str,
) -> list[dict[str, object]]:
    payload = frame.drop(columns=[target_col, item_id_col]).rename(columns={timestamp_col: "timestamp"})
    return _json_records(payload)


def _request_id(
    csv_path: str | Path,
    model_type: str,
    item_id: str,
    prediction_length: int,
    freq: str,
    forecast_start: object,
) -> str:
    key = "|".join(
        [
            str(Path(csv_path)),
            model_type,
            item_id,
            str(prediction_length),
            freq,
            _timestamp_to_iso(forecast_start),
        ]
    )
    return str(uuid5(NAMESPACE_URL, key))


def _timestamp_to_iso(value: object) -> str:
    return value.isoformat() if isinstance(value, pd.Timestamp) else str(value)
