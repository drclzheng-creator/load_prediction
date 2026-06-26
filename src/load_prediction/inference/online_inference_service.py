"""Online inference helpers for saved forecasting models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import joblib
import pandas as pd
from pandas.tseries.frequencies import to_offset

from load_prediction.configs import DistributionGridConfig, ForecastPostprocessingConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.inference.model_manifest import ModelManifest, load_model_manifest
from load_prediction.models.autogluon_forecaster import AutoGluonTimeSeriesForecaster
from load_prediction.models.base_forecaster import BaseForecastModel, ForecastFrame
from load_prediction.models.lstm_forecaster import LSTMForecaster
from load_prediction.models.mdn_forecaster import MDNForecaster
from load_prediction.postprocessing import ForecastPostProcessor
from load_prediction.preprocessing import TimeSeriesCleaner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineInferenceTask:
    """Task configuration in the online inference request."""

    task_type: str = "load_forecast"
    forecast_type: str = "point"
    forecast_scale: str | None = None
    prediction_length: int = 96
    freq: str = "15min"


@dataclass(frozen=True)
class OnlineInferenceRequest:
    """Serializable request shape for one online forecast call."""

    request_id: str | None = None
    request_time: str | None = None
    item_id: str = ""
    task: OnlineInferenceTask = field(default_factory=OnlineInferenceTask)
    history_load: list[dict[str, Any]] = field(default_factory=list)
    future_covariates: list[dict[str, Any]] = field(default_factory=list)
    forecast_options: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnlineInferenceResponse:
    """Serializable response for online forecast calls."""

    forecast: list[dict[str, Any]]
    prediction_length: int
    freq: str
    model_type: str
    item_id: str | None = None
    request_id: str | None = None
    response_id: str | None = None
    model_version: str | None = None
    model_name: str | None = None
    scale_name: str | None = None
    response_time: str | None = None
    task_status: str = "COMPLETED"
    task_type: str = "load_forecast"
    forecast_type: str = "point"
    forecast_scale: str | None = None
    model_inference_time_ms: float | None = None
    warnings: list[str] = field(default_factory=list)


def load_online_inference_request(path: str | Path) -> OnlineInferenceRequest:
    """Load a serialized online inference request from JSON."""

    request_path = Path(path)
    values = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Online inference request JSON must be an object")
    return online_inference_request_from_dict(values)


def online_inference_request_from_dict(values: dict[str, Any]) -> OnlineInferenceRequest:
    """Build an online inference request from a JSON-compatible dictionary."""

    _reject_legacy_request_fields(values)
    item_id = str(values.get("item_id") or "").strip()
    if not item_id:
        raise ValueError("item_id is required")

    task_values = values.get("task") or {}
    if not isinstance(task_values, dict):
        raise ValueError("task must be an object")
    forecast_options = values.get("forecast_options") or {}
    trace = values.get("trace") or {}
    history_load = values.get("history_load") or []
    future_covariates = values.get("future_covariates") or []
    if not isinstance(forecast_options, dict):
        raise ValueError("forecast_options must be an object")
    if not isinstance(trace, dict):
        raise ValueError("trace must be an object")
    if not isinstance(history_load, list):
        raise ValueError("history_load must be an array")
    if not isinstance(future_covariates, list):
        raise ValueError("future_covariates must be an array")

    request_time = values.get("request_time")
    if request_time is not None:
        request_time = str(request_time)

    return OnlineInferenceRequest(
        request_id=str(values.get("request_id") or "") or None,
        request_time=request_time,
        item_id=item_id,
        task=_task_from_dict(task_values),
        history_load=history_load,
        future_covariates=future_covariates,
        forecast_options=forecast_options,
        trace=trace,
    )


def _task_from_dict(values: dict[str, Any]) -> OnlineInferenceTask:
    return OnlineInferenceTask(
        task_type=str(values.get("task_type") or "load_forecast"),
        forecast_type=str(values.get("forecast_type") or "point"),
        forecast_scale=str(values.get("forecast_scale") or "") or None,
        prediction_length=int(values.get("prediction_length", 96)),
        freq=str(values.get("freq", "15min")),
    )


def _reject_legacy_request_fields(values: dict[str, Any]) -> None:
    legacy_fields = {
        "model_path",
        "history",
        "known_covariates",
        "timestamp_col",
        "target_col",
        "item_id_col",
        "postprocess",
    }
    present = sorted(field for field in legacy_fields if field in values)
    if present:
        raise ValueError(f"Legacy request fields are not supported: {present}")


def run_online_inference_from_json(
    path: str | Path,
    model_path: str | Path,
    output_path: str | Path | None = None,
) -> OnlineInferenceResponse:
    """Load a request JSON file and run one online forecast."""

    response = run_online_inference(load_online_inference_request(path), model_path=model_path)
    if output_path is not None:
        save_online_inference_response(response, output_path)
    return response


def save_online_inference_response(
    response: OnlineInferenceResponse,
    path: str | Path,
) -> Path:
    """Save an online inference response as JSON."""

    response_path = Path(path)
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(_response_protocol_dict(response), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return response_path


def run_online_inference(
    request: OnlineInferenceRequest,
    model_path: str | Path,
) -> OnlineInferenceResponse:
    """Load a saved model and run one online prediction request."""

    if not request.history_load:
        raise ValueError("Online inference request requires non-empty history_load records")

    manifest = load_model_manifest(model_path)
    model = load_forecast_model(model_path, manifest.model_type)
    validate_request_schema(request)
    validate_model_compatibility(request, manifest)

    history = _dataset_from_records(request.history_load, item_id=request.item_id, freq=request.task.freq)
    known_covariates = _frame_from_records(request.future_covariates, item_id=request.item_id)

    inference_started = perf_counter()
    forecast = model.predict(
        history,
        prediction_length=request.task.prediction_length,
        freq=request.task.freq,
        known_covariates=known_covariates,
    )
    forecast = normalize_forecast_frame(forecast)
    residual_std = float(getattr(model, "residual_std_", 0.0))
    forecast = ForecastPostProcessor(_postprocess_config_from_manifest(manifest)).transform(
        forecast,
        residual_std=residual_std,
    )
    model_inference_time_ms = (perf_counter() - inference_started) * 1000.0
    response = OnlineInferenceResponse(
        forecast=_forecast_records(forecast),
        prediction_length=request.task.prediction_length,
        freq=request.task.freq,
        model_type=manifest.model_type,
        item_id=request.item_id,
        request_id=request.request_id,
        response_id=request.request_id or str(uuid4()),
        model_version=str(manifest.manifest_version),
        model_name=manifest.model_name,
        scale_name=manifest.scale_name,
        response_time=datetime.now(timezone.utc).isoformat(),
        task_type=request.task.task_type,
        forecast_type=request.task.forecast_type,
        forecast_scale=request.task.forecast_scale or manifest.scale_name,
        model_inference_time_ms=model_inference_time_ms,
    )
    _log_online_response(response)
    return response


def validate_request_schema(request: OnlineInferenceRequest) -> None:
    """Validate request structure and required fields only."""

    if not request.item_id:
        raise ValueError("item_id is required")
    if not request.history_load:
        raise ValueError("Online inference request requires non-empty history_load records")

    history_frame = pd.DataFrame(request.history_load)
    known_frame = pd.DataFrame(request.future_covariates)
    _require_columns(history_frame, ["timestamp", "actual_load"], "history_load")
    if request.future_covariates:
        _require_columns(known_frame, ["timestamp"], "future_covariates")
        if "actual_load" in known_frame.columns:
            raise ValueError("future_covariates must not contain actual_load")


def validate_model_compatibility(
    request: OnlineInferenceRequest,
    manifest: ModelManifest,
) -> None:
    """Validate request against model and manifest expectations."""

    task = request.task
    history_frame = pd.DataFrame(request.history_load)
    known_frame = pd.DataFrame(request.future_covariates)

    errors: list[str] = []
    if task.prediction_length != manifest.prediction_length:
        errors.append(
            f"prediction_length mismatch request={task.prediction_length} manifest={manifest.prediction_length}"
        )
    if task.freq != manifest.freq:
        errors.append(f"freq mismatch request={task.freq} manifest={manifest.freq}")

    errors.extend(_missing_columns(history_frame, manifest.required_history_columns, "history_load"))
    errors.extend(_missing_required_values(history_frame, manifest.required_history_columns, "history_load"))
    errors.extend(_negative_target_errors(history_frame, manifest))
    if len(history_frame) < manifest.required_history_length:
        errors.append(
            f"history_load length={len(history_frame)} shorter than required_history_length={manifest.required_history_length}"
        )
    if manifest.required_known_covariates:
        if known_frame.empty:
            errors.append("future_covariates required by manifest but request provided none")
        else:
            errors.extend(
                _missing_columns(
                    known_frame,
                    manifest.required_known_covariate_columns,
                    "future_covariates",
                )
            )
            errors.extend(
                _missing_required_values(
                    known_frame,
                    manifest.required_known_covariate_columns,
                    "future_covariates",
                )
            )
            if len(known_frame) < task.prediction_length:
                errors.append(
                    f"future_covariates length={len(known_frame)} shorter than prediction_length={task.prediction_length}"
                )
    errors.extend(_validate_timeline(request, history_frame, known_frame))
    if errors:
        for error in errors:
            logger.warning("Online request validation failed: %s", error)
        raise ValueError("Invalid online inference request: " + "; ".join(errors))
    logger.info(
        "Online request validation passed model_type=%s prediction_length=%s required_known_covariates=%s",
        manifest.model_type,
        task.prediction_length,
        manifest.required_known_covariates,
    )


def validate_online_request(
    request: OnlineInferenceRequest,
    manifest: ModelManifest,
) -> None:
    """Backward-compatible validation entrypoint."""

    validate_request_schema(request)
    validate_model_compatibility(request, manifest)


def load_forecast_model(model_path: str | Path, model_type: str = "joblib") -> BaseForecastModel:
    """Load a saved model artifact by type."""

    path = Path(model_path)
    normalized_type = model_type.lower()
    if normalized_type in {"joblib", "sklearn", "lightgbm", "gmm"}:
        model_file = path / "model.joblib" if path.is_dir() else path
        return joblib.load(model_file)
    if normalized_type in {"lstm", "bilstm", "bi_lstm", "torch_lstm"}:
        return LSTMForecaster.load(path, device="cpu")
    if normalized_type in {"mdn", "mixture_density_network"}:
        return MDNForecaster.load(path, device="cpu")
    if normalized_type in {"autogluon", "autogluon_timeseries"}:
        return _load_autogluon_model(path)
    raise ValueError(
        "Unsupported model_type "
        f"'{model_type}'. Use joblib, lightgbm, sklearn, gmm, mdn, lstm, bilstm, or autogluon."
    )


def _require_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def _missing_columns(frame: pd.DataFrame, columns: list[str], frame_name: str) -> list[str]:
    missing = [column for column in columns if column not in frame.columns]
    return [f"{frame_name} missing required columns={missing}"] if missing else []


def _missing_required_values(
    frame: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> list[str]:
    present_columns = [column for column in columns if column in frame.columns]
    if not present_columns:
        return []
    null_counts = frame[present_columns].isna().sum()
    missing_counts = {column: int(count) for column, count in null_counts.items() if int(count) > 0}
    if not missing_counts:
        return []
    return [f"{frame_name} contains null values in required columns={missing_counts}"]


def _negative_target_errors(
    history_frame: pd.DataFrame,
    manifest: ModelManifest,
) -> list[str]:
    contract = manifest.online_request_contract or {}
    if not contract.get("requires_non_negative_target", True):
        return []
    if "actual_load" not in history_frame.columns:
        return []
    target = pd.to_numeric(history_frame["actual_load"], errors="coerce")
    negative_count = int((target < 0).sum())
    if negative_count == 0:
        return []
    return [f"history contains negative target values count={negative_count}"]


def _validate_timeline(
    request: OnlineInferenceRequest,
    history_frame: pd.DataFrame,
    known_frame: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    history = history_frame.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history = history.sort_values(["timestamp"])
    if history.duplicated(["timestamp"]).any():
        errors.append("history_load contains duplicate timestamp rows")
    inferred_history_freq = _infer_single_item_freq(history)
    if inferred_history_freq and not _same_frequency(inferred_history_freq, request.task.freq):
        errors.append(f"history frequency mismatch inferred={inferred_history_freq} request={request.task.freq}")
    errors.extend(_history_continuity_errors(history, request.task.freq))

    if known_frame.empty:
        return errors
    known = known_frame.copy()
    known["timestamp"] = pd.to_datetime(known["timestamp"])
    known = known.sort_values(["timestamp"])
    if known.duplicated(["timestamp"]).any():
        errors.append("future_covariates contains duplicate timestamp rows")

    last_history_timestamp = pd.Timestamp(history["timestamp"].max())
    expected_index = pd.date_range(
        last_history_timestamp,
        periods=request.task.prediction_length + 1,
        freq=request.task.freq,
    )[1:]
    actual_index = pd.to_datetime(known["timestamp"]).head(request.task.prediction_length)
    if len(actual_index) < request.task.prediction_length:
        errors.append(
            f"future_covariates has {len(actual_index)} rows, expected {request.task.prediction_length}"
        )
        return errors
    if not actual_index.reset_index(drop=True).equals(pd.Series(expected_index)):
        errors.append("future_covariates timestamps do not match expected future horizon")
    return errors


def _infer_single_item_freq(frame: pd.DataFrame) -> str | None:
    if len(frame) < 3:
        return None
    return pd.infer_freq(pd.to_datetime(frame["timestamp"]))


def _history_continuity_errors(
    history: pd.DataFrame,
    freq: str,
) -> list[str]:
    errors: list[str] = []
    if len(history) < 2:
        return errors
    expected_index = pd.date_range(
        pd.Timestamp(history["timestamp"].min()),
        pd.Timestamp(history["timestamp"].max()),
        freq=freq,
    )
    actual_index = pd.DatetimeIndex(pd.to_datetime(history["timestamp"]))
    if len(expected_index) != len(actual_index) or not actual_index.equals(expected_index):
        errors.append(f"history timestamps are not continuous at freq={freq}")
    return errors


def _same_frequency(left: str, right: str) -> bool:
    try:
        return to_offset(left) == to_offset(right)
    except ValueError:
        return left == right


def _load_autogluon_model(path: Path) -> AutoGluonTimeSeriesForecaster:
    try:
        from autogluon.timeseries import TimeSeriesPredictor
    except ImportError as exc:
        raise ImportError(
            "AutoGluon inference requires optional dependencies. "
            "Install with: pip install autogluon.timeseries"
        ) from exc

    predictor_path = path if path.is_dir() else path.parent
    predictor = TimeSeriesPredictor.load(str(predictor_path))
    return AutoGluonTimeSeriesForecaster(
        prediction_length=int(predictor.prediction_length),
        freq=str(predictor.freq),
        known_covariates_names=list(getattr(predictor, "known_covariates_names", []) or []),
        eval_metric=str(getattr(predictor, "eval_metric", "MAE")),
        path=str(predictor_path),
        predictor=predictor,
    )


def normalize_forecast_frame(forecast: ForecastFrame) -> ForecastFrame:
    """Normalize model-specific forecast frames to the online response shape."""

    if "horizon_step" in forecast.frame.columns:
        return forecast
    frame = forecast.frame.sort_values([forecast.item_id_col, forecast.timestamp_col]).copy()
    frame["horizon_step"] = frame.groupby(forecast.item_id_col).cumcount() + 1
    return ForecastFrame(
        frame=frame,
        timestamp_col=forecast.timestamp_col,
        item_id_col=forecast.item_id_col,
        prediction_col=forecast.prediction_col,
    )


def _dataset_from_records(
    records: list[dict[str, Any]],
    item_id: str,
    freq: str,
) -> TimeSeriesDataset:
    frame = pd.DataFrame(records).copy()
    frame["item_id"] = item_id
    frame = frame.rename(columns={"actual_load": "target"})
    return TimeSeriesDataset(
        frame=frame,
        timestamp_col="timestamp",
        target_col="target",
        item_id_col="item_id",
        freq=freq,
    )


def _frame_from_records(
    records: list[dict[str, Any]],
    item_id: str | None = None,
) -> pd.DataFrame | None:
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if item_id is not None:
        frame["item_id"] = item_id
    return frame


def _forecast_records(forecast: ForecastFrame) -> list[dict[str, Any]]:
    frame = forecast.frame.copy()
    frame[forecast.timestamp_col] = pd.to_datetime(frame[forecast.timestamp_col])
    return _json_records(frame)


def _response_protocol_dict(response: OnlineInferenceResponse) -> dict[str, Any]:
    results = [_response_result_record(record, response.item_id) for record in response.forecast]
    model_name = response.model_name or response.model_type
    return {
        "code": 0,
        "message": None,
        "request_id": response.request_id,
        "response_id": response.response_id,
        "task_status": response.task_status,
        "item_id": response.item_id,
        "response_time": response.response_time,
        "model": {
            "model_type": response.model_type,
            "model_name": model_name,
            "model_version": response.model_version,
            "scale_name": response.scale_name,
        },
        "task": {
            "task_type": response.task_type,
            "forecast_type": response.forecast_type,
            "forecast_scale": response.forecast_scale,
            "prediction_length": response.prediction_length,
            "freq": response.freq,
        },
        "results": results,
        "summary": {
            "item_count": 1 if response.item_id else 0,
            "forecast_steps": len(results),
            "clipped_negative_count": 0,
            "missing_covariate_count": 0,
            "invalid_row_count": 0,
            "model_inference_time_ms": response.model_inference_time_ms,
        },
        "warnings": response.warnings,
    }


def _response_result_record(record: dict[str, Any], response_item_id: str | None) -> dict[str, Any]:
    result = dict(record)
    if response_item_id is not None:
        result.pop("item_id", None)
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for record in frame.to_dict("records"):
        records.append(
            {
                key: value.isoformat() if isinstance(value, pd.Timestamp) else value
                for key, value in record.items()
            }
        )
    return records


def _postprocess_config_from_manifest(manifest: ModelManifest) -> ForecastPostprocessingConfig:
    payload = dict(manifest.postprocess_config or {})
    distribution_grid_values = payload.get("distribution_grid")
    if isinstance(distribution_grid_values, dict):
        payload["distribution_grid"] = DistributionGridConfig(**distribution_grid_values)
    return ForecastPostprocessingConfig(**payload)


def _log_online_response(response: OnlineInferenceResponse) -> None:
    sample = response.forecast[0] if response.forecast else {}
    logger.info(
        "Online inference response model_type=%s prediction_length=%s freq=%s rows=%s sample=%s model_inference_time_ms=%.3f",
        response.model_type,
        response.prediction_length,
        response.freq,
        len(response.forecast),
        sample,
        response.model_inference_time_ms or 0.0,
    )
