"""Online inference helpers for saved forecasting models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineInferenceRequest:
    """Serializable request shape for one online forecast call."""

    model_path: str
    model_type: str = "joblib"
    history: list[dict[str, Any]] = field(default_factory=list)
    known_covariates: list[dict[str, Any]] = field(default_factory=list)
    prediction_length: int = 96
    freq: str = "15min"
    timestamp_col: str = "timestamp"
    target_col: str = "target"
    item_id_col: str = "item_id"
    postprocess: ForecastPostprocessingConfig = field(default_factory=ForecastPostprocessingConfig)


@dataclass(frozen=True)
class OnlineInferenceResponse:
    """Serializable response for online forecast calls."""

    forecast: list[dict[str, Any]]
    prediction_length: int
    freq: str
    model_type: str


def load_online_inference_request(path: str | Path) -> OnlineInferenceRequest:
    """Load a serialized online inference request from JSON."""

    request_path = Path(path)
    values = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Online inference request JSON must be an object")
    return online_inference_request_from_dict(values)


def online_inference_request_from_dict(values: dict[str, Any]) -> OnlineInferenceRequest:
    """Build an online inference request from a JSON-compatible dictionary."""

    payload = dict(values)
    postprocess_values = payload.get("postprocess")
    if postprocess_values is None:
        postprocess = ForecastPostprocessingConfig()
    elif isinstance(postprocess_values, ForecastPostprocessingConfig):
        postprocess = postprocess_values
    elif isinstance(postprocess_values, dict):
        postprocess_payload = dict(postprocess_values)
        distribution_grid_values = postprocess_payload.get("distribution_grid")
        if isinstance(distribution_grid_values, dict):
            postprocess_payload["distribution_grid"] = DistributionGridConfig(
                **distribution_grid_values
            )
        postprocess = ForecastPostprocessingConfig(**postprocess_payload)
    else:
        raise ValueError("postprocess must be an object when provided")
    payload["postprocess"] = postprocess
    return OnlineInferenceRequest(**payload)


def run_online_inference_from_json(
    path: str | Path,
    output_path: str | Path | None = None,
) -> OnlineInferenceResponse:
    """Load a prediction_request.json file and run one online forecast."""

    response = run_online_inference(load_online_inference_request(path))
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
        json.dumps(asdict(response), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return response_path


def run_online_inference(request: OnlineInferenceRequest) -> OnlineInferenceResponse:
    """Load a saved model and run one online prediction request."""

    if not request.history:
        raise ValueError("Online inference request requires non-empty history records")
    model = load_forecast_model(request.model_path, request.model_type)
    manifest = load_model_manifest(request.model_path)
    validate_online_request(request, manifest)
    history = _dataset_from_records(
        request.history,
        timestamp_col=request.timestamp_col,
        target_col=request.target_col,
        item_id_col=request.item_id_col,
        freq=request.freq,
    )
    known_covariates = _frame_from_records(request.known_covariates, request.timestamp_col)
    forecast = model.predict(
        history,
        prediction_length=request.prediction_length,
        freq=request.freq,
        known_covariates=known_covariates,
    )
    forecast = normalize_forecast_frame(forecast)
    residual_std = float(getattr(model, "residual_std_", 0.0))
    forecast = ForecastPostProcessor(request.postprocess).transform(
        forecast,
        residual_std=residual_std,
    )
    response = OnlineInferenceResponse(
        forecast=_forecast_records(forecast),
        prediction_length=request.prediction_length,
        freq=request.freq,
        model_type=request.model_type,
    )
    _log_online_response(response)
    return response


def validate_online_request(
    request: OnlineInferenceRequest,
    manifest: ModelManifest,
) -> None:
    """Validate online request shape before feature construction and prediction."""

    if not request.history:
        raise ValueError("Online inference request requires non-empty history records")
    history_frame = pd.DataFrame(request.history)
    known_frame = pd.DataFrame(request.known_covariates)
    _require_columns(
        history_frame,
        [request.timestamp_col, request.item_id_col, request.target_col],
        "history",
    )
    if request.known_covariates:
        _require_columns(
            known_frame,
            [request.timestamp_col, request.item_id_col],
            "known_covariates",
        )
        if request.target_col in known_frame.columns:
            logger.warning("known_covariates must not contain future target column")
            raise ValueError("known_covariates must not contain future target column")

    errors: list[str] = []
    if request.prediction_length != manifest.prediction_length:
        errors.append(
            f"prediction_length mismatch request={request.prediction_length} manifest={manifest.prediction_length}"
        )
    if request.freq != manifest.freq:
        errors.append(f"freq mismatch request={request.freq} manifest={manifest.freq}")
    if request.timestamp_col != manifest.timestamp_col:
        errors.append(
            f"timestamp_col mismatch request={request.timestamp_col} manifest={manifest.timestamp_col}"
        )
    if request.target_col != manifest.target_col:
        errors.append(f"target_col mismatch request={request.target_col} manifest={manifest.target_col}")
    if request.item_id_col != manifest.item_id_col:
        errors.append(f"item_id_col mismatch request={request.item_id_col} manifest={manifest.item_id_col}")

    errors.extend(_missing_columns(history_frame, manifest.required_history_columns, "history"))
    errors.extend(_missing_required_values(history_frame, manifest.required_history_columns, "history"))
    errors.extend(_negative_target_errors(history_frame, request, manifest))
    if len(history_frame) < manifest.required_history_length:
        errors.append(
            f"history length={len(history_frame)} shorter than required_history_length={manifest.required_history_length}"
        )
    if manifest.required_known_covariates:
        if known_frame.empty:
            errors.append("known_covariates required by manifest but request provided none")
        else:
            errors.extend(
                _missing_columns(
                    known_frame,
                    manifest.required_known_covariate_columns,
                    "known_covariates",
                )
            )
            errors.extend(
                _missing_required_values(
                    known_frame,
                    manifest.required_known_covariate_columns,
                    "known_covariates",
                )
            )
            if len(known_frame) < request.prediction_length:
                errors.append(
                    f"known_covariates length={len(known_frame)} shorter than prediction_length={request.prediction_length}"
                )

    errors.extend(_validate_timeline(request, history_frame, known_frame))
    if errors:
        for error in errors:
            logger.warning("Online request validation failed: %s", error)
        raise ValueError("Invalid online inference request: " + "; ".join(errors))
    logger.info(
        "Online request validation passed model_type=%s prediction_length=%s required_known_covariates=%s",
        request.model_type,
        request.prediction_length,
        manifest.required_known_covariates,
    )


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
        logger.warning("%s missing required columns=%s", frame_name, missing)
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
    missing_counts = {
        column: int(count)
        for column, count in null_counts.items()
        if int(count) > 0
    }
    if not missing_counts:
        return []
    return [f"{frame_name} contains null values in required columns={missing_counts}"]


def _negative_target_errors(
    history_frame: pd.DataFrame,
    request: OnlineInferenceRequest,
    manifest: ModelManifest,
) -> list[str]:
    contract = manifest.online_request_contract or {}
    if not contract.get("requires_non_negative_target", True):
        return []
    if request.target_col not in history_frame.columns:
        return []
    target = pd.to_numeric(history_frame[request.target_col], errors="coerce")
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
    history[request.timestamp_col] = pd.to_datetime(history[request.timestamp_col])
    history = history.sort_values([request.item_id_col, request.timestamp_col])
    if history.duplicated([request.item_id_col, request.timestamp_col]).any():
        errors.append("history contains duplicate item_id/timestamp rows")
    inferred_history_freq = _infer_single_item_freq(history, request)
    if inferred_history_freq and not _same_frequency(inferred_history_freq, request.freq):
        errors.append(f"history frequency mismatch inferred={inferred_history_freq} request={request.freq}")
    errors.extend(_history_continuity_errors(history, request))

    if known_frame.empty:
        return errors
    known = known_frame.copy()
    known[request.timestamp_col] = pd.to_datetime(known[request.timestamp_col])
    known = known.sort_values([request.item_id_col, request.timestamp_col])
    if known.duplicated([request.item_id_col, request.timestamp_col]).any():
        errors.append("known_covariates contains duplicate item_id/timestamp rows")

    for item_id, item_history in history.groupby(request.item_id_col, sort=False):
        item_known = known[known[request.item_id_col] == item_id]
        if item_known.empty:
            errors.append(f"known_covariates missing rows for item_id={item_id}")
            continue
        last_history_timestamp = pd.Timestamp(item_history[request.timestamp_col].max())
        expected_index = pd.date_range(
            last_history_timestamp,
            periods=request.prediction_length + 1,
            freq=request.freq,
        )[1:]
        actual_index = pd.to_datetime(item_known[request.timestamp_col]).head(
            request.prediction_length
        )
        if len(actual_index) < request.prediction_length:
            errors.append(
                f"known_covariates item_id={item_id} has {len(actual_index)} rows, expected {request.prediction_length}"
            )
            continue
        if not actual_index.reset_index(drop=True).equals(pd.Series(expected_index)):
            errors.append(
                f"known_covariates timestamps for item_id={item_id} do not match expected future horizon"
            )
    return errors


def _infer_single_item_freq(
    frame: pd.DataFrame,
    request: OnlineInferenceRequest,
) -> str | None:
    frequencies = []
    for _, group in frame.groupby(request.item_id_col, sort=False):
        if len(group) < 3:
            continue
        inferred = pd.infer_freq(pd.to_datetime(group[request.timestamp_col]))
        if inferred:
            frequencies.append(inferred)
    unique = set(frequencies)
    return unique.pop() if len(unique) == 1 else None


def _history_continuity_errors(
    history: pd.DataFrame,
    request: OnlineInferenceRequest,
) -> list[str]:
    errors: list[str] = []
    for item_id, group in history.groupby(request.item_id_col, sort=False):
        if len(group) < 2:
            continue
        expected_index = pd.date_range(
            pd.Timestamp(group[request.timestamp_col].min()),
            pd.Timestamp(group[request.timestamp_col].max()),
            freq=request.freq,
        )
        actual_index = pd.DatetimeIndex(pd.to_datetime(group[request.timestamp_col]))
        if len(expected_index) != len(actual_index) or not actual_index.equals(expected_index):
            errors.append(f"history timestamps for item_id={item_id} are not continuous at freq={request.freq}")
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
    timestamp_col: str,
    target_col: str,
    item_id_col: str,
    freq: str,
) -> TimeSeriesDataset:
    frame = pd.DataFrame(records)
    return TimeSeriesDataset(
        frame=frame,
        timestamp_col=timestamp_col,
        target_col=target_col,
        item_id_col=item_id_col,
        freq=freq,
    )


def _frame_from_records(
    records: list[dict[str, Any]],
    timestamp_col: str,
) -> pd.DataFrame | None:
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    return frame


def _forecast_records(forecast: ForecastFrame) -> list[dict[str, Any]]:
    frame = forecast.frame.copy()
    frame[forecast.timestamp_col] = pd.to_datetime(frame[forecast.timestamp_col])
    return _json_records(frame)


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


def _log_online_response(response: OnlineInferenceResponse) -> None:
    sample = response.forecast[0] if response.forecast else {}
    logger.info(
        "Online inference response model_type=%s prediction_length=%s freq=%s rows=%s sample=%s",
        response.model_type,
        response.prediction_length,
        response.freq,
        len(response.forecast),
        sample,
    )
