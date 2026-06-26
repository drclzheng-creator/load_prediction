"""Model manifest for offline/online consistency checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any

from load_prediction.configs import PipelineConfig

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "model_manifest.json"


@dataclass(frozen=True)
class ModelManifest:
    model_name: str
    scale_name: str
    model_type: str
    prediction_length: int
    freq: str
    timestamp_col: str
    target_col: str
    item_id_col: str
    required_known_covariates: list[str]
    required_history_columns: list[str]
    required_known_covariate_columns: list[str]
    required_history_length: int
    cleaning_config: dict[str, Any]
    online_request_contract: dict[str, Any]
    feature_config: dict[str, Any]
    postprocess_config: dict[str, Any]
    manifest_version: int = 1


def build_model_manifest(config: PipelineConfig) -> ModelManifest:
    known_covariates = list(config.scale.feature.known_covariates)
    return ModelManifest(
        model_name=config.model.name,
        scale_name=config.scale.name,
        model_type=_model_type(config.model.name),
        prediction_length=config.scale.prediction_length,
        freq=config.scale.freq,
        timestamp_col="timestamp",
        target_col="target",
        item_id_col="item_id",
        required_known_covariates=known_covariates,
        required_history_columns=["timestamp", "actual_load", *known_covariates],
        required_known_covariate_columns=["timestamp", *known_covariates],
        required_history_length=_required_history_length(config),
        cleaning_config=asdict(config.cleaning),
        online_request_contract=_online_request_contract(config),
        feature_config=asdict(config.scale.feature),
        postprocess_config=asdict(config.postprocess),
    )


def save_model_manifest(manifest: ModelManifest, model_dir: str | Path) -> Path:
    output_path = Path(model_dir) / MANIFEST_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved model manifest path=%s", output_path)
    return output_path


def load_model_manifest(model_path: str | Path) -> ModelManifest:
    manifest_path = _manifest_path(model_path)
    if not manifest_path.exists():
        logger.warning("Model manifest not found path=%s", manifest_path)
        raise FileNotFoundError(
            f"Model manifest not found: {manifest_path}. "
            "Please retrain or republish the model artifact with model_manifest.json."
        )
    values = json.loads(manifest_path.read_text(encoding="utf-8"))
    values.setdefault("cleaning_config", {})
    values.setdefault("online_request_contract", _default_online_request_contract(values))
    values = _normalize_loaded_manifest(values)
    return ModelManifest(**values)


def _manifest_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    model_dir = path if path.is_dir() else path.parent
    return model_dir / MANIFEST_FILENAME


def _required_history_length(config: PipelineConfig) -> int:
    feature = config.scale.feature
    lengths = [
        *(int(lag) for lag in feature.lag_steps),
        *(int(window) for window in feature.rolling_windows),
    ]
    if config.model.name in {"lstm", "torch_lstm", "bilstm", "bi_lstm"}:
        context_length = int(config.model.params.get("context_length", 672))
        lag_feature_steps = config.model.params.get("lag_feature_steps")
        if config.model.params.get("add_lag_features", True):
            lag_lengths = (
                [int(lag) for lag in lag_feature_steps]
                if lag_feature_steps is not None
                else [96, 192, 672]
            )
            context_length += max(lag_lengths or [0])
        lengths.append(context_length)
    return max(lengths or [1])


def _model_type(model_name: str) -> str:
    if model_name in {"sklearn_random_forest", "random_forest", "sklearn"}:
        return "sklearn"
    if model_name in {"sklearn_hist_gradient_boosting", "hist_gradient_boosting"}:
        return "sklearn"
    if model_name in {"lightgbm", "lgbm"}:
        return "lightgbm"
    if model_name in {"gmm", "gaussian_mixture", "gaussian_mixture_model"}:
        return "gmm"
    if model_name in {"mdn", "mixture_density_network"}:
        return "mdn"
    if model_name in {"autogluon", "autogluon_timeseries"}:
        return "autogluon"
    if model_name in {"bilstm", "bi_lstm"}:
        return "bilstm"
    if model_name in {"lstm", "torch_lstm"}:
        return "lstm"
    return model_name


def _online_request_contract(config: PipelineConfig) -> dict[str, Any]:
    return {
        "input_cleaning_boundary": "upstream",
        "description": (
            "Online inference requests must provide already-cleaned data. "
            "The inference service validates schema, continuity, nulls, non-negative target, "
            "known covariate coverage, and manifest consistency, but does not run TimeSeriesCleaner."
        ),
        "requires_cleaned_history": True,
        "requires_manifest_freq_alignment": True,
        "requires_no_duplicate_item_timestamps": True,
        "requires_no_missing_required_values": True,
        "requires_non_negative_target": config.cleaning.non_negative_target,
        "requires_full_known_covariate_horizon": True,
        "forbids_future_target_in_known_covariates": True,
    }


def _default_online_request_contract(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_cleaning_boundary": "upstream",
        "description": (
            "Online inference requests must provide already-cleaned data. "
            "This manifest was created before explicit contract fields were added."
        ),
        "requires_cleaned_history": True,
        "requires_manifest_freq_alignment": True,
        "requires_no_duplicate_item_timestamps": True,
        "requires_no_missing_required_values": True,
        "requires_non_negative_target": True,
        "requires_full_known_covariate_horizon": True,
        "forbids_future_target_in_known_covariates": True,
    }


def _normalize_loaded_manifest(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    history_columns = list(normalized.get("required_history_columns") or [])
    covariate_columns = list(normalized.get("required_known_covariate_columns") or [])

    normalized["required_history_columns"] = _normalize_history_columns(history_columns)
    normalized["required_known_covariate_columns"] = _normalize_known_covariate_columns(covariate_columns)
    normalized["required_history_length"] = int(normalized.get("required_history_length") or 1)
    normalized["required_known_covariates"] = [
        column for column in normalized.get("required_known_covariates") or [] if column != "item_id"
    ]
    return normalized


def _normalize_history_columns(columns: list[str]) -> list[str]:
    normalized = []
    for column in columns:
        if column == "target":
            normalized.append("actual_load")
        elif column == "item_id":
            continue
        else:
            normalized.append(column)
    if "timestamp" not in normalized:
        normalized.insert(0, "timestamp")
    if "actual_load" not in normalized:
        normalized.append("actual_load")
    return normalized


def _normalize_known_covariate_columns(columns: list[str]) -> list[str]:
    normalized = [column for column in columns if column != "item_id"]
    if "timestamp" not in normalized:
        normalized.insert(0, "timestamp")
    return normalized
