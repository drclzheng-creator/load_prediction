"""Optuna-based hyperparameter tuning for forecast models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import json
import logging
from pathlib import Path
from typing import Any, Literal

from load_prediction.configs import ModelSpecConfig, PipelineConfig
from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.pipeline import LoadForecastPipeline, PipelineRunResult

logger = logging.getLogger(__name__)

MetricDirection = Literal["minimize", "maximize"]
SearchSpace = Callable[[Any, PipelineConfig], dict[str, Any]]


@dataclass(frozen=True)
class TuningConfig:
    """Optuna tuning settings shared across forecasters."""

    n_trials: int = 20
    metric_name: str = "mae"
    direction: MetricDirection = "minimize"
    study_name: str | None = None
    storage: str | None = None
    load_if_exists: bool = True
    timeout: float | None = None
    n_jobs: int = 1
    sampler_seed: int | None = 42
    save_trial_outputs: bool = False
    tuning_run_name: str = "tuning"
    user_attrs: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class TuningResult:
    """Result bundle returned by Optuna pipeline tuning."""

    study: Any
    best_config: PipelineConfig
    best_params: dict[str, Any]
    best_value: float
    best_trial_number: int


def tune_pipeline(
    base_config: PipelineConfig,
    data: TimeSeriesDataset,
    tuning_config: TuningConfig | None = None,
    search_space: SearchSpace | None = None,
    known_covariates: Any | None = None,
) -> TuningResult:
    """Tune one configured forecast pipeline with Optuna.

    The sampled parameters are merged into ``base_config.model.params`` and the
    existing ``LoadForecastPipeline`` is used as the objective. By default trial
    model/artifact saving is disabled to keep tuning runs lightweight.
    """

    optuna = _import_optuna()
    settings = tuning_config or TuningConfig()
    sampler = None
    if settings.sampler_seed is not None:
        sampler = optuna.samplers.TPESampler(seed=settings.sampler_seed)
    study = optuna.create_study(
        direction=settings.direction,
        study_name=settings.study_name,
        storage=settings.storage,
        load_if_exists=settings.load_if_exists,
        sampler=sampler,
    )
    if settings.user_attrs:
        for key, value in settings.user_attrs.items():
            study.set_user_attr(key, value)

    space = search_space or suggest_model_params

    def objective(trial: Any) -> float:
        sampled_params = space(trial, base_config)
        trial_config = pipeline_config_for_trial(
            base_config,
            sampled_params=sampled_params,
            trial_number=trial.number,
            save_trial_outputs=settings.save_trial_outputs,
            tuning_run_name=settings.tuning_run_name,
        )
        logger.info(
            "Starting Optuna trial number=%s model=%s params=%s",
            trial.number,
            trial_config.model.name,
            sampled_params,
        )
        result = LoadForecastPipeline(trial_config).run(
            data,
            known_covariates=known_covariates,
        )
        value = _metric_value(result, settings.metric_name)
        _set_trial_attrs(trial, result, sampled_params)
        logger.info(
            "Finished Optuna trial number=%s metric=%s value=%s",
            trial.number,
            settings.metric_name,
            value,
        )
        return value

    study.optimize(
        objective,
        n_trials=settings.n_trials,
        timeout=settings.timeout,
        n_jobs=settings.n_jobs,
    )
    best_params = dict(study.best_trial.user_attrs.get("model_params", study.best_params))
    best_config = with_model_params(base_config, best_params)
    _save_tuning_summary(
        base_config=base_config,
        tuning_config=settings,
        study=study,
        best_params=best_params,
    )
    return TuningResult(
        study=study,
        best_config=best_config,
        best_params=best_params,
        best_value=float(study.best_value),
        best_trial_number=int(study.best_trial.number),
    )


def pipeline_config_for_trial(
    base_config: PipelineConfig,
    sampled_params: Mapping[str, Any],
    trial_number: int,
    save_trial_outputs: bool = False,
    tuning_run_name: str = "tuning",
) -> PipelineConfig:
    """Build a trial config by applying sampled model params."""

    config = with_model_params(base_config, sampled_params)
    output = base_config.output
    run_name = _trial_run_name(output.run_name, tuning_run_name, trial_number)
    if save_trial_outputs:
        trial_output = replace(
            output,
            run_name=run_name,
        )
    else:
        trial_output = replace(
            output,
            save_model=False,
            save_artifacts=False,
            save_plot=False,
            save_metrics=False,
            run_name=run_name,
        )
    return replace(config, output=trial_output)


def with_model_params(
    config: PipelineConfig,
    sampled_params: Mapping[str, Any],
) -> PipelineConfig:
    """Return a config with sampled model params merged into the base params."""

    model = config.model
    merged_params = {**model.params, **dict(sampled_params)}
    tuned_model = ModelSpecConfig(
        name=model.name,
        random_state=model.random_state,
        params=merged_params,
        scale_features=model.scale_features,
        scale_target=model.scale_target,
    )
    return replace(config, model=tuned_model)


def _trial_run_name(
    base_run_name: str | None,
    tuning_run_name: str,
    trial_number: int,
) -> str:
    parts = [part for part in (base_run_name, tuning_run_name, f"trial_{trial_number:04d}") if part]
    return "/".join(parts)


def _tuning_output_dir(config: PipelineConfig, tuning_run_name: str) -> Path:
    output = config.output
    base = Path(output.root_dir) / config.model.name / config.scale.name
    if output.run_name:
        base = base / output.run_name
    if tuning_run_name:
        base = base / tuning_run_name
    return base


def _save_tuning_summary(
    base_config: PipelineConfig,
    tuning_config: TuningConfig,
    study: Any,
    best_params: Mapping[str, Any],
) -> Path:
    output_dir = _tuning_output_dir(base_config, tuning_config.tuning_run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "tuning_summary.json"
    summary = {
        "study_name": tuning_config.study_name,
        "model_name": base_config.model.name,
        "forecast_profile": base_config.scale.name,
        "metric_name": tuning_config.metric_name,
        "direction": tuning_config.direction,
        "n_trials": tuning_config.n_trials,
        "completed_trials": len(study.trials),
        "best_trial_number": int(study.best_trial.number),
        "best_value": float(study.best_value),
        "best_params": dict(best_params),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Saved Optuna tuning summary path=%s", summary_path)
    return summary_path


def suggest_model_params(trial: Any, config: PipelineConfig) -> dict[str, Any]:
    """Default Optuna search space for every supported forecaster family."""

    name = config.model.name.lower()
    if name in {"lightgbm", "lgbm"}:
        return _suggest_lightgbm_params(trial, config.model.params)
    if name in {"sklearn_hist_gradient_boosting", "hist_gradient_boosting"}:
        return _suggest_sklearn_hgb_params(trial, config.model.params)
    if name in {"sklearn_random_forest", "random_forest", "sklearn"}:
        return _suggest_random_forest_params(trial, config.model.params)
    if name in {"gmm", "gaussian_mixture", "gaussian_mixture_model"}:
        return _suggest_gmm_params(trial, config.model.params)
    if name in {"mdn", "mixture_density_network"}:
        return _suggest_mdn_params(trial, config.model.params)
    if name in {"lstm", "torch_lstm", "bilstm", "bi_lstm"}:
        return _suggest_lstm_params(trial, config.model.params)
    if name in {"autogluon", "autogluon_timeseries"}:
        return _suggest_autogluon_params(trial, config.model.params)
    raise ValueError(f"No default Optuna search space for model '{config.model.name}'")


def _suggest_lightgbm_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    params = _preserve_quantile_settings(base_params)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 80, 600, step=40),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": int(base_params.get("n_jobs", 1)),
            "verbosity": int(base_params.get("verbosity", -1)),
            "objective": str(base_params.get("objective", "regression_l1")),
        }
    )
    return params


def _suggest_sklearn_hgb_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    params = _preserve_quantile_settings(base_params)
    params.update(
        {
            "max_iter": trial.suggest_int("max_iter", 40, 240, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 63),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 80),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            "loss": str(base_params.get("loss", "absolute_error")),
        }
    )
    return params


def _suggest_random_forest_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 80, 500, step=40),
        "max_depth": trial.suggest_int("max_depth", 4, 32),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 12),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 1.0]),
        "criterion": str(base_params.get("criterion", "absolute_error")),
        "n_jobs": int(base_params.get("n_jobs", -1)),
    }


def _suggest_gmm_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    point_estimator = str(base_params.get("point_estimator", "lightgbm"))
    params: dict[str, Any] = {
        "n_components": trial.suggest_int("n_components", 2, 6),
        "reg_covar": trial.suggest_float("reg_covar", 1e-8, 1e-3, log=True),
        "gmm_max_iter": trial.suggest_int("gmm_max_iter", 80, 240, step=20),
        "n_init": trial.suggest_int("n_init", 1, 5),
        "conditional_distribution": trial.suggest_categorical(
            "conditional_distribution",
            [True, False],
        ),
        "point_estimator": point_estimator,
        "distribution_grid_size": int(base_params.get("distribution_grid_size", 80)),
        "distribution_std_width": float(base_params.get("distribution_std_width", 4.0)),
    }
    if point_estimator in {"lightgbm", "lgbm"}:
        params["point_params"] = _suggest_lightgbm_point_params(trial, base_params)
    elif point_estimator in {"hist_gradient_boosting", "hgb", "sklearn_hgb"}:
        params["point_params"] = _suggest_hgb_point_params(trial, base_params)
    return params


def _suggest_mdn_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_components": trial.suggest_int("n_components", 2, 5),
        "hidden_size": trial.suggest_categorical("hidden_size", [16, 32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "epochs": int(base_params.get("epochs", 20)),
        "max_train_samples": base_params.get("max_train_samples", 4096),
        "train_sample_strategy": str(base_params.get("train_sample_strategy", "random")),
        "validation_fraction": float(base_params.get("validation_fraction", 0.1)),
        "early_stopping_patience": int(base_params.get("early_stopping_patience", 5)),
        "device": str(base_params.get("device", "cpu")),
        "distribution_grid_size": int(base_params.get("distribution_grid_size", 0)),
        "distribution_std_width": float(base_params.get("distribution_std_width", 4.0)),
    }


def _suggest_lstm_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context_length": trial.suggest_categorical(
            "context_length",
            list(base_params.get("context_length_choices", [192, 336, 672])),
        ),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "n_components": trial.suggest_int("n_components", 2, 5),
        "use_point_head": bool(base_params.get("use_point_head", False)),
        "point_loss_weight": float(base_params.get("point_loss_weight", 0.0)),
        "epochs": int(base_params.get("epochs", 20)),
        "max_train_samples": base_params.get("max_train_samples", 4096),
        "train_sample_strategy": str(base_params.get("train_sample_strategy", "last")),
        "validation_fraction": float(base_params.get("validation_fraction", 0.1)),
        "early_stopping_patience": int(base_params.get("early_stopping_patience", 5)),
        "device": str(base_params.get("device", "cpu")),
        "distribution_grid_size": int(base_params.get("distribution_grid_size", 80)),
        "distribution_std_width": float(base_params.get("distribution_std_width", 4.0)),
    }


def _suggest_autogluon_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(base_params)
    params["presets"] = trial.suggest_categorical(
        "presets",
        list(base_params.get("preset_choices", ["fast_training", "medium_quality"])),
    )
    params["time_limit"] = int(base_params.get("time_limit", 600))
    return params


def _suggest_lightgbm_point_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    point_params = dict(base_params.get("point_params", {}))
    return {
        "n_estimators": trial.suggest_int("point_n_estimators", 80, 600, step=40),
        "learning_rate": trial.suggest_float("point_learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("point_num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("point_min_child_samples", 5, 80),
        "objective": str(point_params.get("objective", "regression_l1")),
        "n_jobs": int(point_params.get("n_jobs", 1)),
        "verbosity": int(point_params.get("verbosity", -1)),
    }


def _suggest_hgb_point_params(trial: Any, base_params: Mapping[str, Any]) -> dict[str, Any]:
    point_params = dict(base_params.get("point_params", {}))
    return {
        "max_iter": trial.suggest_int("point_max_iter", 40, 240, step=20),
        "learning_rate": trial.suggest_float("point_learning_rate", 0.01, 0.2, log=True),
        "max_leaf_nodes": trial.suggest_int("point_max_leaf_nodes", 15, 63),
        "min_samples_leaf": trial.suggest_int("point_min_samples_leaf", 10, 80),
        "loss": str(point_params.get("loss", "absolute_error")),
    }


def _preserve_quantile_settings(base_params: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in ("enable_quantiles", "lower_quantile", "upper_quantile", "quantile_levels"):
        if key in base_params:
            params[key] = base_params[key]
    return params


def _metric_value(result: PipelineRunResult, metric_name: str) -> float:
    if metric_name not in result.metrics:
        available = ", ".join(sorted(result.metrics))
        raise KeyError(f"Metric '{metric_name}' not found. Available metrics: {available}")
    return float(result.metrics[metric_name])


def _set_trial_attrs(
    trial: Any,
    result: PipelineRunResult,
    sampled_params: Mapping[str, Any],
) -> None:
    trial.set_user_attr("metrics", result.metrics)
    trial.set_user_attr("output_dir", str(result.output_dir))
    trial.set_user_attr("model_params", dict(sampled_params))


def _import_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "Optuna tuning requires the optional dependency: pip install optuna"
        ) from exc
    return optuna
