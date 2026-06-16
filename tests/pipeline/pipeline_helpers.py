from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from load_prediction.configs import (
    ModelSpecConfig,
    ArtifactConfig,
    PipelineConfig,
)
from load_prediction.data import CSVLoadDataLoader
from load_prediction.models.base_forecaster import ForecastFrame
from load_prediction.models.gmm_forecaster import GMMForecaster
from load_prediction.models.lstm_forecaster import LSTMForecaster
from load_prediction.models.mdn_forecaster import MDNForecaster
from load_prediction.models.parameter_tuning import TuningConfig, tune_pipeline
from load_prediction.pipeline import LoadForecastPipeline

DEFAULT_CONFIG = Path("configs/day_ahead.yaml")
DEFAULT_SHORT_TERM_4H_CONFIG = Path("configs/short_term_4h.yaml")
DEFAULT_TEST_CSV = Path("inputs/EWELD/eweld_industrial_u141_2018_2019_15min.csv")
DEFAULT_TEST_START_TIME = "2019-01-01 00:00:00"
DEFAULT_TEST_END_TIME = "2019-12-31 23:45:00"
SELECTED_FEATURE_COLUMNS = (
    "temperature_f",
    "dew_point_f",
    "humidity_pct",
    "wind_direction",
    "wind_speed_mph",
    "wind_gust_mph",
    "pressure_in",
)


def _test_csv_path() -> Path:
    return Path(os.environ.get("LOAD_PREDICTION_TEST_CSV", DEFAULT_TEST_CSV))


def _day_ahead_config() -> PipelineConfig:
    config_path = Path(os.environ.get("LOAD_PREDICTION_TEST_CONFIG", DEFAULT_CONFIG))
    config = PipelineConfig.from_yaml(config_path)
    csv_path = _test_csv_path()
    if str(csv_path) != str(DEFAULT_TEST_CSV):
        config = PipelineConfig(
            data=config.data.__class__(
                **{**config.data.__dict__, "path": str(csv_path)}
            ),
            cleaning=config.cleaning,
            model=config.model,
            evaluation=config.evaluation,
            postprocess=config.postprocess,
            output=config.output,
            scale=config.scale,
        )
    return config


def _short_term_4h_config() -> PipelineConfig:
    config_path = Path(os.environ.get("LOAD_PREDICTION_SHORT_TERM_4H_CONFIG", DEFAULT_SHORT_TERM_4H_CONFIG))
    config = PipelineConfig.from_yaml(config_path)
    csv_path = _test_csv_path()
    if str(csv_path) != str(DEFAULT_TEST_CSV):
        config = PipelineConfig(
            data=config.data.__class__(
                **{**config.data.__dict__, "path": str(csv_path)}
            ),
            cleaning=config.cleaning,
            model=config.model,
            evaluation=config.evaluation,
            postprocess=config.postprocess,
            output=config.output,
            scale=config.scale,
        )
    return config


def _load_data(config: PipelineConfig):
    csv_path = Path(config.data.path or _test_csv_path())
    if not csv_path.exists():
        raise FileNotFoundError(f"Test CSV does not exist: {csv_path}")
    return CSVLoadDataLoader(config.data).load(csv_path)


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


def _autogluon_model_config() -> ModelSpecConfig:
    return ModelSpecConfig(
        name="autogluon",
        random_state=42,
        params={
            "eval_metric": "MAE",
            "presets": "medium_quality",
            "time_limit": 600,
            "enable_ensemble": False,
            "quantile_levels": [
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                0.95,
            ],
            "hyperparameters": {
                "RecursiveTabular": {
                    "model_name": "GBM",
                },
            },
        },
    )


def _with_time_range(config: PipelineConfig) -> PipelineConfig:
    return PipelineConfig(
        data=config.data.__class__(
            **{
                **config.data.__dict__,
                "start_time": DEFAULT_TEST_START_TIME,
                "end_time": DEFAULT_TEST_END_TIME,
            }
        ),
        cleaning=config.cleaning,
        model=config.model,
        evaluation=config.evaluation,
        postprocess=config.postprocess,
        output=config.output,
        scale=config.scale,
    )


def _assert_day_ahead_run_outputs(
    result,
    model_name: str,
    run_name: str | None = None,
) -> None:
    _assert_scale_run_outputs(result, model_name, "day_ahead", run_name=run_name)


def _assert_scale_run_outputs(
    result,
    model_name: str,
    scale_name: str,
    run_name: str | None = None,
) -> None:
    expected_output_dir = Path("outputs") / model_name / scale_name
    if run_name:
        expected_output_dir = expected_output_dir / run_name
    assert result.output_dir == expected_output_dir
    assert result.model_path is not None
    assert result.model_path.exists()
    assert result.forecast_csv_path is not None
    assert result.forecast_csv_path.exists()
    assert result.forecast_csv_path.stat().st_size > 0
    assert result.forecast_plot_path is not None
    assert result.forecast_plot_path.exists()
    assert result.forecast_plot_path.stat().st_size > 0
    assert result.metrics_path is not None
    assert result.metrics_path.exists()
    train_log = result.output_dir / "logs" / "train.log"
    assert train_log.exists()
    assert train_log.stat().st_size > 0


def _assert_day_ahead_rolling_windows(result, prediction_length: int) -> None:
    assert "evaluation_window" in result.forecast.frame.columns
    window_sizes = result.forecast.frame.groupby("evaluation_window").size().tolist()
    assert len(result.test_data.frame) % prediction_length == 0
    assert sum(window_sizes) == len(result.test_data.frame)
    assert all(size == prediction_length for size in window_sizes[:-1])
    assert window_sizes[-1] == prediction_length


def _assert_basic_forecast_result(result, prediction_length: int) -> None:
    assert len(result.forecast.frame) == len(result.test_data.frame)
    _assert_day_ahead_rolling_windows(result, prediction_length)
    assert result.metrics is not None
    assert result.forecast.frame["prediction"].notna().all()


def _assert_quantile_forecast_columns(frame: pd.DataFrame) -> None:
    assert {
        "0.05",
        "0.1",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
        "0.6",
        "0.7",
        "0.8",
        "0.9",
        "0.95",
        "prediction_lower",
        "prediction_upper",
    }.issubset(frame.columns)


def _assert_distribution_grid(frame: pd.DataFrame) -> None:
    assert {"distribution_values", "distribution_probabilities"}.issubset(frame.columns)
    first_values = json.loads(frame["distribution_values"].iloc[0])
    first_probabilities = json.loads(frame["distribution_probabilities"].iloc[0])
    assert len(first_values) == len(first_probabilities)


def _plot_gmm_curve_distribution(
    forecast: ForecastFrame,
    output_path: Path,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("GMM distribution plotting requires matplotlib") from exc

    frame = forecast.frame.sort_values(forecast.timestamp_col).copy()
    required = {"prediction", "prediction_lower", "prediction_upper", "prediction_density"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"GMM forecast missing distribution columns: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (curve_ax, density_ax) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    curve_ax.plot(
        frame[forecast.timestamp_col],
        frame["prediction"],
        color="#d62728",
        linewidth=1.8,
        label="Point forecast",
    )
    curve_ax.fill_between(
        frame[forecast.timestamp_col],
        frame["prediction_lower"],
        frame["prediction_upper"],
        color="#d62728",
        alpha=0.18,
        label="GMM P10-P90 distribution band",
    )
    curve_ax.set_title("GMM Forecast Curve Probability Distribution")
    curve_ax.set_ylabel("Load")
    curve_ax.grid(True, alpha=0.25)
    curve_ax.legend(loc="best")

    density_ax.plot(
        frame[forecast.timestamp_col],
        frame["prediction_density"],
        color="#2ca02c",
        linewidth=1.4,
        label="Density at point forecast",
    )
    density_ax.fill_between(
        frame[forecast.timestamp_col],
        frame["prediction_density"],
        color="#2ca02c",
        alpha=0.15,
    )
    density_ax.set_ylabel("Density")
    density_ax.set_xlabel("Timestamp")
    density_ax.grid(True, alpha=0.25)
    density_ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_gmm_sample_distributions(
    forecast: ForecastFrame,
    output_dir: Path,
    sample_size: int = 5,
    random_state: int = 42,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("GMM distribution plotting requires matplotlib") from exc

    frame = forecast.frame.sort_values(forecast.timestamp_col).copy()
    if len(frame) < sample_size:
        raise AssertionError(f"Need at least {sample_size} forecast rows for distribution samples")
    required = {"distribution_values", "distribution_probabilities", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise AssertionError(f"GMM forecast missing sampled distribution columns: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sampled = frame.sample(n=sample_size, random_state=random_state).sort_values(
        forecast.timestamp_col
    )
    paths: list[Path] = []
    for index, (_, row) in enumerate(sampled.iterrows(), 1):
        values = pd.Series(json.loads(row["distribution_values"]), dtype=float)
        probabilities = pd.Series(json.loads(row["distribution_probabilities"]), dtype=float)
        timestamp = pd.Timestamp(row[forecast.timestamp_col])
        output_path = output_dir / f"gmm_point_distribution_{index}.png"

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(values, probabilities, color="#1f77b4", linewidth=1.8)
        ax.fill_between(values, probabilities, color="#1f77b4", alpha=0.16)
        ax.axvline(float(row["prediction"]), color="#d62728", linewidth=1.5, label="Prediction")
        ax.set_title(f"GMM Point Distribution - {timestamp}")
        ax.set_xlabel("Load")
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        paths.append(output_path)
    return paths


def run_lightgbm_pipeline():
    config = _with_time_range(_day_ahead_config())
    data = _load_data(config)

    assert data.freq == "15min"
    assert data.frame[data.timestamp_col].min() == pd.Timestamp(DEFAULT_TEST_START_TIME)
    assert data.frame[data.timestamp_col].max() == pd.Timestamp(DEFAULT_TEST_END_TIME)
    assert set(data.covariate_columns) == set(SELECTED_FEATURE_COLUMNS)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "lightgbm")
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    _assert_quantile_forecast_columns(result.forecast.frame)
    _assert_distribution_grid(result.forecast.frame)
    assert (result.forecast.frame["prediction"] >= 0).all()


def run_lightgbm_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    config = _with_time_range(_day_ahead_config())
    _run_pipeline_with_parameter_tuning(
        config,
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def _run_pipeline_with_parameter_tuning(
    config: PipelineConfig,
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    if not enable_parameter_tuning:
        import pytest

        pytest.skip(
            "Pass enable_parameter_tuning=True to run real Optuna tuning"
        )

    data = _load_data(config)

    tuning = TuningConfig(
        n_trials=n_trials,
        metric_name="mae",
        direction="minimize",
        study_name=f"{config.model.name}_{config.scale.name}",
        save_trial_outputs=save_trial_outputs,
        tuning_run_name="tuning",
    )
    result = tune_pipeline(config, data, tuning_config=tuning)

    assert result.best_trial_number >= 0
    assert result.best_value > 0
    assert result.best_params
    assert result.best_config.model.name == config.model.name

    tuning_dir = Path(config.output.root_dir) / config.model.name / config.scale.name
    if config.output.run_name:
        tuning_dir = tuning_dir / config.output.run_name
    tuning_dir = tuning_dir / "tuning"
    summary_path = tuning_dir / "tuning_summary.json"
    assert tuning_dir.exists()
    assert any(tuning_dir.glob("trial_*/artifacts/metrics.json"))
    assert summary_path.exists()

    tuning_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert tuning_summary["best_trial_number"] == result.best_trial_number
    assert tuning_summary["best_value"] == result.best_value
    assert tuning_summary["best_params"] == result.best_params


def run_short_term_4h_lightgbm_pipeline():
    config = _with_time_range(_short_term_4h_config())
    data = _load_data(config)

    assert config.scale.name == "short_term_4h"
    assert config.scale.freq == "15min"
    assert config.scale.prediction_length == 16
    assert data.freq == "15min"

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_scale_run_outputs(result, "lightgbm", "short_term_4h")
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    assert result.forecast.frame["horizon_step"].min() == 1
    assert result.forecast.frame["horizon_step"].max() == 16
    _assert_quantile_forecast_columns(result.forecast.frame)
    _assert_distribution_grid(result.forecast.frame)


def run_short_term_4h_lightgbm_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _with_time_range(_short_term_4h_config()),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )



def run_sklearn_pipeline():
    config = _sklearn_config()
    data = _load_data(config)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "sklearn_hist_gradient_boosting")
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    _assert_quantile_forecast_columns(result.forecast.frame)
    _assert_distribution_grid(result.forecast.frame)


def run_sklearn_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _sklearn_config(),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def _sklearn_config() -> PipelineConfig:
    base_config = _with_time_range(_day_ahead_config())
    return _with_model(
        base_config,
        ModelSpecConfig(
            name="sklearn_hist_gradient_boosting",
            random_state=42,
            params={
                "max_iter": 80,
                "enable_quantiles": True,
                "lower_quantile": 0.1,
                "upper_quantile": 0.9,
                "quantile_levels": [
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                ],
                "loss": "absolute_error",
            },
        ),
    )


def run_mdn_pipeline():
    config = _mdn_config()
    data = _load_data(config)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "mdn", run_name="e2e_mdn")
    assert isinstance(pipeline.model, MDNForecaster)
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    expected_columns = {
        "prediction",
        "0.1",
        "0.9",
        "prediction_lower",
        "prediction_upper",
        "prediction_density",
        "mdn_weights",
        "mdn_means",
        "mdn_stds",
        "distribution_values",
        "distribution_probabilities",
    }
    assert expected_columns.issubset(result.forecast.frame.columns)
    _assert_distribution_grid(result.forecast.frame)
    loaded = MDNForecaster.load(result.model_path, device="cpu")
    assert loaded.fitted_ is True


def run_mdn_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _mdn_config(),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def _mdn_config() -> PipelineConfig:
    base_config = _with_time_range(_day_ahead_config())
    return PipelineConfig(
        data=base_config.data,
        cleaning=base_config.cleaning,
        model=ModelSpecConfig(
            name="mdn",
            random_state=42,
            params={
                "n_components": 2,
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "epochs": 35,
                "batch_size": 128,
                "learning_rate": 0.001,
                "nll_loss_weight": 1.0,
                "mae_loss_weight": 0.0,
                "max_train_samples": 16384,
                "train_sample_strategy": "last",
                "validation_fraction": 0.1,
                "early_stopping_patience": 8,
                "distribution_grid_size": 25,
                "device": "cpu",
            },
        ),
        evaluation=base_config.evaluation,
        postprocess=base_config.postprocess,
        output=ArtifactConfig(
            root_dir=base_config.output.root_dir,
            run_name="e2e_mdn",
            save_model=True,
            save_artifacts=True,
            save_plot=True,
            save_metrics=True,
        ),
        scale=base_config.scale,
    )


def run_gmm_pipeline_with_distribution_plots():
    config = _gmm_config()
    data = _load_data(config)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "gmm", run_name="e2e_probability_plots")
    assert isinstance(pipeline.model, GMMForecaster)
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    expected_columns = {
        "prediction",
        "0.1",
        "0.9",
        "prediction_lower",
        "prediction_upper",
        "prediction_density",
        "gmm_weights",
        "gmm_means",
        "gmm_stds",
        "distribution_values",
        "distribution_probabilities",
    }
    assert expected_columns.issubset(result.forecast.frame.columns)

    artifacts_dir = result.output_dir / "artifacts"
    curve_distribution_path = _plot_gmm_curve_distribution(
        result.forecast,
        artifacts_dir / "gmm_curve_probability_distribution.png",
    )
    sample_distribution_paths = _plot_gmm_sample_distributions(
        result.forecast,
        artifacts_dir / "gmm_point_distributions",
        sample_size=5,
        random_state=42,
    )

    assert curve_distribution_path.exists()
    assert curve_distribution_path.stat().st_size > 0
    assert len(sample_distribution_paths) == 5


def run_gmm_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _gmm_config(),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def _gmm_config() -> PipelineConfig:
    base_config = _with_time_range(_day_ahead_config())
    return PipelineConfig(
        data=base_config.data,
        cleaning=base_config.cleaning,
        model=ModelSpecConfig(
            name="gmm",
            random_state=42,
            params={
                "n_components": 3,
                "point_estimator": "lightgbm",
                "point_params": {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "objective": "regression_l1",
                    "num_leaves": 31,
                    "n_jobs": 1,
                    "verbosity": -1,
                },
                "distribution_grid_size": 80,
                "distribution_std_width": 4.0,
                "gmm_max_iter": 120,
                "n_init": 2,
            },
        ),
        evaluation=base_config.evaluation,
        postprocess=base_config.postprocess,
        output=ArtifactConfig(
            root_dir=base_config.output.root_dir,
            run_name="e2e_probability_plots",
            save_model=True,
            save_artifacts=True,
            save_plot=True,
            save_metrics=True,
        ),
        scale=base_config.scale,
    )


def run_autogluon_pipeline():
    config = _with_model(_with_time_range(_day_ahead_config()), _autogluon_model_config())
    data = _load_data(config)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "autogluon")
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    _assert_quantile_forecast_columns(result.forecast.frame)
    _assert_distribution_grid(result.forecast.frame)


def run_autogluon_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _with_model(_with_time_range(_day_ahead_config()), _autogluon_model_config()),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def run_lstm_pipeline():
    config = _lstm_config()
    data = _load_data(config)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "lstm")
    assert isinstance(pipeline.model, LSTMForecaster)
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    assert {
        "0.1",
        "0.9",
        "prediction_lower",
        "prediction_upper",
        "prediction_density",
        "lstm_weights",
        "lstm_means",
        "lstm_stds",
        "distribution_values",
        "distribution_probabilities",
    }.issubset(result.forecast.frame.columns)
    assert pipeline.model.context_length == 672
    loaded = LSTMForecaster.load(result.model_path, device="cpu")
    assert loaded.fitted_ is True
    online_covariates = result.test_data.frame[
        [result.test_data.timestamp_col, result.test_data.item_id_col, *SELECTED_FEATURE_COLUMNS]
    ].head(config.scale.prediction_length)
    online_forecast = loaded.predict(
        result.train_data,
        prediction_length=config.scale.prediction_length,
        freq=config.scale.freq,
        known_covariates=online_covariates,
    )
    assert len(online_forecast.frame) == config.scale.prediction_length
    assert {"prediction", "0.1", "0.9", "lstm_weights", "lstm_means", "lstm_stds"}.issubset(
        online_forecast.frame.columns
    )
    assert result.training_history_csv_path is not None
    assert result.training_history_csv_path.exists()


def run_lstm_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _lstm_config(),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def _lstm_config() -> PipelineConfig:
    base_config = _with_time_range(_day_ahead_config())
    return _with_model(
        base_config,
        ModelSpecConfig(
            name="lstm",
            random_state=42,
            params={
                "context_length": 672,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "epochs": 50,
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_train_samples": 4096,
                "train_sample_strategy": "last",
                "device": "cpu",
                "validation_fraction": 0.1,
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 1e-4,
                "use_point_head": False,
                "nll_loss_weight": 1.0,
                "point_loss_weight": 0.0,
            },
        ),
    )


def run_lstm_config_params():
    config = _day_ahead_config()
    forecaster = LSTMForecaster.from_config(
        ModelSpecConfig(
            name="lstm",
            random_state=42,
            params={
                "max_train_samples": 8192,
                "train_sample_strategy": "last",
                "weight_decay": 1e-4,
                "use_point_head": True,
                "nll_loss_weight": 1.0,
                "point_loss_weight": 0.1,
                "add_calendar_features": True,
                "add_lag_features": True,
                "lag_feature_steps": (96, 192, 672),
            },
        ),
        config.scale,
    )

    assert forecaster.max_train_samples == 8192
    assert forecaster.train_sample_strategy == "last"
    assert forecaster.weight_decay == 1e-4
    assert forecaster.use_point_head is True
    assert forecaster.nll_loss_weight == 1.0
    assert forecaster.point_loss_weight == 0.1
    assert forecaster.add_calendar_features is True
    assert forecaster.add_lag_features is True
    assert forecaster.lag_feature_steps == (96, 192, 672)


def run_bilstm_pipeline():
    config = _bilstm_config()
    data = _load_data(config)

    pipeline = LoadForecastPipeline(config)
    result = pipeline.run(data)

    _assert_day_ahead_run_outputs(result, "bilstm")
    assert isinstance(pipeline.model, LSTMForecaster)
    assert pipeline.model.bidirectional is True
    loaded = LSTMForecaster.load(result.model_path, device="cpu")
    assert loaded.bidirectional is True
    _assert_basic_forecast_result(result, config.scale.prediction_length)
    assert {
        "0.1",
        "0.9",
        "prediction_lower",
        "prediction_upper",
        "prediction_density",
        "lstm_weights",
        "lstm_means",
        "lstm_stds",
        "distribution_values",
        "distribution_probabilities",
    }.issubset(result.forecast.frame.columns)


def run_bilstm_pipeline_with_parameter_tuning(
    enable_parameter_tuning: bool = False,
    n_trials: int = 2,
    save_trial_outputs: bool = True,
):
    _run_pipeline_with_parameter_tuning(
        _bilstm_config(),
        enable_parameter_tuning=enable_parameter_tuning,
        n_trials=n_trials,
        save_trial_outputs=save_trial_outputs,
    )


def _bilstm_config() -> PipelineConfig:
    base_config = _with_time_range(_day_ahead_config())
    return _with_model(
        base_config,
        ModelSpecConfig(
            name="bilstm",
            random_state=42,
            params={
                "context_length": 672,
                "hidden_size": 64,
                "num_layers": 1,
                "dropout": 0.1,
                "epochs": 50,
                "batch_size": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "max_train_samples": 4096,
                "train_sample_strategy": "last",
                "device": "cpu",
                "validation_fraction": 0.1,
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 1e-4,
                "use_point_head": False,
                "nll_loss_weight": 1.0,
                "point_loss_weight": 0.0,
            },
        ),
    )
