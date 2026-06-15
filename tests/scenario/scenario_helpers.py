from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from load_prediction.probability import (
    PointScenarioConfig,
    generate_point_scenarios_from_mixture_forecast,
    generate_point_scenarios_from_quantile_forecast,
    save_point_scenarios_from_forecast_csv,
)

REAL_MDN_FORECAST_CSV = Path(
    "outputs/mdn/day_ahead/e2e_mdn/artifacts/forecast.csv"
)
REAL_GMM_FORECAST_CSV = Path(
    "outputs/gmm/day_ahead/e2e_probability_plots/artifacts/forecast.csv"
)
REAL_LIGHTGBM_FORECAST_CSV = Path(
    "outputs/lightgbm/day_ahead/artifacts/forecast.csv"
)
REAL_SKLEARN_FORECAST_CSV = Path(
    "outputs/sklearn_hist_gradient_boosting/day_ahead/artifacts/forecast.csv"
)
REAL_LSTM_FORECAST_CSV = Path(
    "outputs/lstm/day_ahead/artifacts/forecast.csv"
)
REAL_AUTOGLUON_FORECAST_CSV = Path(
    "outputs/autogluon/day_ahead/artifacts/forecast.csv"
)


def _assert_real_forecast_generates_point_scenarios(
    forecast_csv_path: Path,
    required_columns: set[str],
    generator: Callable[[pd.DataFrame, PointScenarioConfig], pd.DataFrame],
    expected_source_type: str,
    random_state: int = 11,
) -> None:
    if not forecast_csv_path.exists():
        raise FileNotFoundError(f"Real forecast CSV does not exist: {forecast_csv_path}")
    forecast = pd.read_csv(forecast_csv_path).head(10)
    assert required_columns.issubset(forecast.columns)
    config = PointScenarioConfig(random_state=random_state)

    scenarios = generator(forecast, config)

    for row in scenarios.itertuples():
        values = np.asarray(json.loads(row.point_scenario_values), dtype=float)
        probabilities = np.asarray(json.loads(row.point_scenario_probabilities), dtype=float)
        assert len(values) == config.num_scenarios
        np.testing.assert_allclose(probabilities.sum(), 1.0)

    output = save_point_scenarios_from_forecast_csv(
        forecast_csv_path,
        config=config,
    )

    saved = pd.read_csv(output.csv_path, nrows=10)
    saved_values = json.loads(saved["point_scenario_values"].iloc[0])
    saved_probabilities = json.loads(saved["point_scenario_probabilities"].iloc[0])
    metadata = json.loads(output.metadata_path.read_text(encoding="utf-8"))
    assert output.csv_path == forecast_csv_path.parent.parent / "probability" / "point_scenarios.csv"
    assert output.csv_path.exists()
    assert output.metadata_path == forecast_csv_path.parent.parent / "probability" / "point_scenarios_metadata.json"
    assert output.metadata_path.exists()
    assert {
        "item_id",
        "timestamp",
        "prediction",
        "point_scenario_values",
        "point_scenario_probabilities",
    }.issubset(saved.columns)
    assert len(saved_values) == config.num_scenarios
    assert metadata["source_type"] == expected_source_type


def run_mdn_scenarios():
    _assert_real_forecast_generates_point_scenarios(
        REAL_MDN_FORECAST_CSV,
        {"item_id", "timestamp", "mdn_weights", "mdn_means", "mdn_stds"},
        generate_point_scenarios_from_mixture_forecast,
        "mixture",
    )


def run_gmm_scenarios():
    _assert_real_forecast_generates_point_scenarios(
        REAL_GMM_FORECAST_CSV,
        {"item_id", "timestamp", "gmm_weights", "gmm_means", "gmm_stds"},
        generate_point_scenarios_from_mixture_forecast,
        "mixture",
    )


def run_lightgbm_scenarios():
    _assert_real_forecast_generates_point_scenarios(
        REAL_LIGHTGBM_FORECAST_CSV,
        {
            "item_id",
            "timestamp",
            "prediction",
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
            "distribution_values",
            "distribution_probabilities",
        },
        generate_point_scenarios_from_quantile_forecast,
        "quantile_function",
        random_state=13,
    )


def run_sklearn_scenarios():
    _assert_real_forecast_generates_point_scenarios(
        REAL_SKLEARN_FORECAST_CSV,
        {
            "item_id",
            "timestamp",
            "prediction",
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
            "distribution_values",
            "distribution_probabilities",
        },
        generate_point_scenarios_from_quantile_forecast,
        "quantile_function",
        random_state=17,
    )


def run_lstm_scenarios():
    _assert_real_forecast_generates_point_scenarios(
        REAL_LSTM_FORECAST_CSV,
        {"item_id", "timestamp", "prediction", "lstm_weights", "lstm_means", "lstm_stds"},
        generate_point_scenarios_from_mixture_forecast,
        "mixture",
        random_state=19,
    )


def run_autogluon_scenarios():
    _assert_real_forecast_generates_point_scenarios(
        REAL_AUTOGLUON_FORECAST_CSV,
        {
            "item_id",
            "timestamp",
            "prediction",
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
            "distribution_values",
            "distribution_probabilities",
        },
        generate_point_scenarios_from_quantile_forecast,
        "quantile_function",
        random_state=23,
    )
