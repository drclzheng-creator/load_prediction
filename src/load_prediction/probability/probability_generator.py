"""Point-level scenario generation from probabilistic forecast outputs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from load_prediction.configs import PointScenarioConfig
from load_prediction.probability.forecast_detection import (
    detect_mixture_prefix,
    detect_quantile_columns,
    detect_scenario_source_type,
    has_mixture_parameters,
    has_quantile_columns,
    parse_array,
)
from load_prediction.probability.scenario_output import (
    PointScenarioOutput,
    default_probability_output_dir,
)
from load_prediction.probability.scenario_reduction import (
    point_scenarios_from_density_grid,
    point_scenarios_from_mixture,
    point_scenarios_from_quantiles,
    validate_point_scenario_config,
)

logger = logging.getLogger(__name__)


def generate_point_scenarios_from_mixture_forecast(
    forecast: pd.DataFrame,
    config: PointScenarioConfig | None = None,
    model_prefix: str | None = None,
) -> pd.DataFrame:
    """Add point-level discrete scenarios from GMM/MDN/LSTM mixture parameters."""

    scenario_config = config or PointScenarioConfig()
    validate_point_scenario_config(scenario_config)
    prefix = model_prefix or detect_mixture_prefix(forecast)
    weights_col = f"{prefix}_weights"
    means_col = f"{prefix}_means"
    stds_col = f"{prefix}_stds"
    missing = {weights_col, means_col, stds_col}.difference(forecast.columns)
    if missing:
        raise ValueError(f"Forecast is missing mixture parameter columns: {sorted(missing)}")

    rng = np.random.default_rng(scenario_config.random_state)
    output = forecast.copy()
    scenario_values: list[str] = []
    scenario_probabilities: list[str] = []
    for row in output[[weights_col, means_col, stds_col]].itertuples(index=False):
        values, probabilities = point_scenarios_from_mixture(
            weights=parse_array(row[0], weights_col),
            means=parse_array(row[1], means_col),
            stds=parse_array(row[2], stds_col),
            config=scenario_config,
            random_state=_row_seed(rng),
        )
        scenario_values.append(json.dumps(values.tolist()))
        scenario_probabilities.append(json.dumps(probabilities.tolist()))

    output[scenario_config.values_col] = scenario_values
    output[scenario_config.probabilities_col] = scenario_probabilities
    return output


def generate_point_scenarios_from_forecast(
    forecast: pd.DataFrame,
    config: PointScenarioConfig | None = None,
    model_prefix: str | None = None,
) -> pd.DataFrame:
    """Add point-level scenarios using mixture, quantile, or density-grid input."""

    scenario_config = config or PointScenarioConfig()
    validate_point_scenario_config(scenario_config)
    if has_mixture_parameters(forecast, model_prefix):
        return generate_point_scenarios_from_mixture_forecast(
            forecast,
            config=scenario_config,
            model_prefix=model_prefix,
        )
    if has_quantile_columns(forecast):
        return generate_point_scenarios_from_quantile_forecast(
            forecast,
            config=scenario_config,
        )
    if {"distribution_values", "distribution_probabilities"}.issubset(forecast.columns):
        return generate_point_scenarios_from_density_grid_forecast(
            forecast,
            config=scenario_config,
        )
    raise ValueError(
        "Forecast must contain either mixture parameter columns or "
        "quantile columns or distribution_values/distribution_probabilities columns"
    )


def generate_point_scenarios_from_quantile_forecast(
    forecast: pd.DataFrame,
    config: PointScenarioConfig | None = None,
) -> pd.DataFrame:
    """Add point-level scenarios by sampling the empirical quantile function."""

    scenario_config = config or PointScenarioConfig()
    validate_point_scenario_config(scenario_config)
    quantile_columns = detect_quantile_columns(forecast)
    if len(quantile_columns) < 2:
        raise ValueError("Forecast requires at least two quantile columns for quantile sampling")

    rng = np.random.default_rng(scenario_config.random_state)
    output = forecast.copy()
    scenario_values: list[str] = []
    scenario_probabilities: list[str] = []
    columns = [column for column, _ in quantile_columns]
    levels = np.asarray([level for _, level in quantile_columns], dtype=float)
    for row in output[columns].itertuples(index=False):
        values, probabilities = point_scenarios_from_quantiles(
            levels=levels,
            values=np.asarray(row, dtype=float),
            config=scenario_config,
            random_state=_row_seed(rng),
        )
        scenario_values.append(json.dumps(values.tolist()))
        scenario_probabilities.append(json.dumps(probabilities.tolist()))

    output[scenario_config.values_col] = scenario_values
    output[scenario_config.probabilities_col] = scenario_probabilities
    return output


def generate_point_scenarios_from_density_grid_forecast(
    forecast: pd.DataFrame,
    config: PointScenarioConfig | None = None,
) -> pd.DataFrame:
    """Add point-level scenarios from density grids by converting density to cell mass."""

    scenario_config = config or PointScenarioConfig()
    validate_point_scenario_config(scenario_config)
    missing = {"distribution_values", "distribution_probabilities"}.difference(forecast.columns)
    if missing:
        raise ValueError(f"Forecast is missing distribution grid columns: {sorted(missing)}")

    rng = np.random.default_rng(scenario_config.random_state)
    output = forecast.copy()
    scenario_values: list[str] = []
    scenario_probabilities: list[str] = []
    for row in output[["distribution_values", "distribution_probabilities"]].itertuples(index=False):
        values, probabilities = point_scenarios_from_density_grid(
            values=parse_array(row[0], "distribution_values"),
            densities=parse_array(row[1], "distribution_probabilities"),
            config=scenario_config,
            random_state=_row_seed(rng),
        )
        scenario_values.append(json.dumps(values.tolist()))
        scenario_probabilities.append(json.dumps(probabilities.tolist()))

    output[scenario_config.values_col] = scenario_values
    output[scenario_config.probabilities_col] = scenario_probabilities
    return output


def save_point_scenarios_from_forecast_csv(
    forecast_csv_path: str | Path,
    output_dir: str | Path | None = None,
    config: PointScenarioConfig | None = None,
    model_prefix: str | None = None,
    output_filename: str = "point_scenarios.csv",
) -> PointScenarioOutput:
    """Generate point scenarios from a forecast CSV and save them under outputs."""

    source = Path(forecast_csv_path)
    forecast = pd.read_csv(source)
    scenario_config = config or PointScenarioConfig()
    scenarios = generate_point_scenarios_from_forecast(
        forecast,
        config=scenario_config,
        model_prefix=model_prefix,
    )
    source_type = detect_scenario_source_type(forecast, model_prefix)
    prefix = detect_mixture_prefix(forecast) if source_type == "mixture" and model_prefix is None else model_prefix
    target_dir = Path(output_dir) if output_dir is not None else default_probability_output_dir(source)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / output_filename
    scenarios.to_csv(csv_path, index=False)

    metadata_path = target_dir / "point_scenarios_metadata.json"
    metadata = {
        "source_forecast_csv": str(source),
        "output_csv": str(csv_path),
        "source_type": source_type,
        "model_prefix": prefix,
        "num_rows": int(len(scenarios)),
        "num_scenarios": int(scenario_config.num_scenarios),
        "num_samples": int(scenario_config.num_samples),
        "random_state": scenario_config.random_state,
        "clip_min": scenario_config.clip_min,
        "kmeans_n_init": int(scenario_config.kmeans_n_init),
        "kmeans_max_iter": int(scenario_config.kmeans_max_iter),
        "kmeans_fit_samples": int(scenario_config.kmeans_fit_samples),
        "values_col": scenario_config.values_col,
        "probabilities_col": scenario_config.probabilities_col,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Saved point scenarios rows=%s csv=%s metadata=%s",
        len(scenarios),
        csv_path,
        metadata_path,
    )
    return PointScenarioOutput(
        frame=scenarios,
        csv_path=csv_path,
        metadata_path=metadata_path,
    )


def _row_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))

