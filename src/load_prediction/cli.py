"""Command-line entry points for load forecasting demos."""

from __future__ import annotations

from dataclasses import replace
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from load_prediction.configs import (
    ValidationConfig,
    ModelSpecConfig,
    ArtifactConfig,
    PipelineConfig,
    forecast_profile_preset,
)
from load_prediction.constants import FORECAST_PROFILE_DAY_AHEAD
from load_prediction.data import CSVLoadDataLoader
from load_prediction.pipeline.forecast_pipeline import LoadForecastPipeline, MultiScaleForecastRunner

app = typer.Typer(help="Multi-scale load forecasting CLI.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


@app.command()
def demo(
    scale: str = typer.Option(FORECAST_PROFILE_DAY_AHEAD, help="Scale preset name."),
    input_path: Path = typer.Option(Path("inputs/electricity.csv"), help="Dataset CSV path."),
    output_root: Path = typer.Option(Path("outputs"), help="Root directory for model and evaluation artifacts."),
    run_name: str | None = typer.Option(None, help="Optional run name below model/scale output directory."),
) -> None:
    """Run a demo using a real CSV dataset under inputs."""

    scale_config = forecast_profile_preset(scale)
    data = CSVLoadDataLoader(
        config=PipelineConfig().data.__class__(
            timestamp_col="timestamp",
            target_col="target",
            item_id_col="item_id",
            freq=scale_config.freq,
        )
    ).load(input_path)
    config = PipelineConfig(
        scale=scale_config,
        evaluation=ValidationConfig(train_ratio=0.7),
        model=ModelSpecConfig(name="lightgbm", params={"n_estimators": 80, "learning_rate": 0.05}),
        output=ArtifactConfig(root_dir=str(output_root), run_name=run_name),
    )
    result = LoadForecastPipeline(config).run(data)

    table = Table(title=f"Load Forecast Demo - {scale}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in (result.metrics or {}).items():
        table.add_row(key, f"{value:.6f}")
    table.add_row("forecast_rows", str(len(result.forecast.frame)))
    if result.forecast_csv_path:
        table.add_row("output", str(result.forecast_csv_path))
    if result.forecast_plot_path:
        table.add_row("plot", str(result.forecast_plot_path))
    if result.model_path:
        table.add_row("model", str(result.model_path))
    console.print(table)


@app.command("multi-demo")
def multi_demo(
    scales: str = typer.Option("ultra_short,day_ahead,hourly", help="Comma-separated scale presets."),
    input_path: Path = typer.Option(Path("inputs/electricity.csv"), help="Dataset CSV path."),
) -> None:
    """Run multiple forecast scales over the same real CSV dataset."""

    scale_names = [name.strip() for name in scales.split(",") if name.strip()]
    first_scale = forecast_profile_preset(scale_names[0])
    data = CSVLoadDataLoader(
        config=PipelineConfig().data.__class__(
            timestamp_col="timestamp",
            target_col="target",
            item_id_col="item_id",
            freq=first_scale.freq,
        )
    ).load(input_path)
    base_config = PipelineConfig(model=ModelSpecConfig(name="lightgbm", params={"n_estimators": 80}))
    results = MultiScaleForecastRunner(base_config, scale_names).run(data)

    table = Table(title="Multi-scale Forecast Demo")
    table.add_column("Scale")
    table.add_column("Rows", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MAPE", justify="right")
    for name, result in results.items():
        metrics = result.metrics or {}
        table.add_row(
            name,
            str(len(result.forecast.frame)),
            f"{metrics.get('mae', 0):.4f}",
            f"{metrics.get('rmse', 0):.4f}",
            f"{metrics.get('mape', 0):.4f}",
        )
    console.print(table)


@app.command()
def run_config(
    config_path: Path = typer.Argument(..., help="YAML pipeline config path."),
) -> None:
    """Train and evaluate from a YAML PipelineConfig."""

    config = PipelineConfig.from_yaml(config_path)
    if not config.data.path:
        raise typer.BadParameter("YAML data.path is required for run-config")

    data = CSVLoadDataLoader(config.data).load(config.data.path)
    result = LoadForecastPipeline(config).run(data)
    console.print(result.metrics)
    console.print(f"config: {config_path}")
    console.print(f"input: {config.data.path}")
    console.print(f"saved: {result.forecast_csv_path}")
    console.print(f"plot: {result.forecast_plot_path}")
    console.print(f"model: {result.model_path}")


@app.command()
def forecast(
    csv_path: Path = typer.Argument(..., help="CSV with timestamp, target, optional item_id/covariates."),
    scale: str = typer.Option(FORECAST_PROFILE_DAY_AHEAD, help="Scale preset name."),
    output_root: Path = typer.Option(Path("outputs"), help="Root directory for model and evaluation artifacts."),
    run_name: str | None = typer.Option(None, help="Optional run name below model/scale output directory."),
    timestamp_col: str = typer.Option("timestamp", help="Timestamp column in CSV."),
    target_col: str = typer.Option("target", help="Target load column in CSV."),
    item_id_col: str = typer.Option("item_id", help="Item ID column in CSV."),
    covariates: str = typer.Option("", help="Comma-separated known covariate columns."),
) -> None:
    """Train and evaluate on a local CSV."""

    scale_config = forecast_profile_preset(scale)
    covariate_cols = tuple(col.strip() for col in covariates.split(",") if col.strip())
    if covariate_cols:
        scale_config = replace(
            scale_config,
            feature=replace(scale_config.feature, known_covariates=covariate_cols),
        )
    config = PipelineConfig(
        scale=scale_config,
        evaluation=ValidationConfig(train_ratio=0.7),
    )
    config = PipelineConfig(
        data=config.data.__class__(
            timestamp_col=timestamp_col,
            target_col=target_col,
            item_id_col=item_id_col,
            freq=scale_config.freq,
            covariate_cols=covariate_cols,
        ),
        cleaning=config.cleaning,
        model=config.model,
        evaluation=config.evaluation,
        postprocess=config.postprocess,
        output=ArtifactConfig(root_dir=str(output_root), run_name=run_name),
        scale=scale_config,
    )
    data = CSVLoadDataLoader(config.data).load(csv_path)
    result = LoadForecastPipeline(config).run(data)
    console.print(result.metrics)
    console.print(f"saved: {result.forecast_csv_path}")
    console.print(f"plot: {result.forecast_plot_path}")
    console.print(f"model: {result.model_path}")
