"""Pipeline run result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from load_prediction.data.data_schema import TimeSeriesDataset
from load_prediction.models import ForecastFrame


@dataclass
class PipelineRunResult:
    scale_name: str
    forecast: ForecastFrame
    metrics: dict[str, float] | None
    train_data: TimeSeriesDataset
    test_data: TimeSeriesDataset | None
    output_dir: Path | None = None
    model_path: Path | None = None
    forecast_csv_path: Path | None = None
    forecast_plot_path: Path | None = None
    metrics_path: Path | None = None
    training_history_csv_path: Path | None = None
    training_history_json_path: Path | None = None
    training_curves_path: Path | None = None

