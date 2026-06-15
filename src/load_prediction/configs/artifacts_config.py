"""Pipeline artifact output configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact storage location, save switches, and artifact filenames."""

    root_dir: str = "outputs"
    run_name: str | None = None
    save_model: bool = True
    save_artifacts: bool = True
    save_plot: bool = True
    save_metrics: bool = True
    forecast_filename: str = "forecast.csv"
    plot_filename: str = "forecast.png"
    metrics_filename: str = "metrics.json"
    train_log_filename: str = "train.log"
    training_history_filename: str = "training_history.csv"
    training_history_json_filename: str = "training_history.json"
    training_curves_filename: str = "training_curves.png"

