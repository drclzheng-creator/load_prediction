"""Visualization utilities for forecast diagnostics."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from load_prediction.models.base_forecaster import ForecastFrame

logger = logging.getLogger(__name__)


def plot_forecast_vs_actual(
    actual: pd.DataFrame,
    forecast: ForecastFrame,
    target_col: str = "target",
    output_path: str | Path = "outputs/forecast_vs_actual.png",
    item_id: str | None = None,
    title: str | None = None,
) -> Path:
    """Plot test-set actual values against forecast values.

    Matplotlib is imported lazily so training and batch prediction do not depend
    on plotting unless this function is called.
    """

    plt = _pyplot()

    joined = actual.merge(
        forecast.frame,
        on=[forecast.item_id_col, forecast.timestamp_col],
        how="inner",
    )
    if joined.empty:
        logger.warning("Cannot plot forecast: no overlap between actual and forecast")
        raise ValueError("Forecast and actual frames do not overlap on item_id/timestamp")

    if item_id is None:
        item_id = str(joined[forecast.item_id_col].iloc[0])
    plot_data = joined[joined[forecast.item_id_col].astype(str) == str(item_id)].copy()
    if plot_data.empty:
        logger.warning("Cannot plot forecast: no rows found for item_id=%s", item_id)
        raise ValueError(f"No rows found for item_id={item_id}")

    plot_data = plot_data.sort_values(forecast.timestamp_col)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        plot_data[forecast.timestamp_col],
        plot_data[target_col],
        label="Actual",
        color="#1f77b4",
        linewidth=1.8,
    )
    ax.plot(
        plot_data[forecast.timestamp_col],
        plot_data[forecast.prediction_col],
        label="Forecast",
        color="#d62728",
        linewidth=1.8,
    )
    interval_lower_col = "0.1" if "0.1" in plot_data.columns else "prediction_lower"
    interval_upper_col = "0.9" if "0.9" in plot_data.columns else "prediction_upper"
    if {interval_lower_col, interval_upper_col}.issubset(plot_data.columns):
        ax.fill_between(
            plot_data[forecast.timestamp_col],
            plot_data[interval_lower_col],
            plot_data[interval_upper_col],
            color="#d62728",
            alpha=0.15,
            label="P10-P90 interval" if interval_lower_col == "0.1" else "Prediction interval",
        )

    ax.set_title(title or f"Forecast vs Actual - {item_id}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(target_col)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    logger.info("Saved forecast plot rows=%s item_id=%s path=%s", len(plot_data), item_id, output)
    return output


def plot_training_curves(
    history: pd.DataFrame,
    output_path: str | Path = "outputs/training_curves.png",
    title: str = "Training Curves",
) -> Path:
    """Plot training diagnostics such as loss and learning rate."""

    if history.empty:
        logger.warning("Cannot plot training curves: empty history")
        raise ValueError("Training history is empty")
    required = {"epoch", "train_loss"}
    missing = required - set(history.columns)
    if missing:
        logger.warning("Cannot plot training curves: missing columns=%s", sorted(missing))
        raise ValueError(f"Training history missing columns: {sorted(missing)}")

    plt = _pyplot()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_data = history.sort_values("epoch").copy()

    has_lr = "learning_rate" in plot_data.columns and plot_data["learning_rate"].notna().any()
    fig, axes = plt.subplots(2 if has_lr else 1, 1, figsize=(10, 7 if has_lr else 4), sharex=True)
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    loss_ax = axes[0]
    loss_ax.plot(
        plot_data["epoch"],
        plot_data["train_loss"],
        marker="o",
        linewidth=1.8,
        label="Train loss",
        color="#1f77b4",
    )
    if "validation_loss" in plot_data.columns and plot_data["validation_loss"].notna().any():
        loss_ax.plot(
            plot_data["epoch"],
            plot_data["validation_loss"],
            marker="o",
            linewidth=1.8,
            label="Validation loss",
            color="#d62728",
        )
    loss_ax.set_title(title)
    loss_ax.set_ylabel("Pinball loss")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend(loc="best")

    if has_lr:
        lr_ax = axes[1]
        lr_ax.plot(
            plot_data["epoch"],
            plot_data["learning_rate"],
            marker="o",
            linewidth=1.8,
            label="Learning rate",
            color="#2ca02c",
        )
        lr_ax.set_ylabel("Learning rate")
        lr_ax.grid(True, alpha=0.25)
        lr_ax.legend(loc="best")
        lr_ax.set_xlabel("Epoch")
    else:
        loss_ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    logger.info("Saved training curves rows=%s path=%s", len(plot_data), output)
    return output


def _pyplot():
    mpl_config_dir = Path("tmp/matplotlib")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
