"""Pipeline file output helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from load_prediction.models import ForecastFrame

logger = logging.getLogger(__name__)


def save_forecast_csv(forecast: ForecastFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = ordered_forecast_frame(forecast)
    frame.to_csv(output, index=False)
    logger.info("Saved forecast CSV rows=%s path=%s", len(forecast.frame), output)


def ordered_forecast_frame(forecast: ForecastFrame) -> pd.DataFrame:
    frame = forecast.frame.copy()
    common_columns = [
        forecast.item_id_col,
        forecast.timestamp_col,
        "horizon_step",
        forecast.prediction_col,
        "0.1",
        "0.9",
        "prediction_lower",
        "prediction_upper",
        "prediction_density",
        "distribution_values",
        "distribution_probabilities",
        "evaluation_window",
    ]
    ordered = []
    for column in common_columns:
        if column in frame.columns and column not in ordered:
            ordered.append(column)
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame[ordered]

