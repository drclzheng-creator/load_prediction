"""Forecast evaluation metrics."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from load_prediction.models.base_forecaster import ForecastFrame

logger = logging.getLogger(__name__)


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.mean(np.abs(actual.to_numpy() - predicted.to_numpy())))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - predicted.to_numpy()) ** 2)))


def mape(actual: pd.Series, predicted: pd.Series, epsilon: float = 1e-6) -> float:
    actual_values = actual.to_numpy()
    predicted_values = predicted.to_numpy()
    denominator = np.maximum(np.abs(actual_values), epsilon)
    return float(np.mean(np.abs((actual_values - predicted_values) / denominator)))


def smape(actual: pd.Series, predicted: pd.Series, epsilon: float = 1e-6) -> float:
    actual_values = actual.to_numpy()
    predicted_values = predicted.to_numpy()
    denominator = np.maximum(np.abs(actual_values) + np.abs(predicted_values), epsilon)
    return float(np.mean(2.0 * np.abs(actual_values - predicted_values) / denominator))


def wape(actual: pd.Series, predicted: pd.Series, epsilon: float = 1e-6) -> float:
    actual_values = actual.to_numpy()
    predicted_values = predicted.to_numpy()
    return float(np.sum(np.abs(actual_values - predicted_values)) / max(np.sum(np.abs(actual_values)), epsilon))


def pinball_loss(actual: pd.Series, predicted_quantile: pd.Series, quantile: float) -> float:
    errors = actual.to_numpy() - predicted_quantile.to_numpy()
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def evaluate_forecast(
    actual: pd.DataFrame,
    forecast: ForecastFrame,
    target_col: str = "target",
) -> dict[str, float]:
    joined = actual.merge(
        forecast.frame,
        on=[forecast.item_id_col, forecast.timestamp_col],
        how="inner",
    )
    if joined.empty:
        raise ValueError("Forecast and actual frames do not overlap on item_id/timestamp")

    actual_series = pd.to_numeric(joined[target_col], errors="coerce")
    predicted_series = pd.to_numeric(joined[forecast.prediction_col], errors="coerce")
    valid = actual_series.notna() & predicted_series.notna()
    if not valid.any():
        raise ValueError("No valid numeric values for evaluation")

    actual_series = actual_series[valid]
    predicted_series = predicted_series[valid]
    metrics = {
        "mae": mae(actual_series, predicted_series),
        "rmse": rmse(actual_series, predicted_series),
        "mape": mape(actual_series, predicted_series),
        "smape": smape(actual_series, predicted_series),
        "wape": wape(actual_series, predicted_series),
        "rows": float(len(actual_series)),
    }
    metrics.update(_interval_metrics(joined.loc[valid], actual_series))
    metrics.update(_quantile_metrics(joined.loc[valid], actual_series))
    metrics.update(_horizon_metrics(joined.loc[valid], actual_series, predicted_series))
    logger.info("Evaluated forecast metrics=%s", metrics)
    return metrics


def _interval_metrics(joined: pd.DataFrame, actual_series: pd.Series) -> dict[str, float]:
    interval_columns = _select_interval_columns(joined)
    if interval_columns is None:
        return {}
    lower_col, upper_col, prefix = interval_columns
    lower = pd.to_numeric(joined[lower_col], errors="coerce")
    upper = pd.to_numeric(joined[upper_col], errors="coerce")
    valid = actual_series.notna() & lower.notna() & upper.notna()
    if not valid.any():
        return {}

    actual_values = actual_series[valid]
    lower_values = lower[valid]
    upper_values = upper[valid]
    width = (upper_values - lower_values).clip(lower=0.0)
    return {
        f"{prefix}_coverage": float(((actual_values >= lower_values) & (actual_values <= upper_values)).mean()),
        f"{prefix}_avg_width": float(width.mean()),
        f"{prefix}_normalized_avg_width": float(width.mean() / max(float(actual_values.abs().mean()), 1e-6)),
    }


def _select_interval_columns(joined: pd.DataFrame) -> tuple[str, str, str] | None:
    if {"0.1", "0.9"}.issubset(joined.columns):
        return "0.1", "0.9", "p10_p90"
    if {"prediction_lower", "prediction_upper"}.issubset(joined.columns):
        return "prediction_lower", "prediction_upper", "prediction_interval"
    return None


def _quantile_metrics(joined: pd.DataFrame, actual_series: pd.Series) -> dict[str, float]:
    metrics: dict[str, float] = {}
    losses = []
    for column, quantile in _quantile_columns(joined):
        predicted_quantile = pd.to_numeric(joined[column], errors="coerce")
        valid = actual_series.notna() & predicted_quantile.notna()
        if not valid.any():
            continue
        loss = pinball_loss(actual_series[valid], predicted_quantile[valid], quantile)
        key = f"pinball_loss_p{int(round(quantile * 100)):02d}"
        metrics[key] = loss
        losses.append(loss)
    if losses:
        metrics["pinball_loss_mean"] = float(np.mean(losses))
    return metrics


def _quantile_columns(frame: pd.DataFrame) -> list[tuple[str, float]]:
    quantiles = []
    for column in frame.columns:
        try:
            quantile = float(column)
        except (TypeError, ValueError):
            continue
        if 0.0 < quantile < 1.0:
            quantiles.append((column, quantile))
    return sorted(quantiles, key=lambda item: item[1])


def _horizon_metrics(
    joined: pd.DataFrame,
    actual_series: pd.Series,
    predicted_series: pd.Series,
) -> dict[str, float]:
    if "horizon_step" not in joined.columns:
        return {}
    metrics: dict[str, float] = {}
    horizon = pd.to_numeric(joined["horizon_step"], errors="coerce")
    valid = horizon.notna() & actual_series.notna() & predicted_series.notna()
    if not valid.any():
        return {}

    horizon_frame = pd.DataFrame(
        {
            "horizon_step": horizon[valid].astype(int),
            "actual": actual_series[valid].astype(float),
            "predicted": predicted_series[valid].astype(float),
        }
    )
    for step, group in horizon_frame.groupby("horizon_step", sort=True):
        suffix = f"horizon_{int(step):03d}"
        actual_step = group["actual"]
        predicted_step = group["predicted"]
        metrics[f"mae_{suffix}"] = mae(actual_step, predicted_step)
        metrics[f"rmse_{suffix}"] = rmse(actual_step, predicted_step)
        metrics[f"mape_{suffix}"] = mape(actual_step, predicted_step)
    return metrics
