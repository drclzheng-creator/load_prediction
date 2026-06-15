"""Forecast probability column detection and parsing helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def detect_mixture_prefix(forecast: pd.DataFrame) -> str:
    for prefix in ("gmm", "mdn", "lstm"):
        if {f"{prefix}_weights", f"{prefix}_means", f"{prefix}_stds"}.issubset(forecast.columns):
            return prefix
    raise ValueError(
        "Forecast must contain GMM, MDN, or LSTM-MDN mixture columns: "
        "gmm_weights/gmm_means/gmm_stds, mdn_weights/mdn_means/mdn_stds, "
        "or lstm_weights/lstm_means/lstm_stds"
    )


def has_mixture_parameters(forecast: pd.DataFrame, model_prefix: str | None = None) -> bool:
    if model_prefix is not None:
        return {f"{model_prefix}_weights", f"{model_prefix}_means", f"{model_prefix}_stds"}.issubset(
            forecast.columns
        )
    return any(
        {f"{prefix}_weights", f"{prefix}_means", f"{prefix}_stds"}.issubset(forecast.columns)
        for prefix in ("gmm", "mdn", "lstm")
    )


def detect_quantile_columns(forecast: pd.DataFrame) -> list[tuple[str, float]]:
    quantile_columns: list[tuple[str, float]] = []
    for column in forecast.columns:
        try:
            level = float(column)
        except (TypeError, ValueError):
            continue
        if 0.0 < level < 1.0:
            quantile_columns.append((column, level))
    return sorted(quantile_columns, key=lambda item: item[1])


def has_quantile_columns(forecast: pd.DataFrame) -> bool:
    return len(detect_quantile_columns(forecast)) >= 2


def detect_scenario_source_type(forecast: pd.DataFrame, model_prefix: str | None = None) -> str:
    if has_mixture_parameters(forecast, model_prefix):
        return "mixture"
    if has_quantile_columns(forecast):
        return "quantile_function"
    if {"distribution_values", "distribution_probabilities"}.issubset(forecast.columns):
        return "density_grid"
    raise ValueError(
        "Forecast must contain either mixture parameter columns, quantile columns, "
        "or distribution_values/distribution_probabilities columns"
    )


def parse_array(value: Any, column_name: str) -> np.ndarray:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Column {column_name} contains invalid JSON") from exc
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"Column {column_name} contains an empty array")
    if not np.isfinite(array).all():
        raise ValueError(f"Column {column_name} contains non-finite values")
    return array

