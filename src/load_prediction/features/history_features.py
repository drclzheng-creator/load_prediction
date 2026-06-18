"""Historical target lag and rolling feature builders."""

from __future__ import annotations

import pandas as pd


def build_history_feature_frame(
    target: pd.Series,
    *,
    lag_steps: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> pd.DataFrame:
    numeric_target = pd.to_numeric(target, errors="coerce")
    features = pd.DataFrame(index=target.index)
    for lag in lag_steps:
        features[f"feature__lag__{lag}"] = numeric_target.shift(lag)
    shifted = numeric_target.shift(1)
    for window in rolling_windows:
        features[f"feature__rolling_mean__{window}"] = shifted.rolling(window).mean()
        features[f"feature__rolling_std__{window}"] = shifted.rolling(window).std().fillna(0)
    return features.reset_index(drop=True)


def build_history_features(
    history: pd.DataFrame,
    target_col: str,
    *,
    lag_steps: tuple[int, ...],
    rolling_windows: tuple[int, ...],
) -> dict[str, float]:
    target = pd.to_numeric(history[target_col], errors="coerce").dropna()
    features: dict[str, float] = {}
    for lag in lag_steps:
        features[f"feature__lag__{lag}"] = float(target.iloc[-lag]) if len(target) >= lag else 0.0
    for window in rolling_windows:
        window_values = target.iloc[-window:]
        features[f"feature__rolling_mean__{window}"] = (
            float(window_values.mean()) if not window_values.empty else 0.0
        )
        features[f"feature__rolling_std__{window}"] = (
            float(window_values.std(ddof=1)) if len(window_values) > 1 else 0.0
        )
    return features
