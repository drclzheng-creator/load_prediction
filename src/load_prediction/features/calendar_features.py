"""Calendar and cyclical time feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from load_prediction.features.chinese_calendar_features import (
    build_chinese_calendar_feature_frame,
    build_chinese_calendar_features,
)


def build_calendar_feature_frame(
    timestamps: pd.Series,
    *,
    add_calendar: bool = True,
    add_cyclical_time: bool = True,
    add_chinese_calendar: bool = False,
) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps)
    features = pd.DataFrame(index=range(len(ts)))
    if not add_calendar:
        return features

    features["feature__calendar__dayofweek"] = ts.dt.dayofweek.to_numpy()
    features["feature__calendar__is_weekend"] = (ts.dt.dayofweek >= 5).astype(int).to_numpy()
    features["feature__calendar__year"] = ts.dt.year.to_numpy()
    features["feature__calendar__month"] = ts.dt.month.to_numpy()
    features["feature__calendar__day"] = ts.dt.day.to_numpy()
    features["feature__calendar__quarter"] = ts.dt.quarter.to_numpy()
    features["feature__calendar__dayofyear"] = ts.dt.dayofyear.to_numpy()
    features["feature__calendar__hour"] = ts.dt.hour.to_numpy()
    features["feature__calendar__minute"] = ts.dt.minute.to_numpy()

    if add_cyclical_time:
        minute_of_day = ts.dt.hour.to_numpy() * 60 + ts.dt.minute.to_numpy()
        features["feature__time__sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440)
        features["feature__time__cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440)
        features["feature__time__sin_week"] = np.sin(
            2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
        )
        features["feature__time__cos_week"] = np.cos(
            2 * np.pi * (ts.dt.dayofweek.to_numpy() * 1440 + minute_of_day) / (7 * 1440)
        )

    if add_chinese_calendar:
        features = pd.concat(
            [features, build_chinese_calendar_feature_frame(ts)],
            axis=1,
        )

    return features


def build_calendar_features(
    timestamp: pd.Timestamp,
    *,
    add_calendar: bool = True,
    add_cyclical_time: bool = True,
    add_chinese_calendar: bool = False,
) -> dict[str, float]:
    if not add_calendar:
        return {}

    timestamp = pd.Timestamp(timestamp)
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    features = {
        "feature__calendar__dayofweek": float(timestamp.dayofweek),
        "feature__calendar__is_weekend": float(timestamp.dayofweek >= 5),
        "feature__calendar__year": float(timestamp.year),
        "feature__calendar__month": float(timestamp.month),
        "feature__calendar__day": float(timestamp.day),
        "feature__calendar__quarter": float(timestamp.quarter),
        "feature__calendar__dayofyear": float(timestamp.dayofyear),
        "feature__calendar__hour": float(timestamp.hour),
        "feature__calendar__minute": float(timestamp.minute),
    }
    if add_cyclical_time:
        features.update(
            {
                "feature__time__sin_day": float(np.sin(2 * np.pi * minute_of_day / 1440)),
                "feature__time__cos_day": float(np.cos(2 * np.pi * minute_of_day / 1440)),
                "feature__time__sin_week": float(
                    np.sin(2 * np.pi * (timestamp.dayofweek * 1440 + minute_of_day) / (7 * 1440))
                ),
                "feature__time__cos_week": float(
                    np.cos(2 * np.pi * (timestamp.dayofweek * 1440 + minute_of_day) / (7 * 1440))
                ),
            }
        )
    if add_chinese_calendar:
        features.update(build_chinese_calendar_features(timestamp))
    return features
