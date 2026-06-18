"""Chinese holiday calendar features.

The feature functions use the optional ``chinesecalendar`` package when it is
available. If it is not installed, stable zero-valued columns are returned so
pipelines can still run in environments that have not installed the dependency.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHINESE_CALENDAR_FEATURE_COLUMNS = (
    "feature__chinese_calendar__is_holiday",
    "feature__chinese_calendar__is_workday",
    "feature__chinese_calendar__is_in_lieu",
    "feature__chinese_calendar__is_pre_holiday",
    "feature__chinese_calendar__is_post_holiday",
)

try:  # pragma: no cover - exercised when optional dependency is installed.
    from chinese_calendar import get_holiday_detail, is_holiday, is_in_lieu, is_workday
except ImportError:  # pragma: no cover - current CI/dev env may omit optional dependency.
    get_holiday_detail = None
    is_holiday = None
    is_in_lieu = None
    is_workday = None


def build_chinese_calendar_feature_frame(timestamps: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(timestamps)
    features = pd.DataFrame(index=range(len(ts)))
    if not _chinese_calendar_available():
        _log_missing_dependency()
        for column in CHINESE_CALENDAR_FEATURE_COLUMNS:
            features[column] = 0.0
        return features

    normalized = ts.dt.normalize()
    holiday_flags = np.asarray([_is_holiday(timestamp) for timestamp in normalized], dtype=float)
    workday_flags = np.asarray([_is_workday(timestamp) for timestamp in normalized], dtype=float)
    in_lieu_flags = np.asarray([_is_in_lieu(timestamp) for timestamp in normalized], dtype=float)
    pre_holiday_flags = np.asarray(
        [_is_holiday(timestamp + pd.Timedelta(days=1)) for timestamp in normalized],
        dtype=float,
    )
    post_holiday_flags = np.asarray(
        [_is_holiday(timestamp - pd.Timedelta(days=1)) for timestamp in normalized],
        dtype=float,
    )
    features["feature__chinese_calendar__is_holiday"] = holiday_flags
    features["feature__chinese_calendar__is_workday"] = workday_flags
    features["feature__chinese_calendar__is_in_lieu"] = in_lieu_flags
    features["feature__chinese_calendar__is_pre_holiday"] = pre_holiday_flags
    features["feature__chinese_calendar__is_post_holiday"] = post_holiday_flags
    return features


def build_chinese_calendar_features(timestamp: pd.Timestamp) -> dict[str, float]:
    frame = build_chinese_calendar_feature_frame(pd.Series([timestamp]))
    return {column: float(frame[column].iloc[0]) for column in CHINESE_CALENDAR_FEATURE_COLUMNS}


def _chinese_calendar_available() -> bool:
    return is_holiday is not None and is_workday is not None and is_in_lieu is not None


def _log_missing_dependency() -> None:
    if not getattr(_log_missing_dependency, "logged", False):
        logger.warning(
            "chinesecalendar is not installed; Chinese calendar features will be zero-filled"
        )
        _log_missing_dependency.logged = True


def _is_holiday(timestamp: pd.Timestamp) -> bool:
    assert is_holiday is not None
    return bool(is_holiday(pd.Timestamp(timestamp).date()))


def _is_workday(timestamp: pd.Timestamp) -> bool:
    assert is_workday is not None
    return bool(is_workday(pd.Timestamp(timestamp).date()))


def _is_in_lieu(timestamp: pd.Timestamp) -> bool:
    assert is_in_lieu is not None
    return bool(is_in_lieu(pd.Timestamp(timestamp).date()))
