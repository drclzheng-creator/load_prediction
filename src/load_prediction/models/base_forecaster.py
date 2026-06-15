"""Forecast model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from load_prediction.data.data_schema import TimeSeriesDataset


@dataclass
class ForecastFrame:
    frame: pd.DataFrame
    timestamp_col: str = "timestamp"
    item_id_col: str = "item_id"
    prediction_col: str = "prediction"


class BaseForecastModel(ABC):
    """Interface for load forecasting model adapters."""

    @abstractmethod
    def fit(self, data: TimeSeriesDataset) -> "BaseForecastModel":
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        history: TimeSeriesDataset,
        prediction_length: int,
        freq: str,
        known_covariates: pd.DataFrame | None = None,
    ) -> ForecastFrame:
        raise NotImplementedError

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "model.joblib"
        joblib.dump(self, model_path)
        return model_path

    def training_log(self) -> dict[str, Any]:
        return {}
