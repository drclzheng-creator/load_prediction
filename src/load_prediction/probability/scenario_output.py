"""Point scenario output contracts and save helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PointScenarioOutput:
    """Saved point scenario generation result."""

    frame: pd.DataFrame
    csv_path: Path
    metadata_path: Path


def default_probability_output_dir(forecast_csv_path: Path) -> Path:
    if forecast_csv_path.parent.name == "artifacts":
        return forecast_csv_path.parent.parent / "probability"
    return forecast_csv_path.parent / "probability"

