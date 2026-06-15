"""Forecast model specification configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpecConfig:
    """Model family, random seed, preprocessing flags, and model parameters."""

    name: str = "sklearn_random_forest"
    random_state: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    scale_features: bool = False
    scale_target: bool = False

