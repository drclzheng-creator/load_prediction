"""Variational mode decomposition configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VMDConfig:
    """Numerical controls for variational mode decomposition."""

    num_modes: int = 4
    alpha: float = 2000.0
    tau: float = 0.0
    max_iter: int = 300
    tolerance: float = 1e-6
    init: str = "uniform"
    dc_mode: bool = False
