"""Run the synthetic load forecasting demo without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from load_prediction.cli import app  # noqa: E402


if __name__ == "__main__":
    app()
