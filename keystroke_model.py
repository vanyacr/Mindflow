"""Training and inference helpers for keystroke stress model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np

from keystroke import train_keystroke_model


def train_and_save_keystroke_model(
    X: np.ndarray,
    y: np.ndarray,
    out_path: str | Path = "keystroke_model.pkl",
) -> Dict[str, float]:
    """Wrapper used by teammate scripts/notebooks."""
    return train_keystroke_model(X=X, y=y, model_out=out_path)


def load_stress_model(model_path: str | Path = "keystroke_model.pkl"):
    """Load previously trained keystroke model."""
    return joblib.load(str(model_path))
