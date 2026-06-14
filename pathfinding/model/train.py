"""Train the cost-to-go regressor.

Model choice is deliberate (D6): gradient-boosted trees, not a neural net. On a small
tabular feature set, boosted trees are the strong default and the right-tool-for-the-job
choice; a net would be the wrong-tool signal (and is Phase 3's territory). We use
scikit-learn's ``HistGradientBoostingRegressor`` — the histogram-based variant (same
idea as LightGBM): it bins feature values, which makes it fast on tens of thousands of
rows while remaining ordinary gradient boosting underneath.

The hyperparameters below are sensible defaults, not a tuned configuration — this is a
literacy phase, and restraint is part of the point. ``random_state`` fixes the model's
internal randomness so training is reproducible.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    learning_rate: float = 0.1,   # contribution of each tree; smaller = gentler
    max_iter: int = 300,          # number of boosting rounds (trees)
    random_state: int = 0,        # reproducible fit
) -> HistGradientBoostingRegressor:
    """Fit and return a regressor predicting cost-to-go from feature vectors."""
    model = HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path: Path) -> None:
    """Persist a trained model to ``path`` (joblib; ships with scikit-learn)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path):
    """Load a model previously saved with ``save_model``."""
    return joblib.load(Path(path))
