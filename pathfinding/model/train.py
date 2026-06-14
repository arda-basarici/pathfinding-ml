"""Train the cost-to-go regressor.

Model choice is deliberate: gradient-boosted trees, not a neural net. On small tabular
feature sets, boosted trees are the strong default and the right-tool-for-the-job
choice; reaching for a net here would be the wrong-tool signal (and is Phase 3's
territory anyway). The point of the project is the *measurement*, not the architecture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def train_model(X_train: np.ndarray, y_train: np.ndarray):
    """Fit and return a regressor predicting cost-to-go from feature vectors."""
    raise NotImplementedError  # build step 6


def save_model(model, path: Path) -> None:
    """Persist a trained model to ``path``."""
    raise NotImplementedError  # build step 6


def load_model(path: Path):
    """Load a model previously saved with ``save_model``."""
    raise NotImplementedError  # build step 6
