"""The baseline cost-to-go predictor: the classic Manhattan heuristic.

A model only earns its keep if it beats a *fair* baseline. Here the baseline isn't
"predict the mean" — it's the admissible heuristic A* already uses. Beating Manhattan
on prediction error is necessary but not sufficient; what ultimately matters is
whether it expands fewer nodes without wrecking path quality (see evaluation/).

Exposes the same ``fit``/``predict`` interface as the trained model, so evaluation can
treat the two interchangeably.
"""

from __future__ import annotations

import numpy as np

from ..data.features import FEATURE_NAMES

# Which column of a feature vector holds the Manhattan distance.
_MANHATTAN_COL = FEATURE_NAMES.index("manhattan_to_goal")


class ManhattanBaseline:
    """Predicts cost-to-go as the Manhattan-distance feature (nothing to fit)."""

    def fit(self, X=None, y=None) -> "ManhattanBaseline":
        """No-op. Present only so the baseline matches the model's fit/predict API."""
        return self

    def predict(self, X) -> np.ndarray:
        """Return the ``manhattan_to_goal`` column straight through as the estimate."""
        X = np.asarray(X, dtype=float)
        return X[:, _MANHATTAN_COL]
