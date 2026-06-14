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

from ..data.features import DEFAULT_FEATURES


class ManhattanBaseline:
    """Predicts cost-to-go as the Manhattan-distance feature (nothing to fit).

    Config-aware: it locates ``manhattan_to_goal`` within the *active* feature set, so it
    still works under feature ablations — as long as that feature is present (it's the
    baseline, so requiring it is reasonable).
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        names = DEFAULT_FEATURES if feature_names is None else feature_names
        if "manhattan_to_goal" not in names:
            raise ValueError(
                "ManhattanBaseline requires 'manhattan_to_goal' in the feature set"
            )
        self._col = names.index("manhattan_to_goal")

    def fit(self, X=None, y=None) -> "ManhattanBaseline":
        """No-op. Present only so the baseline matches the model's fit/predict API."""
        return self

    def predict(self, X) -> np.ndarray:
        """Return the ``manhattan_to_goal`` column straight through as the estimate."""
        X = np.asarray(X, dtype=float)
        return X[:, self._col]
