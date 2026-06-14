"""The baseline cost-to-go predictor: the classic Manhattan heuristic.

A model only earns its keep if it beats a *fair* baseline. Here the baseline isn't
"predict the mean" — it's the admissible heuristic A* already uses. Beating Manhattan
on prediction error is necessary but not sufficient; what ultimately matters is
whether it expands fewer nodes without wrecking path quality (see evaluation/).

Exposes the same ``predict`` interface as the learned model so evaluation can treat
them interchangeably.
"""

from __future__ import annotations

import numpy as np


class ManhattanBaseline:
    """Predicts cost-to-go as the Manhattan distance feature (no fitting needed)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the manhattan_to_goal column straight through as the estimate."""
        raise NotImplementedError  # build step 6
