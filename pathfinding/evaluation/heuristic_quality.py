"""Axis 1 — is the learned heuristic a *good estimate* of cost-to-go?

Two questions, kept separate because they mean different things:

  Accuracy   : how close are predictions to true cost-to-go? (MAE / RMSE)
  Admissibility : does it ever OVERESTIMATE? An admissible heuristic never does.
                 Overestimation is what breaks A*'s optimality guarantee, so we report
                 the fraction of cells overestimated and by how much — not just average
                 error. A heuristic can have low MAE and still be inadmissible.

This axis is about the estimator in isolation. Whether better estimates actually buy
faster search is Axis 2 (search_benchmark.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QualityReport:
    mae: float
    rmse: float
    frac_overestimated: float
    max_overestimate: float


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> QualityReport:
    """Accuracy + admissibility summary of predicted vs true cost-to-go."""
    raise NotImplementedError  # build step 7
