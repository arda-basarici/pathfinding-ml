"""Axis 1 — is the learned heuristic a *good estimate* of cost-to-go?

Two questions, kept separate because they mean different things:

  Accuracy   : how close are predictions to true cost-to-go? (MAE / RMSE)
  Admissibility : does it ever OVERESTIMATE? An admissible heuristic never does.
                 Overestimation is what breaks A*'s optimality guarantee, so we report
                 the fraction of cells overestimated and by how much — not just average
                 error. A heuristic can have low MAE and still be inadmissible.

This axis is about the estimator in isolation (the *proxy* metric). Whether better
estimates actually buy faster search is Axis 2 (search_benchmark.py).
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


def evaluate(y_true, y_pred, tol: float = 1e-9) -> QualityReport:
    """Accuracy + admissibility summary of predicted vs true cost-to-go.

    ``error = prediction - truth``; positive error means the heuristic *overestimated*
    (the admissibility-violating direction). ``tol`` ignores floating-point dust.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    frac_overestimated = float(np.mean(error > tol))
    max_overestimate = float(max(0.0, error.max())) if error.size else 0.0

    return QualityReport(mae, rmse, frac_overestimated, max_overestimate)
