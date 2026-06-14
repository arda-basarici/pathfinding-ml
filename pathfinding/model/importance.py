"""Permutation feature importance — what does the model actually rely on?

``HistGradientBoostingRegressor`` doesn't expose ``feature_importances_``, so we use
*permutation* importance: shuffle one feature column and measure how much the model's
error grows. This is model-agnostic and, unlike impurity-based importance, measures a
feature's *actual predictive contribution* (no bias toward high-cardinality features).

Reported in MAE units on held-out cells: "scrambling this feature costs the model this
many steps of error." Computed on a subsample for speed.
"""

from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance


def feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    sample_size: int = 20000,
    seed: int = 0,
) -> list[tuple[str, float, float]]:
    """Permutation importance per feature, sorted high to low.

    Returns ``(name, importance_mean, importance_std)`` tuples, in MAE units (how much
    the held-out MAE worsens when that feature is shuffled).
    """
    rng = np.random.default_rng(seed)
    if sample_size and len(y) > sample_size:
        idx = rng.choice(len(y), size=sample_size, replace=False)
        X, y = X[idx], y[idx]

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_absolute_error",
    )
    order = np.argsort(result.importances_mean)[::-1]
    return [
        (feature_names[i], float(result.importances_mean[i]), float(result.importances_std[i]))
        for i in order
    ]
