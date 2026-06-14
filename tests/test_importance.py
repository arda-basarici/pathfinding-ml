"""Tests for permutation feature importance."""

from __future__ import annotations

import numpy as np

from pathfinding.data.dataset import assemble
from pathfinding.data.features import DEFAULT_FEATURES
from pathfinding.maze.generator import make_mazes
from pathfinding.model.importance import feature_importance
from pathfinding.model.train import train_model


def test_feature_importance_covers_all_features_sorted():
    rng = np.random.default_rng(0)
    data = assemble(make_mazes(8, rng, size_range=(11, 15)), test_fraction=0.25, rng=rng)
    model = train_model(data.X_train, data.y_train)

    imp = feature_importance(
        model, data.X_test, data.y_test, DEFAULT_FEATURES, n_repeats=2, sample_size=5000
    )

    # one entry per feature, all accounted for
    assert {name for name, _, _ in imp} == set(DEFAULT_FEATURES)
    assert len(imp) == len(DEFAULT_FEATURES)

    means = [m for _, m, _ in imp]
    assert all(np.isfinite(m) for m in means)
    assert means == sorted(means, reverse=True)     # returned high-to-low
    assert imp[0][1] > 0                            # the top feature genuinely matters
