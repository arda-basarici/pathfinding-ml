"""Tests for the configurable core: feature registry, generator registry,
ExperimentConfig, config-aware baseline, and the prediction-transform hook.

The guiding invariant: the *default* configuration must reproduce the original behavior,
so the baseline stays comparable. New knobs are additive.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathfinding.config import ExperimentConfig
from pathfinding.data.features import (
    DEFAULT_FEATURES,
    feature_vector,
    resolve_feature_names,
)
from pathfinding.maze.generator import (
    GENERATORS,
    make_mazes,
    random_obstacles,
    structured_maze,
)
from pathfinding.maze.grid import Grid
from pathfinding.model.baseline import ManhattanBaseline
from pathfinding.model.predict import LearnedHeuristic


class _Dummy:
    """A stand-in model that predicts a constant, for testing the transform hook."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


# --------------------------------------------------------------------------- #
# Feature registry.
# --------------------------------------------------------------------------- #
def test_default_features_unchanged():
    assert DEFAULT_FEATURES == [
        "manhattan_to_goal",
        "euclidean_to_goal",
        "row_delta",
        "col_delta",
        "local_obstacle_density",
        "blocked_neighbors",
    ]


def test_feature_vector_subset_selection():
    grid = Grid(5, 5, frozenset())
    vec = feature_vector(grid, (0, 0), (3, 4), ["manhattan_to_goal", "row_delta"])
    assert vec == [7.0, 3.0]                     # only those two, in that order


def test_global_obstacle_density_feature():
    grid = Grid(4, 4, frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}))   # 4 of 16 blocked
    assert feature_vector(grid, (2, 2), (3, 3), ["global_obstacle_density"]) == [0.25]


def test_line_of_sight_obstacles_feature():
    grid = Grid(1, 5, frozenset({(0, 2)}))       # wall on the straight line
    vec = feature_vector(grid, (0, 0), (0, 4), ["line_of_sight_obstacles"])
    assert vec == [1.0]


# --------------------------------------------------------------------------- #
# Generator registry — dispatch reproduces the direct call exactly.
# --------------------------------------------------------------------------- #
def test_generator_registry_structured_matches_direct():
    direct = structured_maze(15, 15, np.random.default_rng(0))
    via = GENERATORS["structured"](15, 15, np.random.default_rng(0))
    assert direct.blocked == via.blocked
    assert (direct.height, direct.width) == (via.height, via.width)


def test_generator_registry_scattered_matches_direct():
    rng_a = np.random.default_rng(1)
    density = float(rng_a.uniform(0.20, 0.35))
    direct = random_obstacles(20, 20, density, rng_a)

    rng_b = np.random.default_rng(1)
    via = GENERATORS["scattered"](20, 20, rng_b, (0.20, 0.35))
    assert direct.blocked == via.blocked


def test_make_mazes_tags_style():
    mazes = make_mazes(20, np.random.default_rng(0))
    assert all(m.style in {"scattered", "structured"} for m in mazes)


# --------------------------------------------------------------------------- #
# ExperimentConfig.
# --------------------------------------------------------------------------- #
def test_experiment_config_defaults_and_isolation():
    c = ExperimentConfig()
    assert c.feature_names == DEFAULT_FEATURES
    assert c.size_range == (15, 45)
    assert "feature_names" in c.to_dict()
    # Mutating a config's list must not leak into the module-level default.
    c.feature_names.append("global_obstacle_density")
    assert "global_obstacle_density" not in DEFAULT_FEATURES


# --------------------------------------------------------------------------- #
# Config-aware baseline.
# --------------------------------------------------------------------------- #
def test_baseline_finds_manhattan_in_custom_set():
    feats = ["euclidean_to_goal", "manhattan_to_goal", "row_delta"]   # manhattan is col 1
    X = np.zeros((2, 3))
    X[:, 1] = [4.0, 9.0]
    assert list(ManhattanBaseline(feats).predict(X)) == [4.0, 9.0]


def test_baseline_requires_manhattan_present():
    with pytest.raises(ValueError):
        ManhattanBaseline(["euclidean_to_goal", "row_delta"])


# --------------------------------------------------------------------------- #
# Prediction-transform hook.
# --------------------------------------------------------------------------- #
def test_transform_default_clamps_negatives():
    grid = Grid(5, 5, frozenset())
    h = LearnedHeuristic(_Dummy(-3.0), grid)          # default transform clamps at 0
    assert h((0, 0), (4, 4)) == 0.0


def test_transform_custom_replaces_default():
    grid = Grid(5, 5, frozenset())
    h = LearnedHeuristic(_Dummy(-3.0), grid, transform=lambda p: float(p) * 2)
    assert h((0, 0), (4, 4)) == -6.0


# --------------------------------------------------------------------------- #
# Feature-set resolution (--features / --add / --drop).
# --------------------------------------------------------------------------- #
def test_resolve_default():
    assert resolve_feature_names() == DEFAULT_FEATURES


def test_resolve_add_and_drop():
    names = resolve_feature_names(add=["global_obstacle_density"], drop=["blocked_neighbors"])
    assert "global_obstacle_density" in names
    assert "blocked_neighbors" not in names
    assert names[0] == "manhattan_to_goal"


def test_resolve_explicit_features_override():
    names = resolve_feature_names(features=["manhattan_to_goal", "global_obstacle_density"])
    assert names == ["manhattan_to_goal", "global_obstacle_density"]


def test_resolve_rejects_unknown_and_missing_manhattan():
    with pytest.raises(ValueError):
        resolve_feature_names(add=["bogus_feature"])
    with pytest.raises(ValueError):
        resolve_feature_names(drop=["manhattan_to_goal"])
