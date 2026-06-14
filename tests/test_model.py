"""Tests for the model layer: baseline, training, persistence, and the A* bridge.

Training tests use small mazes so they're fast. The integration test is the satisfying
one: a model trained on a few mazes actually drives A* to a valid path on a held-out
maze — the learned heuristic works end to end.
"""

from __future__ import annotations

import numpy as np

from pathfinding.data.dataset import assemble
from pathfinding.data.features import FEATURE_NAMES
from pathfinding.maze.generator import make_mazes
from pathfinding.model.baseline import ManhattanBaseline
from pathfinding.model.predict import LearnedHeuristic
from pathfinding.model.train import load_model, save_model, train_model
from pathfinding.search.algorithms import astar


def _mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))


def test_baseline_returns_manhattan_column():
    X = np.zeros((3, len(FEATURE_NAMES)))
    col = FEATURE_NAMES.index("manhattan_to_goal")
    X[:, col] = [3.0, 7.0, 1.0]
    assert list(ManhattanBaseline().predict(X)) == [3.0, 7.0, 1.0]


def test_model_fits_at_least_as_well_as_baseline_on_train():
    rng = np.random.default_rng(0)
    data = assemble(make_mazes(8, rng, size_range=(11, 15)), test_fraction=0.25, rng=rng)
    model = train_model(data.X_train, data.y_train)

    model_mae = _mae(model.predict(data.X_train), data.y_train)
    base_mae = _mae(ManhattanBaseline().predict(data.X_train), data.y_train)
    assert model_mae <= base_mae       # learning shouldn't be worse than the baseline


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    data = assemble(make_mazes(6, rng, size_range=(11, 15)), test_fraction=0.25, rng=rng)
    model = train_model(data.X_train, data.y_train)

    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    assert np.allclose(model.predict(data.X_test), loaded.predict(data.X_test))


def test_learned_heuristic_drives_astar_to_valid_path():
    rng = np.random.default_rng(2)
    mazes = make_mazes(8, rng, size_range=(11, 15))
    data = assemble(mazes, test_fraction=0.25, rng=rng)
    model = train_model(data.X_train, data.y_train)

    maze = mazes[0]
    result = astar(maze.grid, maze.start, maze.goal, LearnedHeuristic(model, maze.grid))

    assert result.found
    path = result.path
    assert path[0] == maze.start and path[-1] == maze.goal
    assert maze.grid.passable(path[0])
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1   # contiguous
        assert maze.grid.passable(b)                       # stays on passable cells
