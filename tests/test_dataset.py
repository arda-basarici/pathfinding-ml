"""Tests for dataset assembly — especially the whole-maze holdout (the leakage guard).

The integrity check (total rows == total reachable cells) confirms assembly neither
drops nor duplicates cells; the split checks confirm no maze straddles train and test.
"""

from __future__ import annotations

import numpy as np

from pathfinding.data.dataset import assemble, split_mazes_by_maze
from pathfinding.data.features import FEATURE_NAMES
from pathfinding.data.labels import true_cost_to_go
from pathfinding.maze.generator import make_mazes


def test_split_is_disjoint_and_covers_all():
    train, test = split_mazes_by_maze(20, test_fraction=0.25, rng=np.random.default_rng(0))
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(range(20))
    assert len(test) == 5                       # round(0.25 * 20)


def test_split_is_reproducible():
    a = split_mazes_by_maze(20, 0.25, np.random.default_rng(1))
    b = split_mazes_by_maze(20, 0.25, np.random.default_rng(1))
    assert a == b


def test_assemble_shapes_and_whole_maze_holdout():
    rng = np.random.default_rng(123)
    mazes = make_mazes(12, rng)
    data = assemble(mazes, test_fraction=0.25, rng=rng)

    # Feature width matches the canonical feature list.
    assert data.X_train.shape[1] == len(FEATURE_NAMES)
    assert data.X_test.shape[1] == len(FEATURE_NAMES)
    # X and y row counts agree.
    assert data.X_train.shape[0] == data.y_train.shape[0]
    assert data.X_test.shape[0] == data.y_test.shape[0]
    # Whole-maze holdout: no maze in both splits, all mazes accounted for.
    assert set(data.train_maze_ids).isdisjoint(data.test_maze_ids)
    assert set(data.train_maze_ids) | set(data.test_maze_ids) == set(range(12))


def test_row_count_equals_total_reachable_cells():
    rng = np.random.default_rng(7)
    mazes = make_mazes(12, rng)
    # Re-derive the split deterministically to count expected rows per side.
    data = assemble(mazes, test_fraction=0.25, rng=np.random.default_rng(7))

    expected_train = sum(
        len(true_cost_to_go(mazes[i].grid, mazes[i].goal)) for i in data.train_maze_ids
    )
    expected_test = sum(
        len(true_cost_to_go(mazes[i].grid, mazes[i].goal)) for i in data.test_maze_ids
    )
    assert data.y_train.shape[0] == expected_train
    assert data.y_test.shape[0] == expected_test


def test_labels_are_nonnegative_and_finite():
    rng = np.random.default_rng(99)
    data = assemble(make_mazes(8, rng), test_fraction=0.25, rng=rng)
    assert np.all(np.isfinite(data.y_train)) and np.all(data.y_train >= 0)
    assert np.all(np.isfinite(data.y_test)) and np.all(data.y_test >= 0)
