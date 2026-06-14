"""Assemble the (features, label) dataset — with the leakage guard built in.

The one rule that defines this module: **split by whole maze, never by cell.** Cells
from the same maze share structure, so putting some cells of a maze in train and
others in test lets the test answers leak through their neighbours. Holding out whole
mazes is the honest split, and ``assemble`` asserts it rather than trusting it.

Pipeline: mazes -> per-cell (features, true cost-to-go) rows -> X/y arrays, split so
that no maze contributes to both train and test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..maze.grid import Maze
from .features import DEFAULT_FEATURES, feature_vector
from .labels import true_cost_to_go


@dataclass
class Dataset:
    """Train/test arrays plus the maze provenance used to verify the split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_maze_ids: list[int]
    test_maze_ids: list[int]


def split_mazes_by_maze(
    n_mazes: int,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Partition maze indices into disjoint train/test id lists (whole-maze holdout)."""
    order = rng.permutation(n_mazes)
    n_test = int(round(test_fraction * n_mazes))
    test_ids = sorted(int(i) for i in order[:n_test])
    train_ids = sorted(int(i) for i in order[n_test:])
    return train_ids, test_ids


def _rows_for(
    mazes: list[Maze],
    ids: list[int],
    feature_names: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) for the given maze ids: one row per reachable cell."""
    X: list[list[float]] = []
    y: list[float] = []
    for i in ids:
        maze = mazes[i]
        costs = true_cost_to_go(maze.grid, maze.goal)   # reachable cells only
        for cell, cost in costs.items():
            X.append(feature_vector(maze.grid, cell, maze.goal, feature_names, window))
            y.append(cost)

    if not X:   # keep a 2-D shape even when empty, so models see the right width
        return (
            np.empty((0, len(feature_names)), dtype=float),
            np.empty((0,), dtype=float),
        )
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def assemble(
    mazes: list[Maze],
    test_fraction: float,
    rng: np.random.Generator,
    feature_names: list[str] | None = None,
    window: int = 2,
) -> Dataset:
    """Build the full dataset and ASSERT the split is a clean whole-maze holdout.

    ``feature_names`` selects which features to compute (default: the six). The
    assertions are the codified leakage guard: if a refactor ever reintroduces
    cell-level mixing or drops a maze, these fail loudly.
    """
    names = DEFAULT_FEATURES if feature_names is None else feature_names
    train_ids, test_ids = split_mazes_by_maze(len(mazes), test_fraction, rng)

    # Leakage guard: no maze in both splits, and together they cover every maze.
    assert set(train_ids).isdisjoint(test_ids), "maze leaked across train/test"
    assert set(train_ids) | set(test_ids) == set(range(len(mazes))), "mazes dropped"

    X_train, y_train = _rows_for(mazes, train_ids, names, window)
    X_test, y_test = _rows_for(mazes, test_ids, names, window)
    return Dataset(X_train, y_train, X_test, y_test, train_ids, test_ids)
