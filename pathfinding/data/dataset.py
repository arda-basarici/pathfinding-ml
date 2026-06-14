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

from ..maze.grid import Cell, Grid


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
    raise NotImplementedError  # build step 5


def assemble(
    mazes: list[Grid],
    goals: list[Cell],
    test_fraction: float,
    rng: np.random.Generator,
) -> Dataset:
    """Build the full dataset and ASSERT train/test maze ids are disjoint.

    The disjointness assertion is the codified leakage guard — if a refactor ever
    reintroduces cell-level mixing, this fails loudly.
    """
    raise NotImplementedError  # build step 5
