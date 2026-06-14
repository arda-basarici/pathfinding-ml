"""Maze generators.

Produces ``Grid`` instances for training and evaluation. We keep more than one
generator on purpose: a learned heuristic that only works on one maze *style* has
overfit to that style, and mixing generators is part of the honesty story.

All generators take an explicit ``numpy.random.Generator`` so runs are reproducible.
"""

from __future__ import annotations

import numpy as np

from .grid import Cell, Grid


def random_obstacles(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
) -> Grid:
    """A grid where each cell is independently blocked with prob ``obstacle_density``.

    Simple, fast, and produces scattered obstacles (not corridors). Good first
    generator; not very "maze-like".
    """
    raise NotImplementedError  # build step 1


def recursive_division(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> Grid:
    """A structured maze with corridors and walls via recursive division.

    Produces longer detours than scattered obstacles, which stresses the heuristic
    harder (Manhattan distance is a worse estimate when walls force big detours).
    """
    raise NotImplementedError  # build step 1


def make_mazes(
    n: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    obstacle_density: float = 0.3,
) -> list[Grid]:
    """Generate ``n`` mazes (the population we later split by whole maze).

    Each returned maze should be solvable from its start to its goal; callers decide
    start/goal (commonly opposite corners). Unsolvable draws are rejected and redrawn.
    """
    raise NotImplementedError  # build step 1
