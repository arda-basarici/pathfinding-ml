"""Tests for the maze layer: Grid mechanics and the two generators.

The connectivity/solvability checks use a *local* BFS written independently of the
generator's own ``_reachable`` — you don't validate code with the very function it
ships, or a shared bug passes both sides silently.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from pathfinding.maze.generator import (
    make_mazes,
    random_obstacles,
    structured_maze,
)
from pathfinding.maze.grid import Grid


# --------------------------------------------------------------------------- #
# Independent reachability helper (do not import the generator's _reachable).
# --------------------------------------------------------------------------- #
def reachable_cells(grid: Grid, start) -> set:
    """Every passable cell reachable from ``start`` (independent BFS)."""
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for neighbour in grid.neighbors(cell):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return seen


# --------------------------------------------------------------------------- #
# Grid mechanics — hand-built 3x3 with the centre blocked.
# --------------------------------------------------------------------------- #
def make_grid_with_blocked_center() -> Grid:
    return Grid(height=3, width=3, blocked=frozenset({(1, 1)}))


def test_in_bounds():
    grid = make_grid_with_blocked_center()
    assert grid.in_bounds((0, 0))
    assert grid.in_bounds((2, 2))
    assert not grid.in_bounds((-1, 0))
    assert not grid.in_bounds((3, 0))
    assert not grid.in_bounds((0, 3))


def test_passable():
    grid = make_grid_with_blocked_center()
    assert grid.passable((0, 0))
    assert not grid.passable((1, 1))      # blocked
    assert not grid.passable((5, 5))      # out of bounds is not passable


def test_neighbors_skip_blocked_and_oob():
    grid = make_grid_with_blocked_center()
    # Corner (0,0): in-bounds neighbours are (1,0) and (0,1), both passable.
    assert set(grid.neighbors((0, 0))) == {(1, 0), (0, 1)}
    # Edge centre (0,1): down is the blocked centre, so only (0,0) and (0,2).
    assert set(grid.neighbors((0, 1))) == {(0, 0), (0, 2)}


def test_step_cost_is_unit():
    assert Grid.step_cost((0, 0), (0, 1)) == 1.0


# --------------------------------------------------------------------------- #
# random_obstacles — deterministic at the density extremes.
# --------------------------------------------------------------------------- #
def test_random_obstacles_dimensions_and_extremes():
    rng = np.random.default_rng(0)
    empty = random_obstacles(10, 12, 0.0, rng)
    assert empty.height == 10 and empty.width == 12
    assert empty.blocked == frozenset()

    full = random_obstacles(8, 8, 1.0, rng)
    assert len(full.blocked) == 8 * 8       # every cell blocked


def test_random_obstacles_density_is_in_range():
    rng = np.random.default_rng(1)
    grid = random_obstacles(100, 100, 0.3, rng)
    fraction = len(grid.blocked) / (100 * 100)
    assert 0.25 < fraction < 0.35           # ~0.3, loose bounds for randomness


# --------------------------------------------------------------------------- #
# structured_maze — perfect maze, so fully connected.
# --------------------------------------------------------------------------- #
def test_structured_maze_is_fully_connected():
    rng = np.random.default_rng(2)
    grid = structured_maze(21, 31, rng)
    passable = {
        (r, c)
        for r in range(grid.height)
        for c in range(grid.width)
        if grid.passable((r, c))
    }
    # Every passable cell must be reachable from any one of them.
    some_cell = next(iter(passable))
    assert reachable_cells(grid, some_cell) == passable


def test_structured_maze_forces_odd_dimensions():
    rng = np.random.default_rng(3)
    grid = structured_maze(20, 20, rng)     # even -> coerced to 19
    assert grid.height == 19 and grid.width == 19


# --------------------------------------------------------------------------- #
# make_mazes — solvable instances, valid endpoints, sizes in range.
# --------------------------------------------------------------------------- #
def test_make_mazes_are_solvable_and_well_formed():
    rng = np.random.default_rng(42)
    mazes = make_mazes(30, rng, size_range=(15, 45), density_range=(0.20, 0.35))

    assert len(mazes) == 30
    for maze in mazes:
        # endpoints distinct, in bounds, passable
        assert maze.start != maze.goal
        assert maze.grid.passable(maze.start)
        assert maze.grid.passable(maze.goal)
        # goal genuinely reachable from start
        assert maze.goal in reachable_cells(maze.grid, maze.start)
        # sizes within the requested range (odd coercion stays inside [15, 45])
        assert 15 <= maze.grid.height <= 45
        assert 15 <= maze.grid.width <= 45


def test_make_mazes_is_reproducible():
    a = make_mazes(5, np.random.default_rng(7))
    b = make_mazes(5, np.random.default_rng(7))
    assert [m.grid.blocked for m in a] == [m.grid.blocked for m in b]
    assert [(m.start, m.goal) for m in a] == [(m.start, m.goal) for m in b]
