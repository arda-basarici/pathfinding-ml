"""Tests for the search layer.

The headline test is **optimality**: A* with an admissible heuristic must return a path
of the same cost as Dijkstra (which is optimal by construction). If that ever fails,
either the heuristic isn't admissible or the search is buggy — both are exactly the
errors this project is meant to catch.
"""

from __future__ import annotations

import math

import numpy as np

from pathfinding.maze.generator import make_mazes
from pathfinding.maze.grid import Grid
from pathfinding.search.algorithms import astar, dijkstra, greedy
from pathfinding.search.heuristics import euclidean, manhattan, zero
from pathfinding.search.instrumentation import SearchResult


def assert_valid_path(grid: Grid, result: SearchResult, start, goal) -> None:
    """A found path must start at start, end at goal, be contiguous and passable,
    and have cost equal to its step count (unit grid)."""
    path = result.path
    assert path[0] == start
    assert path[-1] == goal
    for cell in path:
        assert grid.passable(cell)
    for a, b in zip(path, path[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1   # consecutive cells adjacent
    assert result.path_cost == len(path) - 1


# --------------------------------------------------------------------------- #
# Heuristic values.
# --------------------------------------------------------------------------- #
def test_heuristic_values():
    assert zero((0, 0), (5, 5)) == 0.0
    assert manhattan((0, 0), (3, 4)) == 7.0
    assert euclidean((0, 0), (3, 4)) == 5.0


# --------------------------------------------------------------------------- #
# Known small cases.
# --------------------------------------------------------------------------- #
def test_dijkstra_known_shortest_path():
    grid = Grid(1, 5, frozenset())            # a 1x5 corridor
    result = dijkstra(grid, (0, 0), (0, 4))
    assert result.found
    assert result.path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    assert result.path_cost == 4


def test_unreachable_goal():
    # 3x3 with (2,2) walled off behind (1,2) and (2,1).
    grid = Grid(3, 3, frozenset({(1, 2), (2, 1)}))
    result = dijkstra(grid, (0, 0), (2, 2))
    assert not result.found
    assert result.path == []
    assert result.path_cost == math.inf


# --------------------------------------------------------------------------- #
# The core invariants, checked across many generated mazes.
# --------------------------------------------------------------------------- #
def test_astar_manhattan_is_optimal_and_not_more_work():
    rng = np.random.default_rng(123)
    for maze in make_mazes(25, rng):
        d = dijkstra(maze.grid, maze.start, maze.goal)
        a = astar(maze.grid, maze.start, maze.goal, manhattan)

        assert d.found and a.found
        assert a.path_cost == d.path_cost                  # admissible => optimal
        assert a.nodes_expanded <= d.nodes_expanded        # informed => no more work
        assert_valid_path(maze.grid, a, maze.start, maze.goal)
        assert_valid_path(maze.grid, d, maze.start, maze.goal)


def test_astar_with_zero_heuristic_equals_dijkstra():
    rng = np.random.default_rng(7)
    for maze in make_mazes(15, rng):
        d = dijkstra(maze.grid, maze.start, maze.goal)
        a = astar(maze.grid, maze.start, maze.goal, zero)
        assert a.path_cost == d.path_cost
        assert a.nodes_expanded == d.nodes_expanded         # zero h IS dijkstra


def test_greedy_finds_valid_path_never_cheaper_than_optimal():
    rng = np.random.default_rng(99)
    for maze in make_mazes(25, rng):
        d = dijkstra(maze.grid, maze.start, maze.goal)
        g = greedy(maze.grid, maze.start, maze.goal, manhattan)

        assert g.found
        assert_valid_path(maze.grid, g, maze.start, maze.goal)
        assert g.path_cost >= d.path_cost                   # greedy may be suboptimal


def test_euclidean_also_optimal():
    # Euclidean is admissible too, so A* with it must still be optimal.
    rng = np.random.default_rng(5)
    for maze in make_mazes(15, rng):
        d = dijkstra(maze.grid, maze.start, maze.goal)
        a = astar(maze.grid, maze.start, maze.goal, euclidean)
        assert a.path_cost == d.path_cost
