"""Tests for label generation (exact cost-to-go).

The key check cross-validates the BFS labels against Dijkstra: the label for a cell must
equal the search distance from that cell to the goal. Two independent routes to the same
number — if they disagree, one is wrong.
"""

from __future__ import annotations

import numpy as np

from pathfinding.data.labels import true_cost_to_go
from pathfinding.maze.generator import make_mazes
from pathfinding.maze.grid import Grid
from pathfinding.search.algorithms import dijkstra


def test_known_costs_on_corridor():
    grid = Grid(1, 5, frozenset())
    cost = true_cost_to_go(grid, goal=(0, 4))
    assert cost[(0, 4)] == 0.0
    assert cost[(0, 2)] == 2.0
    assert cost[(0, 0)] == 4.0


def test_blocked_goal_returns_empty():
    grid = Grid(3, 3, frozenset({(1, 1)}))
    assert true_cost_to_go(grid, goal=(1, 1)) == {}


def test_unreachable_and_blocked_cells_omitted():
    # (2,2) is walled off behind (1,2) and (2,1); those two are blocked.
    grid = Grid(3, 3, frozenset({(1, 2), (2, 1)}))
    cost = true_cost_to_go(grid, goal=(0, 0))
    assert cost[(0, 0)] == 0.0
    assert (2, 2) not in cost          # unreachable
    assert (1, 2) not in cost          # blocked
    assert (2, 1) not in cost          # blocked


def test_labels_match_dijkstra_distances():
    rng = np.random.default_rng(321)
    for maze in make_mazes(10, rng):
        cost = true_cost_to_go(maze.grid, maze.goal)

        # The start's label must match the search cost start -> goal.
        assert cost[maze.start] == dijkstra(maze.grid, maze.start, maze.goal).path_cost

        # Sample some passable cells and cross-check each against Dijkstra.
        passable = [
            (r, c)
            for r in range(maze.grid.height)
            for c in range(maze.grid.width)
            if maze.grid.passable((r, c))
        ]
        for idx in rng.choice(len(passable), size=min(8, len(passable)), replace=False):
            cell = passable[int(idx)]
            result = dijkstra(maze.grid, cell, maze.goal)
            if result.found:
                assert cost[cell] == result.path_cost
            else:
                assert cell not in cost          # disconnected pocket
