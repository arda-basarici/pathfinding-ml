"""Axis 2 — does the learned heuristic make *search* better, and at what cost?

This is the deliverable. For each test maze, run each (algorithm, heuristic) combo and
record two numbers:

    nodes_expanded   : the work done           (the speed axis)
    path_cost        : length of the path found (the quality axis)

Quality is reported as the optimality gap vs the true optimum (Dijkstra / admissible
A*). The expected and honest result: the learned heuristic expands fewer nodes but
sometimes returns longer paths. One number can't capture that — the output is the
tradeoff (a frontier), never a single 'score'.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..maze.grid import Cell, Grid


@dataclass
class BenchmarkRow:
    """One (maze, algorithm, heuristic) measurement."""

    maze_id: int
    algorithm: str
    heuristic: str
    nodes_expanded: int
    path_cost: float
    optimal_cost: float
    found: bool

    @property
    def optimality_gap(self) -> float:
        """``path_cost / optimal_cost - 1`` — 0.0 means optimal path."""
        raise NotImplementedError  # build step 7


def run_benchmark(
    mazes: list[Grid],
    starts: list[Cell],
    goals: list[Cell],
    model,
) -> list[BenchmarkRow]:
    """Run every algorithm x heuristic over every test maze; return tidy rows."""
    raise NotImplementedError  # build step 7
