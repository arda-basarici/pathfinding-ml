"""Axis 2 — does the learned heuristic make *search* better, and at what cost?

This is the deliverable. For each test maze, run each (algorithm, heuristic) combo and
record two numbers:

    nodes_expanded   : the work done           (the speed axis)
    path_cost        : length of the path found (the quality axis)

Quality is reported as the optimality gap vs the true optimum (Dijkstra / admissible
A*). The expected and honest result: the learned heuristic expands fewer nodes but
sometimes returns longer paths. One number can't capture that — the output is the
tradeoff (a frontier), never a single 'score'.

The combos compared (decision D1/D7): Dijkstra as the uninformed anchor, then A* and
Greedy each with the classic Manhattan heuristic and with the learned one.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..maze.grid import Maze
from ..model.predict import precompute_learned_heuristic
from ..search.algorithms import astar, dijkstra, greedy
from ..search.heuristics import manhattan


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
    maze_style: str | None = None

    @property
    def optimality_gap(self) -> float:
        """``path_cost / optimal_cost - 1`` — 0.0 means an optimal path."""
        if self.optimal_cost == 0:
            return 0.0
        return self.path_cost / self.optimal_cost - 1.0


def run_benchmark(mazes: list[Maze], model, window: int = 2) -> list[BenchmarkRow]:
    """Run every algorithm x heuristic over every maze; return tidy rows.

    ``model`` is the trained cost-to-go regressor; it's wrapped per maze in a
    ``LearnedHeuristic`` (features depend on the maze). The Dijkstra run doubles as the
    optimum each maze's rows are scored against.
    """
    rows: list[BenchmarkRow] = []
    for maze_id, maze in enumerate(mazes):
        grid, start, goal = maze.grid, maze.start, maze.goal
        learned = precompute_learned_heuristic(model, grid, goal, window)

        optimum = dijkstra(grid, start, goal)
        optimal_cost = optimum.path_cost

        runs = [
            ("dijkstra", "none", optimum),
            ("astar", "manhattan", astar(grid, start, goal, manhattan)),
            ("astar", "learned", astar(grid, start, goal, learned)),
            ("greedy", "manhattan", greedy(grid, start, goal, manhattan)),
            ("greedy", "learned", greedy(grid, start, goal, learned)),
        ]
        for algorithm, heuristic_name, result in runs:
            rows.append(
                BenchmarkRow(
                    maze_id=maze_id,
                    algorithm=algorithm,
                    heuristic=heuristic_name,
                    nodes_expanded=result.nodes_expanded,
                    path_cost=result.path_cost,
                    optimal_cost=optimal_cost,
                    found=result.found,
                    maze_style=maze.style,
                )
            )
    return rows


def summarize(rows: list[BenchmarkRow]) -> dict[tuple[str, str], dict[str, float]]:
    """Aggregate rows by (algorithm, heuristic): mean work, mean gap, found-rate.

    The optimality gap is averaged over *found* rows only (it's undefined otherwise).
    """
    groups: dict[tuple[str, str], list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        groups[(row.algorithm, row.heuristic)].append(row)

    summary: dict[tuple[str, str], dict[str, float]] = {}
    for key, group in groups.items():
        found = [r for r in group if r.found]
        summary[key] = {
            "mean_nodes_expanded": float(np.mean([r.nodes_expanded for r in group])),
            "mean_optimality_gap": (
                float(np.mean([r.optimality_gap for r in found])) if found else float("nan")
            ),
            "found_rate": len(found) / len(group),
            "n": float(len(group)),
        }
    return summary
