"""Ground-truth labels: the exact cost-to-go for every cell.

The training target is h*(cell) = true shortest-path cost from ``cell`` to the goal.
We get it *for free and exactly* by running a backward uniform-cost search (BFS on a
unit grid) from the goal: one sweep labels every reachable cell at once. No noise, no
approximation — the rare luxury of perfect labels, the same property that makes the
blackjack project honest.

This is the quantity the learned heuristic tries to predict, and the quantity the
classic heuristics only *estimate*.
"""

from __future__ import annotations

from ..maze.grid import Cell, Grid


def true_cost_to_go(grid: Grid, goal: Cell) -> dict[Cell, float]:
    """Exact shortest-path cost from every reachable cell to ``goal``.

    Implemented as a backward BFS/uniform-cost sweep from ``goal``. Unreachable cells
    are omitted from the returned mapping (callers must not treat 'missing' as 0).
    """
    raise NotImplementedError  # build step 3
