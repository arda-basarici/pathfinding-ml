"""Ground-truth labels: the exact cost-to-go for every cell.

The training target is h*(cell) = true shortest-path cost from ``cell`` to the goal.
We get it *for free and exactly* by running a backward breadth-first search from the
goal: one sweep labels every reachable cell at once. No noise, no approximation — the
rare luxury of perfect labels (the same property that makes the blackjack project
honest).

Why *backward* works in one pass: movement is symmetric on a 4-connected unit grid
(if you can step a->b you can step b->a, same cost), so the distance from the goal to a
cell equals that cell's distance to the goal. BFS layers = distances because every step
costs 1.

This is the quantity the learned heuristic tries to predict, and the quantity the
classic heuristics only *estimate*.
"""

from __future__ import annotations

from collections import deque

from ..maze.grid import Cell, Grid


def true_cost_to_go(grid: Grid, goal: Cell) -> dict[Cell, float]:
    """Exact shortest-path cost from every reachable cell to ``goal``.

    Backward BFS from ``goal`` over passable cells. Unreachable cells are omitted from
    the result (callers must treat 'missing' as unreachable, not as cost 0). If ``goal``
    itself is blocked, returns an empty mapping.
    """
    if not grid.passable(goal):
        return {}

    cost: dict[Cell, float] = {goal: 0.0}
    queue: deque[Cell] = deque([goal])
    while queue:
        cell = queue.popleft()
        for neighbour in grid.neighbors(cell):
            if neighbour not in cost:
                cost[neighbour] = cost[cell] + 1.0
                queue.append(neighbour)
    return cost
