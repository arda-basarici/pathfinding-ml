"""Heuristics: the estimate of remaining cost h(cell, goal).

A heuristic is any callable ``(cell, goal) -> float``. The classic ones below are
*admissible* (never overestimate true cost), which is what guarantees A* optimality.
The learned heuristic (see ``pathfinding.model.predict``) is the experiment: it may
be more accurate on average but is **not** guaranteed admissible — that is precisely
the tradeoff we measure.

    zero      -> turns A* into Dijkstra (uninformed)
    manhattan -> admissible on a 4-connected unit grid (the baseline to beat)
    euclidean -> admissible but looser on a 4-connected grid
"""

from __future__ import annotations

from typing import Callable

from ..maze.grid import Cell

Heuristic = Callable[[Cell, Cell], float]


def zero(cell: Cell, goal: Cell) -> float:
    """h = 0 everywhere. A* with this heuristic *is* Dijkstra."""
    return 0.0


def manhattan(cell: Cell, goal: Cell) -> float:
    """|dr| + |dc|. The incumbent baseline; admissible on a 4-connected unit grid.

    Admissible because every step changes row or col by exactly 1, so the true cost is
    at least the sum of the row and column gaps — Manhattan never overestimates.
    """
    return float(abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))


def euclidean(cell: Cell, goal: Cell) -> float:
    """Straight-line distance. Admissible but an underestimate on a 4-connected grid.

    Always <= Manhattan, so also admissible — but looser, so it guides A* less well
    (it under-promises more), which usually means more nodes expanded than Manhattan.
    """
    return float(((cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2) ** 0.5)
