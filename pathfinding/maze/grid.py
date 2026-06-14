"""Grid maze data structure.

A maze is a rectangular grid of cells. Each cell is either passable or blocked.
Movement is 4-connected (up/down/left/right), unit step cost. This module owns the
*representation* only — generation lives in ``generator.py``, search in
``pathfinding.search``.

Coordinate convention: a cell is ``(row, col)``, 0-indexed, row 0 at the top.
"""

from __future__ import annotations

from dataclasses import dataclass

Cell = tuple[int, int]


@dataclass(frozen=True)
class Grid:
    """An immutable rectangular grid maze.

    Attributes:
        height: number of rows.
        width: number of columns.
        blocked: set of blocked cells (impassable).

    A frozen dataclass so a maze can be hashed / used as a dict key and so a maze
    cannot be mutated out from under a search that is running on it.
    """

    height: int
    width: int
    blocked: frozenset[Cell]

    def in_bounds(self, cell: Cell) -> bool:
        """True if ``cell`` lies within the grid bounds."""
        raise NotImplementedError  # build step 1

    def passable(self, cell: Cell) -> bool:
        """True if ``cell`` is in bounds and not blocked."""
        raise NotImplementedError  # build step 1

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Passable 4-connected neighbours of ``cell``."""
        raise NotImplementedError  # build step 1

    @staticmethod
    def step_cost(a: Cell, b: Cell) -> float:
        """Cost of moving between adjacent cells ``a`` and ``b`` (unit grid: 1.0)."""
        raise NotImplementedError  # build step 1
