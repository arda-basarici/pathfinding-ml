"""Grid maze data structure.

A maze is a rectangular grid of cells. Each cell is either passable or blocked.
Movement is 4-connected (up/down/left/right), unit step cost. This module owns the
*representation* only — generation lives in ``generator.py``, search in
``pathfinding.search``. (Deliberately no reachability/search logic here: a data
structure shouldn't know how to explore itself.)

Coordinate convention: a cell is ``(row, col)``, 0-indexed, row 0 at the top.
"""

from __future__ import annotations

from dataclasses import dataclass

Cell = tuple[int, int]

# 4-connected moves: up, down, left, right.
_MOVES: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Grid:
    """An immutable rectangular grid maze.

    Attributes:
        height: number of rows.
        width: number of columns.
        blocked: set of blocked (impassable) cells.

    Frozen so a maze can be hashed / used as a dict key and cannot be mutated out
    from under a search running on it.
    """

    height: int
    width: int
    blocked: frozenset[Cell]

    def in_bounds(self, cell: Cell) -> bool:
        """True if ``cell`` lies within the grid bounds."""
        row, col = cell
        return 0 <= row < self.height and 0 <= col < self.width

    def passable(self, cell: Cell) -> bool:
        """True if ``cell`` is in bounds and not blocked."""
        return self.in_bounds(cell) and cell not in self.blocked

    def neighbors(self, cell: Cell) -> list[Cell]:
        """Passable 4-connected neighbours of ``cell``."""
        row, col = cell
        result = []
        for d_row, d_col in _MOVES:
            candidate = (row + d_row, col + d_col)
            if self.passable(candidate):
                result.append(candidate)
        return result

    @staticmethod
    def step_cost(a: Cell, b: Cell) -> float:
        """Cost of moving between adjacent cells ``a`` and ``b`` (unit grid: 1.0).

        ``a`` and ``b`` are unused for a uniform-cost grid, but the signature leaves
        room for weighted terrain later without touching callers.
        """
        return 1.0


@dataclass(frozen=True)
class Maze:
    """A solvable problem instance: a grid plus a start and goal.

    Separated from ``Grid`` on purpose — the grid is the static map; start/goal are
    the *question* asked of it. Generators return these, already verified solvable.
    """

    grid: Grid
    start: Cell
    goal: Cell
