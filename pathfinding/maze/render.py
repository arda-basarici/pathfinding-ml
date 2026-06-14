"""Render a maze as ASCII text — for eyeballing what the generators produce.

Presentation only; kept out of ``Grid``/``Maze`` so the data types stay logic-free.
Legend:  '#' wall   ' ' open   'S' start   'G' goal   '*' path cell

The optional ``path`` overlay is unused for now but is why this lives in the package:
step 2 (search) will print found paths through this same function.
"""

from __future__ import annotations

from collections.abc import Iterable

from .grid import Cell, Maze


def to_ascii(maze: Maze, path: Iterable[Cell] | None = None) -> str:
    """Return ``maze`` as a multi-line string. ``path`` cells are drawn as '*'."""
    grid = maze.grid
    path_set = set(path) if path is not None else set()

    lines = []
    for row in range(grid.height):
        chars = []
        for col in range(grid.width):
            cell = (row, col)
            if cell == maze.start:
                chars.append("S")
            elif cell == maze.goal:
                chars.append("G")
            elif cell in path_set:
                chars.append("*")
            elif cell in grid.blocked:
                chars.append("#")
            else:
                chars.append(" ")
        lines.append("".join(chars))
    return "\n".join(lines)
