"""Maze generators.

Produces ``Maze`` instances (grid + start + goal, verified solvable) for training
and evaluation. Two *unlike* generators on purpose (decision D7) — a heuristic that
only works on one maze style has overfit to that style:

    random_obstacles : cells blocked independently at random. Scattered, open, many
                       short detours. Manhattan distance is usually a decent estimate.
    structured_maze  : a "perfect" corridor maze (recursive backtracker). Long forced
                       detours, so Manhattan distance is often a poor estimate — the
                       harder, more interesting case for a learned heuristic.

Sizes and obstacle density vary per maze (a third axis of variation, on top of the
two styles), so the heuristic must generalise across scales too. All randomness flows
through an explicit ``numpy.random.Generator`` for reproducibility.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .grid import Cell, Grid, Maze


# --------------------------------------------------------------------------- #
# Low-level generators: produce a Grid (the static map only).
# --------------------------------------------------------------------------- #
def random_obstacles(
    height: int,
    width: int,
    obstacle_density: float,
    rng: np.random.Generator,
) -> Grid:
    """Grid where each cell is independently blocked with prob ``obstacle_density``."""
    blocked = {
        (row, col)
        for row in range(height)
        for col in range(width)
        if rng.random() < obstacle_density
    }
    return Grid(height, width, frozenset(blocked))


def structured_maze(height: int, width: int, rng: np.random.Generator) -> Grid:
    """A perfect (fully-connected, loop-free) maze via the recursive backtracker.

    Passages live on odd coordinates, walls on even ones, so dimensions are forced
    odd. The carve guarantees every passage cell is reachable from every other, so
    any two passage cells form a solvable instance.
    """
    # Force odd dimensions so passage/wall coordinates alternate cleanly.
    rows = height if height % 2 == 1 else height - 1
    cols = width if width % 2 == 1 else width - 1

    blocked = {(r, c) for r in range(rows) for c in range(cols)}  # start: all walls
    start = (1, 1)
    blocked.discard(start)
    stack = [start]

    while stack:
        row, col = stack[-1]
        # Unvisited passage neighbours are two steps away and still walls.
        candidates = []
        for d_row, d_col in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            n_row, n_col = row + d_row, col + d_col
            if 0 < n_row < rows and 0 < n_col < cols and (n_row, n_col) in blocked:
                candidates.append((n_row, n_col, d_row, d_col))

        if candidates:
            n_row, n_col, d_row, d_col = candidates[int(rng.integers(len(candidates)))]
            blocked.discard((row + d_row // 2, col + d_col // 2))  # knock out the wall
            blocked.discard((n_row, n_col))  # open the neighbour
            stack.append((n_row, n_col))
        else:
            stack.pop()  # dead end, backtrack

    return Grid(rows, cols, frozenset(blocked))


# --------------------------------------------------------------------------- #
# Solvability + endpoint selection.
# --------------------------------------------------------------------------- #
def _reachable(grid: Grid, start: Cell, goal: Cell) -> bool:
    """True if ``goal`` is reachable from ``start`` (BFS flood fill over passables)."""
    if not (grid.passable(start) and grid.passable(goal)):
        return False
    seen = {start}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for neighbour in grid.neighbors(cell):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return False


def _random_passable(grid: Grid, rng: np.random.Generator) -> Cell:
    """Uniformly sample a passable cell (rejection sampling)."""
    while True:
        cell = (int(rng.integers(grid.height)), int(rng.integers(grid.width)))
        if grid.passable(cell):
            return cell


def _make_solvable(
    grid: Grid,
    rng: np.random.Generator,
    attempts: int,
) -> Maze | None:
    """Pick distinct, reachable start/goal on ``grid``; give up after ``attempts``."""
    for _ in range(attempts):
        start = _random_passable(grid, rng)
        goal = _random_passable(grid, rng)
        if start != goal and _reachable(grid, start, goal):
            return Maze(grid, start, goal)
    return None


# --------------------------------------------------------------------------- #
# Public entry point: a population of mixed, varied, solvable mazes.
# --------------------------------------------------------------------------- #
def make_mazes(
    n: int,
    rng: np.random.Generator,
    size_range: tuple[int, int] = (15, 45),
    density_range: tuple[float, float] = (0.20, 0.35),
    structured_fraction: float = 0.5,
    endpoint_attempts: int = 200,
) -> list[Maze]:
    """Generate ``n`` solvable mazes, mixing both styles with varied size/density.

    Each maze independently: picks a style (``structured_fraction`` of the time the
    corridor maze), a random height and width in ``size_range``, and — for the
    scattered style — a random density in ``density_range``. Unsolvable draws (rare;
    only possible for the scattered style) are discarded and redrawn.
    """
    mazes: list[Maze] = []
    while len(mazes) < n:
        height = int(rng.integers(size_range[0], size_range[1] + 1))
        width = int(rng.integers(size_range[0], size_range[1] + 1))

        if rng.random() < structured_fraction:
            grid = structured_maze(height, width, rng)
        else:
            density = float(rng.uniform(*density_range))
            grid = random_obstacles(height, width, density, rng)

        maze = _make_solvable(grid, rng, endpoint_attempts)
        if maze is not None:
            mazes.append(maze)
    return mazes
