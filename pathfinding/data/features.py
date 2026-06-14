"""Per-cell features for predicting cost-to-go — a registry of named functions.

Each feature is a function ``(grid, cell, goal, window) -> float`` registered by name in
``FEATURE_FUNCS``. A run picks a list of feature names; ``feature_vector`` builds the row
from exactly those. This makes feature *ablation* a configuration, not a code edit — and
adding a new feature is a registration, not a refactor.

``DEFAULT_FEATURES`` is the original six and stays the reproducible baseline. New features
are registered but left *out* of the default, so they're opt-in via config.

Every feature must be computable from maze + cell + goal alone — never from the label
(true cost-to-go). That's the leakage rule (out-of-bounds counts as impassable, so border
cells legitimately read as more enclosed).

Hypotheses behind each feature:
    manhattan_to_goal      : the admissible baseline estimate. Strong by construction.
    euclidean_to_goal      : straight-line; differs from Manhattan near diagonals.
    row_delta / col_delta  : signed offsets; let the model learn directional asymmetry.
    local_obstacle_density : impassable fraction in a window. Dense surroundings -> detours.
    blocked_neighbors      : how boxed-in (0..4). Dead-end-ish cells cost more.
    global_obstacle_density: whole-maze density (one value per maze). Calibrates scale.
    line_of_sight_obstacles: blocked cells on the straight line to the goal. Senses
                             structure *between* cell and goal without solving the maze.
"""

from __future__ import annotations

from typing import Callable

from ..maze.grid import Cell, Grid

FeatureFn = Callable[[Grid, Cell, Cell, int], float]

_ORTHO: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# Feature functions. Uniform signature (grid, cell, goal, window); features that
# don't need the window simply ignore it.
# --------------------------------------------------------------------------- #
def _manhattan_to_goal(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    return float(abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))


def _euclidean_to_goal(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    return float(((cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2) ** 0.5)


def _row_delta(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    return float(goal[0] - cell[0])


def _col_delta(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    return float(goal[1] - cell[1])


def _local_obstacle_density(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    row, col = cell
    total = 0
    impassable = 0
    for d_r in range(-window, window + 1):
        for d_c in range(-window, window + 1):
            total += 1
            if not grid.passable((row + d_r, col + d_c)):
                impassable += 1
    return impassable / total


def _blocked_neighbors(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    row, col = cell
    return float(
        sum(1 for d_r, d_c in _ORTHO if not grid.passable((row + d_r, col + d_c)))
    )


def _global_obstacle_density(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    return len(grid.blocked) / (grid.height * grid.width)


def _line_cells(a: Cell, b: Cell) -> list[Cell]:
    """Cells on the straight line from a to b (Bresenham), inclusive of endpoints."""
    (r0, c0), (r1, c1) = a, b
    d_r, d_c = abs(r1 - r0), abs(c1 - c0)
    s_r = 1 if r0 < r1 else -1
    s_c = 1 if c0 < c1 else -1
    err = d_r - d_c
    cells = []
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 > -d_c:
            err -= d_c
            r += s_r
        if e2 < d_r:
            err += d_r
            c += s_c


def _line_of_sight_obstacles(grid: Grid, cell: Cell, goal: Cell, window: int) -> float:
    # cell and goal are in bounds, so every Bresenham line cell between them is too —
    # membership in grid.blocked is enough here (no need for the OOB-aware grid.passable).
    interior = _line_cells(cell, goal)[1:-1]   # exclude endpoints
    return float(sum(1 for p in interior if p in grid.blocked))


FEATURE_FUNCS: dict[str, FeatureFn] = {
    "manhattan_to_goal": _manhattan_to_goal,
    "euclidean_to_goal": _euclidean_to_goal,
    "row_delta": _row_delta,
    "col_delta": _col_delta,
    "local_obstacle_density": _local_obstacle_density,
    "blocked_neighbors": _blocked_neighbors,
    # registered but OFF by default — opt in via config:
    "global_obstacle_density": _global_obstacle_density,
    "line_of_sight_obstacles": _line_of_sight_obstacles,
}

# The original six — the reproducible baseline feature set.
DEFAULT_FEATURES: list[str] = [
    "manhattan_to_goal",
    "euclidean_to_goal",
    "row_delta",
    "col_delta",
    "local_obstacle_density",
    "blocked_neighbors",
]

# Back-compat alias (some callers/tests import FEATURE_NAMES).
FEATURE_NAMES = DEFAULT_FEATURES


def feature_vector(
    grid: Grid,
    cell: Cell,
    goal: Cell,
    feature_names: list[str] | None = None,
    window: int = 2,
) -> list[float]:
    """Build a feature row for ``cell`` from the selected features (default: the six)."""
    names = DEFAULT_FEATURES if feature_names is None else feature_names
    return [FEATURE_FUNCS[name](grid, cell, goal, window) for name in names]


def extract_features(
    grid: Grid,
    cell: Cell,
    goal: Cell,
    window: int = 2,
) -> dict[str, float]:
    """The default six features as a name->value dict (for inspection/tests)."""
    return {name: FEATURE_FUNCS[name](grid, cell, goal, window) for name in DEFAULT_FEATURES}


def resolve_feature_names(
    features: list[str] | None = None,
    add: list[str] | None = None,
    drop: list[str] | None = None,
) -> list[str]:
    """Resolve a feature set for a run.

    ``features`` (if given) is the explicit full list; otherwise start from
    ``DEFAULT_FEATURES``, remove ``drop``, then append ``add``. Validates names and
    requires ``manhattan_to_goal`` (it anchors the Manhattan baseline).
    """
    if features:
        names = list(features)
    else:
        names = list(DEFAULT_FEATURES)
        if drop:
            dropping = set(drop)
            names = [f for f in names if f not in dropping]
        if add:
            for f in add:
                if f not in names:
                    names.append(f)

    unknown = [f for f in names if f not in FEATURE_FUNCS]
    if unknown:
        raise ValueError(f"unknown features: {unknown}; available: {sorted(FEATURE_FUNCS)}")
    if "manhattan_to_goal" not in names:
        raise ValueError("manhattan_to_goal must remain (it anchors the Manhattan baseline)")
    return names
