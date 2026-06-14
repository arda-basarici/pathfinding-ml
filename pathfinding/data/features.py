"""Per-cell features for predicting cost-to-go — each with a hypothesis.

A feature earns its place only if there's a reason to think it carries signal about
how far a cell really is from the goal. Stated hypotheses (to confirm or kill later):

    manhattan_to_goal   : the admissible baseline estimate. Strong, by construction.
    euclidean_to_goal   : straight-line distance; differs from Manhattan near diagonals.
    row_delta/col_delta : signed offsets to the goal; let the model learn directional
                          asymmetry (e.g. walls that bite more vertically than horizontally).
    local_obstacle_density : impassable fraction in a window around the cell. Hypothesis:
                          dense / edge-y surroundings force detours, so true cost > Manhattan.
    blocked_neighbors   : how boxed-in the cell is (0..4). Hypothesis: dead-end-ish cells
                          cost more than their distance suggests.

IMPORTANT: features must be computable from the maze + cell + goal ALONE — never from
the label (true cost-to-go). Letting the answer leak in as a feature is exactly the
leakage trap this project is about. (Out-of-bounds cells count as impassable, so border
cells legitimately read as more 'enclosed'.)
"""

from __future__ import annotations

from ..maze.grid import Cell, Grid

# Canonical feature order (keeps train/predict aligned).
FEATURE_NAMES: list[str] = [
    "manhattan_to_goal",
    "euclidean_to_goal",
    "row_delta",
    "col_delta",
    "local_obstacle_density",
    "blocked_neighbors",
]

_ORTHO: tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def extract_features(
    grid: Grid,
    cell: Cell,
    goal: Cell,
    window: int = 2,
) -> dict[str, float]:
    """Feature dict for one ``cell`` (keys == FEATURE_NAMES). No label information."""
    row, col = cell
    d_row = goal[0] - row
    d_col = goal[1] - col

    # Local obstacle density over a (2*window+1)^2 box; OOB counts as impassable.
    total = 0
    impassable = 0
    for d_r in range(-window, window + 1):
        for d_c in range(-window, window + 1):
            total += 1
            if not grid.passable((row + d_r, col + d_c)):
                impassable += 1
    density = impassable / total

    blocked_neighbors = sum(
        1 for d_r, d_c in _ORTHO if not grid.passable((row + d_r, col + d_c))
    )

    return {
        "manhattan_to_goal": float(abs(d_row) + abs(d_col)),
        "euclidean_to_goal": float((d_row**2 + d_col**2) ** 0.5),
        "row_delta": float(d_row),
        "col_delta": float(d_col),
        "local_obstacle_density": float(density),
        "blocked_neighbors": float(blocked_neighbors),
    }


def feature_vector(grid: Grid, cell: Cell, goal: Cell, window: int = 2) -> list[float]:
    """``extract_features`` flattened into FEATURE_NAMES order, for the model."""
    features = extract_features(grid, cell, goal, window)
    return [features[name] for name in FEATURE_NAMES]
