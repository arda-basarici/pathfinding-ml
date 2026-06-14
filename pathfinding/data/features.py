"""Per-cell features for predicting cost-to-go — each with a hypothesis.

A feature earns its place only if there's a reason to think it carries signal about
how far a cell really is from the goal. Stated hypotheses (to confirm or kill later):

    manhattan_to_goal   : the admissible baseline estimate. Strong, by construction.
    euclidean_to_goal   : straight-line distance; differs from Manhattan near diagonals.
    row_delta/col_delta : signed offsets; let the model learn directional asymmetry
                          (e.g. walls that bite more vertically than horizontally).
    local_obstacle_density : blocked fraction in a window around the cell. Hypothesis:
                          dense surroundings force detours, so true cost > Manhattan.
    blocked_neighbors   : how boxed-in the cell is (0..4). Hypothesis: dead-end-ish
                          cells cost more than their distance suggests.

IMPORTANT: features must be computable from the maze + cell + goal ALONE — never from
the label (true cost-to-go). Letting the answer leak in as a feature is exactly the
leakage trap this project is about.
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


def extract_features(
    grid: Grid,
    cell: Cell,
    goal: Cell,
    window: int = 2,
) -> dict[str, float]:
    """Feature dict for one ``cell`` (keys == FEATURE_NAMES). No label information."""
    raise NotImplementedError  # build step 4


def feature_vector(grid: Grid, cell: Cell, goal: Cell, window: int = 2) -> list[float]:
    """``extract_features`` flattened into FEATURE_NAMES order, for the model."""
    raise NotImplementedError  # build step 4
