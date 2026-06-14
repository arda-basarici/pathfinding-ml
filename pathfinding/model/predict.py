"""Bridge a trained model into a search Heuristic.

Search wants a callable ``(cell, goal) -> float``. The model wants a feature vector.
``LearnedHeuristic`` adapts one to the other so a learned model can drop straight into
``astar``/``greedy`` exactly where ``manhattan`` would go.

Caveat we will measure, not hide: the learned estimate is not guaranteed admissible,
so A* using it may return a suboptimal path. That is the experiment.
"""

from __future__ import annotations

from ..data.features import feature_vector
from ..maze.grid import Cell, Grid


class LearnedHeuristic:
    """Wrap a trained model as a search heuristic over a fixed maze.

    Bound to one ``grid`` because features depend on the maze; rebuild per maze at
    evaluation time.
    """

    def __init__(self, model, grid: Grid, window: int = 2) -> None:
        self.model = model
        self.grid = grid
        self.window = window

    def __call__(self, cell: Cell, goal: Cell) -> float:
        """Predicted cost-to-go from ``cell`` to ``goal`` (model on extracted features)."""
        raise NotImplementedError  # build step 6
