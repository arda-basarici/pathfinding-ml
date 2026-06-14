"""Bridge a trained model into a search Heuristic.

Search wants a callable ``(cell, goal) -> float``. The model wants a feature vector.
``LearnedHeuristic`` adapts one to the other so a learned model can drop straight into
``astar``/``greedy`` exactly where ``manhattan`` would go — and crucially it predicts
cost-to-go from cheap local features *without solving the maze*. That is the whole
premise: a cheap estimate in place of the expensive exact answer.

Caveat we measure rather than hide: the learned estimate is not guaranteed admissible,
so A* using it may return a suboptimal path. We clamp predictions at 0 (a negative
remaining-cost is meaningless) but do nothing to force admissibility.
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
        vector = feature_vector(self.grid, cell, goal, self.window)
        prediction = float(self.model.predict([vector])[0])
        return max(0.0, prediction)
