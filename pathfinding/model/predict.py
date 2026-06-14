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


def precompute_learned_heuristic(model, grid: Grid, goal: Cell, window: int = 2):
    """A fast learned heuristic: predict cost-to-go for *all* passable cells in ONE
    batched call, then serve cheap lookups.

    Same values as ``LearnedHeuristic`` — so search behaviour and node counts are
    identical — but it avoids the per-cell ``model.predict`` overhead, which matters
    because A* can query thousands of cells. Bound to one grid + goal.
    """
    cells = [
        (r, c)
        for r in range(grid.height)
        for c in range(grid.width)
        if grid.passable((r, c))
    ]
    if not cells:
        return lambda cell, goal_: 0.0

    vectors = [feature_vector(grid, cell, goal, window) for cell in cells]
    predictions = model.predict(vectors)
    table = {cell: max(0.0, float(p)) for cell, p in zip(cells, predictions)}

    def heuristic(cell: Cell, goal_: Cell) -> float:
        return table.get(cell, 0.0)

    return heuristic
