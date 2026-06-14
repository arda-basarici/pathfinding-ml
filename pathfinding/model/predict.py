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

from typing import Callable

from ..data.features import DEFAULT_FEATURES, feature_vector
from ..maze.grid import Cell, Grid

# A transform maps a raw model prediction to a heuristic value. The default clamps at 0
# (a negative remaining-cost is meaningless). The admissibility experiment plugs in here
# (e.g. scale predictions down so the heuristic stops overestimating).
Transform = Callable[[float], float]


def _clamp_nonneg(prediction: float) -> float:
    return max(0.0, float(prediction))


class LearnedHeuristic:
    """Wrap a trained model as a search heuristic over a fixed maze.

    Bound to one ``grid`` because features depend on the maze; rebuild per maze at
    evaluation time. ``feature_names`` must match what the model was trained on.
    """

    def __init__(
        self,
        model,
        grid: Grid,
        feature_names: list[str] | None = None,
        window: int = 2,
        transform: Transform | None = None,
    ) -> None:
        self.model = model
        self.grid = grid
        self.feature_names = DEFAULT_FEATURES if feature_names is None else feature_names
        self.window = window
        self.transform = transform or _clamp_nonneg

    def __call__(self, cell: Cell, goal: Cell) -> float:
        """Predicted cost-to-go from ``cell`` to ``goal`` (model on extracted features)."""
        vector = feature_vector(self.grid, cell, goal, self.feature_names, self.window)
        return self.transform(self.model.predict([vector])[0])


def precompute_learned_heuristic(
    model,
    grid: Grid,
    goal: Cell,
    feature_names: list[str] | None = None,
    window: int = 2,
    transform: Transform | None = None,
):
    """A fast learned heuristic: predict cost-to-go for *all* passable cells in ONE
    batched call, then serve cheap lookups.

    Same values as ``LearnedHeuristic`` — so search behaviour and node counts are
    identical — but it avoids the per-cell ``model.predict`` overhead, which matters
    because A* can query thousands of cells. Bound to one grid + goal.
    """
    names = DEFAULT_FEATURES if feature_names is None else feature_names
    tf = transform or _clamp_nonneg

    cells = [
        (r, c)
        for r in range(grid.height)
        for c in range(grid.width)
        if grid.passable((r, c))
    ]
    if not cells:
        return lambda cell, goal_: 0.0

    vectors = [feature_vector(grid, cell, goal, names, window) for cell in cells]
    predictions = model.predict(vectors)
    table = {cell: tf(p) for cell, p in zip(cells, predictions)}

    def heuristic(cell: Cell, goal_: Cell) -> float:
        return table.get(cell, 0.0)

    return heuristic
