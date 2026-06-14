"""What a search run produced, and how much work it took.

``nodes_expanded`` is the headline efficiency metric: how many cells the algorithm
pulled off the frontier and processed. The whole project is a story about trading
``nodes_expanded`` (work) against ``path_cost`` (quality).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..maze.grid import Cell


@dataclass
class SearchResult:
    """Outcome of a single search run.

    Attributes:
        found: whether a path from start to goal was found.
        path: the path as a list of cells (start..goal), empty if not found.
        path_cost: total cost of ``path`` (its length for a unit grid).
        nodes_expanded: number of cells expanded (the work / "speed" axis).
        max_frontier: peak frontier size (memory pressure; secondary metric).
    """

    found: bool
    path: list[Cell]
    path_cost: float
    nodes_expanded: int
    max_frontier: int
