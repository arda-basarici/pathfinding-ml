"""Search algorithms, all sharing one instrumented best-first core.

Three algorithms, distinguished only by the priority they pop the frontier on:

    dijkstra:  priority = g(n)            (cost so far; uninformed, optimal)
    astar:     priority = g(n) + h(n)     (informed, optimal iff h admissible)
    greedy:    priority = h(n)            (informed, fast, NOT optimal)

Keeping them as one core makes the comparison fair: same data structures, same
expansion accounting, only the priority differs. Every function returns a
``SearchResult`` so node-expansion counts are measured identically.
"""

from __future__ import annotations

from ..maze.grid import Cell, Grid
from .heuristics import Heuristic
from .instrumentation import SearchResult


def _best_first(
    grid: Grid,
    start: Cell,
    goal: Cell,
    priority: "Callable[[Cell, float], float]",  # noqa: F821 — (cell, g) -> priority
) -> SearchResult:
    """Generic instrumented best-first search.

    ``priority(cell, g)`` maps a node and its cost-so-far to its frontier priority.
    This is the single place expansions are counted, so all three algorithms report
    ``nodes_expanded`` on the same basis.
    """
    raise NotImplementedError  # build step 2


def dijkstra(grid: Grid, start: Cell, goal: Cell) -> SearchResult:
    """Uniform-cost search. Optimal, uninformed — the 'blind but correct' anchor."""
    raise NotImplementedError  # build step 2


def astar(grid: Grid, start: Cell, goal: Cell, heuristic: Heuristic) -> SearchResult:
    """A* with a pluggable heuristic. Optimal iff ``heuristic`` is admissible."""
    raise NotImplementedError  # build step 2


def greedy(grid: Grid, start: Cell, goal: Cell, heuristic: Heuristic) -> SearchResult:
    """Greedy best-first. Expands toward the goal aggressively; not optimal."""
    raise NotImplementedError  # build step 2
