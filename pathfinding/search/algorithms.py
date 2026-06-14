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

import heapq
import itertools
from math import inf
from typing import Callable

from ..maze.grid import Cell, Grid
from .heuristics import Heuristic
from .instrumentation import SearchResult

# A priority function maps (cell, cost-so-far) to its frontier priority.
PriorityFn = Callable[[Cell, float], float]


def _reconstruct(came_from: dict[Cell, Cell], start: Cell, goal: Cell) -> list[Cell]:
    """Walk parent pointers from goal back to start, return start..goal."""
    path = [goal]
    node = goal
    while node != start:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


def _best_first(
    grid: Grid,
    start: Cell,
    goal: Cell,
    priority: PriorityFn,
) -> SearchResult:
    """Generic instrumented best-first search.

    ``priority(cell, g)`` maps a node and its cost-so-far to its frontier priority.
    This is the single place expansions are counted, so all three algorithms report
    ``nodes_expanded`` on the same basis. A monotonic counter breaks priority ties so
    ordering is deterministic (and never compares cells).
    """
    counter = itertools.count()
    g_score: dict[Cell, float] = {start: 0.0}
    came_from: dict[Cell, Cell] = {}
    frontier = [(priority(start, 0.0), next(counter), start)]
    closed: set[Cell] = set()
    nodes_expanded = 0
    max_frontier = 1

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, _, current = heapq.heappop(frontier)

        if current in closed:
            continue  # stale duplicate left on the heap; skip
        closed.add(current)
        nodes_expanded += 1

        if current == goal:
            return SearchResult(
                found=True,
                path=_reconstruct(came_from, start, goal),
                path_cost=g_score[goal],
                nodes_expanded=nodes_expanded,
                max_frontier=max_frontier,
            )

        g_current = g_score[current]
        for neighbour in grid.neighbors(current):
            if neighbour in closed:
                continue
            tentative_g = g_current + grid.step_cost(current, neighbour)
            if tentative_g < g_score.get(neighbour, inf):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                heapq.heappush(
                    frontier,
                    (priority(neighbour, tentative_g), next(counter), neighbour),
                )

    return SearchResult(
        found=False,
        path=[],
        path_cost=inf,
        nodes_expanded=nodes_expanded,
        max_frontier=max_frontier,
    )


def dijkstra(grid: Grid, start: Cell, goal: Cell) -> SearchResult:
    """Uniform-cost search. Optimal, uninformed — the 'blind but correct' anchor."""
    return _best_first(grid, start, goal, lambda cell, g: g)


def astar(grid: Grid, start: Cell, goal: Cell, heuristic: Heuristic) -> SearchResult:
    """A* with a pluggable heuristic. Optimal iff ``heuristic`` is admissible."""
    return _best_first(grid, start, goal, lambda cell, g: g + heuristic(cell, goal))


def greedy(grid: Grid, start: Cell, goal: Cell, heuristic: Heuristic) -> SearchResult:
    """Greedy best-first. Expands toward the goal aggressively; not optimal."""
    return _best_first(grid, start, goal, lambda cell, g: heuristic(cell, goal))
