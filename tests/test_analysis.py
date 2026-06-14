"""Tests for the deeper analysis cuts: gap distribution and within-group summary."""

from __future__ import annotations

import pytest

from pathfinding.evaluation.analysis import gap_distribution, summarize_by_style
from pathfinding.evaluation.search_benchmark import BenchmarkRow


def _row(maze_id, algo, heur, nodes, path_cost, optimal, style=None):
    return BenchmarkRow(
        maze_id, algo, heur, nodes, path_cost, optimal, found=True, maze_style=style
    )


def test_gap_distribution_separates_uniform_from_tail():
    # Gaps of 0, 0, 0.1, 0.2 for astar+learned; an unrelated row is ignored.
    rows = [
        _row(0, "astar", "learned", 10, 100, 100),   # gap 0.0
        _row(1, "astar", "learned", 10, 100, 100),   # gap 0.0
        _row(2, "astar", "learned", 10, 110, 100),   # gap 0.1
        _row(3, "astar", "learned", 10, 120, 100),   # gap 0.2
        _row(4, "dijkstra", "none", 10, 100, 100),   # different combo, excluded
    ]
    d = gap_distribution(rows, "astar", "learned")
    assert d["n"] == 4
    assert d["frac_optimal"] == 0.5                  # two of four are optimal
    assert d["median"] == pytest.approx(0.05)        # median of [0, 0, .1, .2]
    assert d["max"] == pytest.approx(0.2)


def test_gap_distribution_empty():
    assert gap_distribution([], "astar", "learned")["n"] == 0


def test_summarize_by_style_splits_groups():
    rows = [
        _row(0, "astar", "manhattan", 50, 100, 100, style="scattered"),
        _row(1, "astar", "manhattan", 80, 100, 100, style="structured"),
    ]
    by_style = summarize_by_style(rows)
    assert set(by_style.keys()) == {"scattered", "structured"}
    assert by_style["scattered"][("astar", "manhattan")]["mean_nodes_expanded"] == 50.0
    assert by_style["structured"][("astar", "manhattan")]["mean_nodes_expanded"] == 80.0
