"""Tests for the evaluation layer (both axes).

Axis 1 (heuristic_quality) is pinned with hand-computed arrays. Axis 2 (search_benchmark)
is checked by invariants that must hold no matter what the model learned: Dijkstra and
admissible-A* are always optimal, every solvable maze is solved, and informed search
never does more work than blind search.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pathfinding.data.dataset import assemble
from pathfinding.evaluation.heuristic_quality import evaluate
from pathfinding.evaluation.search_benchmark import (
    BenchmarkRow,
    run_benchmark,
    summarize,
)
from pathfinding.maze.generator import make_mazes
from pathfinding.model.train import train_model


# --------------------------------------------------------------------------- #
# Axis 1 — heuristic quality.
# --------------------------------------------------------------------------- #
def test_quality_perfect_predictions():
    report = evaluate([1, 2, 3], [1, 2, 3])
    assert report.mae == 0.0
    assert report.rmse == 0.0
    assert report.frac_overestimated == 0.0
    assert report.max_overestimate == 0.0


def test_quality_overestimate_is_detected():
    # One prediction overestimates by 1 (2 vs 1); the rest are exact.
    report = evaluate(y_true=[1, 2, 3], y_pred=[2, 2, 3])
    assert report.mae == 1 / 3
    assert report.rmse == math.sqrt(1 / 3)
    assert report.frac_overestimated == 1 / 3
    assert report.max_overestimate == 1.0


def test_quality_underestimate_is_admissible():
    # Underestimating never counts as overestimation.
    report = evaluate(y_true=[5, 5, 5], y_pred=[0, 1, 2])
    assert report.frac_overestimated == 0.0
    assert report.max_overestimate == 0.0


# --------------------------------------------------------------------------- #
# Axis 2 — search benchmark.
# --------------------------------------------------------------------------- #
def test_optimality_gap_math():
    row = BenchmarkRow(0, "x", "y", nodes_expanded=1, path_cost=12, optimal_cost=10, found=True)
    assert row.optimality_gap == pytest.approx(0.2)
    exact = BenchmarkRow(0, "x", "y", 1, path_cost=10, optimal_cost=10, found=True)
    assert exact.optimality_gap == 0.0


def test_benchmark_invariants():
    rng = np.random.default_rng(0)
    mazes = make_mazes(10, rng, size_range=(11, 15))
    data = assemble(mazes, test_fraction=0.25, rng=rng)
    model = train_model(data.X_train, data.y_train)

    rows = run_benchmark(mazes, model)
    assert len(rows) == 5 * len(mazes)             # 5 combos per maze
    assert all(r.found for r in rows)              # mazes are solvable

    by_maze = {}
    for r in rows:
        by_maze.setdefault(r.maze_id, {})[(r.algorithm, r.heuristic)] = r

    for combos in by_maze.values():
        dij = combos[("dijkstra", "none")]
        astar_m = combos[("astar", "manhattan")]
        # Dijkstra and admissible A* are optimal.
        assert dij.optimality_gap == 0.0
        assert astar_m.optimality_gap == 0.0
        # Informed search never expands more than blind search.
        assert astar_m.nodes_expanded <= dij.nodes_expanded


def test_summarize_keys_and_rates():
    rng = np.random.default_rng(1)
    mazes = make_mazes(8, rng, size_range=(11, 15))
    data = assemble(mazes, test_fraction=0.25, rng=rng)
    model = train_model(data.X_train, data.y_train)

    summary = summarize(run_benchmark(mazes, model))
    assert ("dijkstra", "none") in summary
    assert ("astar", "learned") in summary
    assert summary[("dijkstra", "none")]["found_rate"] == 1.0
    # Dijkstra and admissible A* average a zero optimality gap.
    assert summary[("dijkstra", "none")]["mean_optimality_gap"] == 0.0
    assert summary[("astar", "manhattan")]["mean_optimality_gap"] == 0.0
