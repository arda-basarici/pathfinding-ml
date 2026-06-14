"""Tests for run persistence: a run record round-trips to disk, never overwriting."""

from __future__ import annotations

import json

from pathfinding.config import ExperimentConfig
from pathfinding.evaluation.heuristic_quality import QualityReport
from pathfinding.persistence import build_record, save_run


def _summary():
    return {
        ("astar", "learned"): {
            "mean_nodes_expanded": 150.0,
            "mean_optimality_gap": 0.002,
            "found_rate": 1.0,
            "n": 250.0,
        }
    }


def test_build_and_save_run_roundtrip(tmp_path):
    config = ExperimentConfig(n=10, seed=3)
    mq = QualityReport(mae=1.0, rmse=2.0, frac_overestimated=0.5, max_overestimate=3.0)
    bq = QualityReport(mae=2.0, rmse=3.0, frac_overestimated=0.0, max_overestimate=0.0)
    summary = _summary()
    gap_dists = {"astar+learned": {"n": 250, "frac_optimal": 0.97, "median": 0.0, "p90": 0.0, "max": 0.1}}
    by_style = {"scattered": summary, "structured": summary}

    record = build_record(config, mq, bq, summary, gap_dists, by_style)
    run_dir = save_run(tmp_path, record, run_id="testrun")

    path = run_dir / "record.json"
    assert path.exists()
    loaded = json.loads(path.read_text())

    assert loaded["run_id"] == "testrun"
    assert "timestamp" in loaded
    assert loaded["config"]["n"] == 10 and loaded["config"]["seed"] == 3
    # tuple keys were flattened to strings for JSON
    assert "astar+learned" in loaded["axis2_summary"]
    assert "astar+learned" in loaded["by_style"]["scattered"]
    assert loaded["axis1_quality"]["learned"]["mae"] == 1.0


def test_save_run_does_not_overwrite(tmp_path):
    config = ExperimentConfig(n=5, seed=1)
    record = build_record(
        config, QualityReport(0, 0, 0, 0), QualityReport(0, 0, 0, 0), {}, {}, {}
    )
    d1 = save_run(tmp_path, record, run_id="a")
    d2 = save_run(tmp_path, record, run_id="b")
    assert d1 != d2
    assert (d1 / "record.json").exists() and (d2 / "record.json").exists()
