"""Tests for the shared experiment.run orchestration (used by CLI and report)."""

from __future__ import annotations

from pathfinding.config import ExperimentConfig
from pathfinding.experiment import run


def test_run_returns_complete_record():
    rec = run(ExperimentConfig(n=6, seed=0))
    assert rec["config"]["n"] == 6
    assert "astar+learned" in rec["axis2_summary"]
    assert "astar+manhattan" in rec["axis2_summary"]
    assert "scattered" in rec["by_style"] or "structured" in rec["by_style"]
    assert rec["feature_importance"]                     # non-empty list


def test_run_cross_regime_record():
    rec = run(ExperimentConfig(n=6, seed=0, train_style="scattered", test_style="structured"))
    assert rec["config"]["train_style"] == "scattered"
    assert rec["config"]["test_style"] == "structured"
    assert "astar+learned" in rec["axis2_summary"]
