"""Smoke test: the package imports. Real tests arrive with each build step.

Build-step test targets (to be filled in as we go):
    step 1  grid bounds/passability/neighbors; generators produce solvable mazes
    step 2  classic A* returns an OPTIMAL path (cross-check vs dijkstra)
    step 3  true_cost_to_go matches search distances on hand-checked mazes
    step 5  assemble() raises if any maze id is in both train and test (leakage guard)
"""

import importlib


def test_package_imports():
    assert importlib.import_module("pathfinding") is not None
