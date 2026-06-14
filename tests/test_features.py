"""Tests for per-cell feature extraction.

Hand-built grids with known answers, so each feature's arithmetic is pinned down. The
density and blocked-neighbor counts also confirm that out-of-bounds is treated as
impassable (border cells read as more enclosed).
"""

from __future__ import annotations

from pathfinding.data.features import FEATURE_NAMES, extract_features, feature_vector
from pathfinding.maze.grid import Grid


def test_keys_match_feature_names():
    grid = Grid(5, 5, frozenset())
    features = extract_features(grid, (2, 2), (0, 0))
    assert set(features.keys()) == set(FEATURE_NAMES)


def test_distance_and_delta_values():
    grid = Grid(5, 5, frozenset())
    f = extract_features(grid, cell=(0, 0), goal=(3, 4))
    assert f["manhattan_to_goal"] == 7.0
    assert f["euclidean_to_goal"] == 5.0          # 3-4-5 triangle
    assert f["row_delta"] == 3.0                   # goal is 3 rows below
    assert f["col_delta"] == 4.0


def test_blocked_neighbors_counts_walls_and_oob():
    # Corner (0,0): up and left are out of bounds -> 2 blocked neighbours.
    grid = Grid(5, 5, frozenset())
    assert extract_features(grid, (0, 0), (4, 4))["blocked_neighbors"] == 2.0

    # Interior cell, fully open -> 0.
    assert extract_features(grid, (2, 2), (4, 4))["blocked_neighbors"] == 0.0

    # One real wall next to an interior cell -> 1.
    walled = Grid(3, 3, frozenset({(0, 1)}))
    assert extract_features(walled, (1, 1), (2, 2))["blocked_neighbors"] == 1.0


def test_local_obstacle_density_includes_oob():
    # 5x5 open grid, cell (0,0), window=2 -> 5x5 box has 9 in-bounds (all passable),
    # so 16 of 25 are out-of-bounds => density 16/25.
    grid = Grid(5, 5, frozenset())
    f = extract_features(grid, (0, 0), (4, 4), window=2)
    assert f["local_obstacle_density"] == 16 / 25

    # Center of a large open grid, window=1 -> nothing impassable.
    big = Grid(11, 11, frozenset())
    assert extract_features(big, (5, 5), (0, 0), window=1)["local_obstacle_density"] == 0.0


def test_feature_vector_matches_name_order():
    grid = Grid(5, 5, frozenset())
    cell, goal = (1, 2), (4, 4)
    features = extract_features(grid, cell, goal)
    vector = feature_vector(grid, cell, goal)
    assert vector == [features[name] for name in FEATURE_NAMES]
