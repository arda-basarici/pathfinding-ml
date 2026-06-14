"""Deeper cuts on benchmark rows — the analysis that turns numbers into a story.

Two views the headline means can't give:

  gap_distribution   : is the optimality gap uniform, or mostly-zero with a bad tail?
                       (A mean of 1.5% could be 'everything ~1.5%' or '80% perfect +
                       a few disasters' — very different diagnoses.)
  summarize_by_style : the within-group view — split every metric by maze type
                       (scattered vs structured) instead of pooling. The hypothesis
                       worth testing: a local-feature heuristic helps on open fields
                       (where Manhattan is already good) and fails on corridor mazes
                       (where cost-to-go is a global property it can't see).

Both consume the ``BenchmarkRow`` list from ``run_benchmark``.
"""

from __future__ import annotations

import numpy as np

from .search_benchmark import BenchmarkRow, summarize


def gap_distribution(
    rows: list[BenchmarkRow],
    algorithm: str,
    heuristic: str,
    tol: float = 1e-9,
) -> dict[str, float]:
    """Distribution of the optimality gap for one (algorithm, heuristic) over mazes.

    Returns the count, fraction of mazes solved *optimally* (gap ~ 0), and the median,
    90th percentile, and worst gap — so 'always a little long' vs 'usually perfect,
    occasionally terrible' becomes visible.
    """
    gaps = [
        r.optimality_gap
        for r in rows
        if r.algorithm == algorithm and r.heuristic == heuristic and r.found
    ]
    if not gaps:
        nan = float("nan")
        return {"n": 0, "frac_optimal": nan, "median": nan, "p90": nan, "max": nan}

    arr = np.asarray(gaps, dtype=float)
    return {
        "n": int(arr.size),
        "frac_optimal": float(np.mean(arr <= tol)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def summarize_by_style(
    rows: list[BenchmarkRow],
) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    """Run ``summarize`` separately within each maze style (the within-group view)."""
    styles = sorted({r.maze_style for r in rows if r.maze_style is not None})
    return {
        style: summarize([r for r in rows if r.maze_style == style])
        for style in styles
    }
