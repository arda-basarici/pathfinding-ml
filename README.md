# pathfinding-ml

**Can a learned heuristic make A\* search less work — and what does it cost you?**

A* is only as good as its heuristic *h(n)*, the estimate of remaining cost to the goal.
With an admissible heuristic (Manhattan distance) A* is optimal but can expand a lot of
the grid. This project trains a model to predict cost-to-go and plugs it in as the
heuristic, then measures the trade honestly: **fewer nodes expanded, at the price of
paths that are sometimes longer than optimal.**

It is deliberately *not* "ML vs A*." A* is a solved, optimal algorithm — beating it
head-on is the wrong tool for a settled problem. The interesting question is ML *inside*
the search, as the heuristic, where the speed/optimality tradeoff actually lives.

## The question, precisely

For known search algorithms, swap the classic heuristic for a learned one and compare on
two axes that must never collapse into one number:

- **Nodes expanded** — the work the algorithm did (the speed story).
- **Path optimality gap** — how much longer than the true shortest path (the quality story).

The expected, honest result is a *frontier*, not a winner: the learned heuristic trades
guaranteed-optimal-but-slow for usually-fast-but-sometimes-suboptimal.

## Pipeline

```
generate mazes → exact cost-to-go labels (backward BFS) → per-cell features
   → assemble dataset (WHOLE-MAZE holdout) → train regressor
   → evaluate: heuristic quality + search benchmark → writeup
```

## Layout

```
pathfinding/
  maze/        grid representation + maze generators
  search/      dijkstra / a* / greedy (one instrumented core), heuristics, results
  data/        exact labels, features (each with a hypothesis), dataset assembly
  model/       Manhattan baseline, trained cost-to-go regressor, learned-heuristic bridge
  evaluation/  heuristic quality (accuracy + admissibility), search benchmark (the tradeoff)
experiments/   notebooks → the tradeoff charts
tests/         pytest
main.py        CLI: generate | label | train | evaluate
```

## Two honesty hooks built into the structure

- **Whole-maze holdout.** Cells from one maze share structure, so a random cell-level
  split leaks test answers through neighbours. `data/dataset.py` splits by whole maze and
  *asserts* the train/test maze sets are disjoint.
- **Admissibility, reported.** A learned heuristic isn't guaranteed to never overestimate,
  and overestimation is exactly what breaks A*'s optimality. We report how often and by how
  much it overestimates — not just average error.

## Status

🔨 Scaffolded. Implementation proceeds one unit at a time (see `ARCHITECTURE.md` for the
build order and the decisions behind the design).

## Context

Phase 2 of [AI Journey](../../README.md) — the project where a trained model is the
centerpiece. Extends the Phase 1 A* maze-solver.
