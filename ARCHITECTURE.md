# Architecture & Decisions

A decisions log: not just *what* the structure is, but *why*. Living document — append
as choices are made or revised.

## Design shape

One-directional data flow, one job per module:

```
maze/  →  data/labels  →  data/features  →  data/dataset  →  model/  →  evaluation/
```

`search/` is orthogonal: it's consumed by both label generation (indirectly) and
evaluation, and is the one place node-expansion is counted, so every algorithm is
measured on the same basis.

## Decisions

**D1 — ML inside the search, not against it.**
The project is a learned *heuristic*, not a learned path-finder competing with A*. A* on
an admissible heuristic is already optimal; "beat A*" is the wrong framing for a solved
problem. The real, open question is the speed/optimality tradeoff of a learned heuristic.

**D2 — Exact labels via backward search.**
The training target is true cost-to-go h*(cell). We get it exactly and cheaply with one
backward BFS/uniform-cost sweep from the goal — no noise, no approximation. Perfect
labels are a luxury; they let us isolate the heuristic's behaviour from label error.

**D3 — Split by whole maze (the leakage guard).**
Cells within a maze are spatially correlated; a random cell-level train/test split leaks
neighbours' answers into the test set. We hold out *whole mazes* and assert disjointness
in `data/dataset.assemble`. This is the project's signature honesty lesson, codified as a
test rather than a comment.

**D4 — Beat a *strong* baseline.**
The baseline is the admissible Manhattan heuristic A* already uses — not "predict the
mean." Beating a fair, already-good baseline is the test with teeth; the honest outcome
may be "barely," and saying so is the point.

**D5 — Two axes, never one score.**
Heuristic quality (accuracy *and* admissibility) is reported separately from downstream
search performance (nodes expanded *vs* path optimality). The deliverable is the
tradeoff frontier; collapsing it to a single number would hide the whole finding.

**D6 — Gradient-boosted trees, not a neural net.**
Small tabular feature set → boosted trees are the strong default and the right-tool
choice. A net here would be the wrong-tool signal and is Phase 3's territory. The project
is about the measurement, not the architecture.

**D7 — Multiple maze generators.**
Scattered-obstacle and structured (recursive-division) mazes, so a heuristic that only
works on one style is exposed as overfit. Mixing generators is part of the honesty story.

## Build order (one unit ≈ one commit)

1. `maze/` — grid + generators (+ tests: bounds, passability, solvable mazes)
2. `search/` — dijkstra / a* / greedy on one instrumented core (+ test: A* path is optimal)
3. `data/labels.py` — backward cost-to-go (+ test: matches search distances)
4. `data/features.py` — features with hypotheses
5. `data/dataset.py` — assembly + whole-maze holdout (+ test: split disjointness)
6. `model/` — Manhattan baseline + trained regressor + learned-heuristic bridge
7. `evaluation/` — heuristic quality, then the search benchmark
8. `experiments/` — charts + writeup

## Open / revisit later

- Acknowledged gaps deferred to a later phase, not forced here: this is **regression**
  (no class-imbalance practice) and **pure simulation** (no real-world-messy-data
  modeling). Both to be covered later in the journey.
- 8-connected movement (diagonals) is a possible extension; start 4-connected.
