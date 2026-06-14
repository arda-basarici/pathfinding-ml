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
*Decided:* no neural-net comparison in this project — deferred to a later, DL-focused
project where the net is the point. Adding one here would buy nothing and dilute scope.

**D7 — Two maze generators, from the start.**
Scattered-obstacle and structured (recursive-backtracker / perfect) mazes, so a heuristic that only
works on one style is exposed as overfit. *Decided:* both generators from day one, not
one-then-maybe-two. Training and testing across two *unlike* maze distributions is a
deliberate generalization / robustness check — it guards against the model learning one
generator's quirks instead of something about pathfinding. This rationale is stated
explicitly in the README and the final writeup, because choosing it on purpose (rather
than for convenience) is itself part of the honesty story.

**D8 — `Grid` is immutable; `Maze` is separate.**
`Grid` (height, width, frozenset of blocked cells) is a frozen dataclass: hashable,
cacheable, and impossible to mutate under a running search. A frozenset of blocked
cells (rather than a NumPy array) is plenty for our sizes (≤ 45×45) and keeps the type
simple; revisit only if we scale up enough that neighbour lookups dominate. `Maze`
(grid + start + goal) is a separate type: the grid is the static *map*, start/goal are
the *question* asked of it. Generators return `Maze` instances already verified solvable.
`Grid` holds no search/reachability logic on purpose — a data structure shouldn't know
how to explore itself.

**D9 — Maze generation specifics.**
- *Structured style = recursive backtracker* (randomized DFS), which produces a
  *perfect* maze: fully connected, no loops. Connectivity is therefore free — any two
  passable cells are reachable — and its long forced detours are exactly the case where
  Manhattan distance is a poor estimate, i.e. where a learned heuristic has room to help.
- *Scattered style = independent random blocking* at a per-maze density. This is an
  **open field with obstacles**, not a walled maze — open cells on the border are
  terrain, not leaks. The two styles are deliberately *different environments* (sealed
  corridor maze vs open obstacle field), not two versions of one thing; that contrast is
  what D7's generalisation test rests on. Scattered grids can be disconnected, so we
  **reject-and-redraw** until start/goal are reachable.
- *Endpoint placement:* start and goal are random passable cells (interior or border) —
  placement affects only the *benchmark queries*, not the per-cell labels, so more
  variety is a richer test, and the problem is symmetric (no reason to privilege one
  end). We do require a **minimum start–goal separation** (Manhattan distance ≥ half the
  larger side) so no instance is trivially short.
- *Variation axes:* per maze we randomise size (each side uniform in [15, 45]) and, for
  the scattered style, density (uniform in [0.20, 0.35]) — a scale/density axis on top
  of the two styles (D7).
- *Reproducibility:* all randomness flows through an injected `numpy.random.Generator`.

**D10 — One instrumented best-first core for all three algorithms.**
Dijkstra, A*, and Greedy are the *same* search differing only in the frontier priority
(`g`, `g + h`, `h`). They share one `_best_first` function, so node-expansion is counted
identically and the benchmark compares like with like — any difference in work is the
algorithm/heuristic, not an accounting artefact. A monotonic counter breaks priority
ties deterministically (and avoids ever comparing cells). The correctness anchor is a
test: A* with an admissible heuristic must return the *same path cost* as Dijkstra.

## Build order (one unit ≈ one commit)

1. `maze/` — grid + generators (+ tests: bounds, passability, solvable mazes) ✅
2. `search/` — dijkstra / a* / greedy on one instrumented core (+ test: A* path is optimal) ✅
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
