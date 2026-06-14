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

## Outcome (in brief)

The full narrative, with figures and reasoning, is in `pathfinding_report.pdf`. In short:
pooled results were a wash, but within-group analysis revealed the learned heuristic helps
in corridor mazes and hurts in open fields — a Simpson's reversal. Adding one feature
(global obstacle density) made it dominate Manhattan: ~17% fewer nodes at ~0.2% optimality
gap, ~97% of mazes optimal, confirmed across seeds. The deeper finding: that feature is a
*regime tag* — the outcome is governed by the training distribution, not the model — shown
by pure-style runs and by asymmetric cross-distribution transfer failure. And three
features match the full seven.

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

**D11 — Batched heuristic prediction at evaluation.**
`precompute_learned_heuristic` predicts cost-to-go for *all* passable cells in one
vectorised call, then serves lookups — instead of calling `model.predict` per cell
during search. Pure speed optimisation: identical `h` values, identical node counts
(a test asserts it matches the per-cell `LearnedHeuristic`), but it makes benchmarking
practical when A* queries thousands of cells. The per-cell version is kept for clarity.

**D12 — Analyse within-group, and report distributions, not just means.**
`evaluation/analysis.py` segments every metric by maze *style* (scattered vs structured)
and reports the optimality-gap *distribution* (fraction optimal, median, p90, max), not
only its mean. Pooling across maze types hides where a heuristic actually wins or fails;
a mean gap hides whether it's uniformly small or a heavy tail. Same discipline as the
steam-reviews within-group test: make a pattern prove itself inside each subgroup before
believing the pooled number. `Maze.style` and `BenchmarkRow.maze_style` carry the tag.

## The turn to experiment-driven design

A record of *why* the project pivots from "train one model" to "an apparatus for asking
honest comparative questions" — and what triggered it. This turn is itself the maturity
signal, and it was driven by a finding, not by gold-plating.

### The finding that forced it

The first end-to-end run looked like a non-result. Pooled over all held-out mazes, the
learned heuristic was a wash: A*+learned expanded ~0.5% *more* nodes than A*+Manhattan
and gave up 1.48% optimality. If we'd stopped there, the verdict was "didn't beat the
baseline."

Segmenting by maze style (the within-group view, D12) reversed it. The pooled average
was hiding two opposite effects that nearly cancelled:

- **Scattered (open) mazes:** learned is *worse* — more nodes and a ~5% gap. Manhattan
  is already near-exact in open space, so there's nothing to gain and the model's
  overestimation only adds error.
- **Structured (corridor) mazes:** learned is *better* — ~7% fewer nodes at a 0% gap.
  Manhattan is a weak guide where walls force detours (it barely beats blind Dijkstra
  there), so a learned estimate has room to help.

A textbook pooling / Simpson's reversal: the honest result isn't "it didn't work," it's
"it helps exactly where the classic heuristic is weak and hurts where it's already
strong." Two caveats keep us honest — the structured 0% gap is partly *free* (perfect
mazes have a unique path, so any path is optimal), and the corridor win rested on only 27
mazes.

### What that implies for how we work

A single result is no longer the deliverable; the *comparisons* are. To claim "the
corridor win holds" or "global features help in corridors," we must run many controlled
variants — feature sets, maze mixes, sizes, later model objectives — and compare them
reliably. That needs three things the one-shot script lacked: variants expressed as
**configuration** (not code edits), runs that are **saved, not overwritten**, and results
that are **reproducible** (seed + config + code version).

### The plan / the design

- **`ExperimentConfig`** — one dataclass is the single source of truth for a run (data
  params, feature set, model params, transforms), saved with the run. A new knob is a new
  field; nothing else moves.
- **Registries** (name → function) for the things we'll *extend* — **features** and
  **maze generators** — so adding an option (a global feature; a braided-maze generator)
  is a registration, not a refactor.
- **Prediction → heuristic transform hook** — default identity; the seam where the
  admissibility experiment (scale down / penalise overestimation) will plug in.
- **Run persistence** — each run writes config + metrics + git hash + seed to
  `experiments/runs/<id>/`, never overwriting. Variants sit side by side; the report can
  cite "config X at commit Y," reproducible from the CLI.
- **Analysis tagging** — mazes (and benchmark rows) carry their *style*, so results
  segment by maze type (within-group), not just pooled. (Size/density segmentation was
  planned too but deferred — random-uniform sampling made it unnecessary; see Open.)

### What we deliberately did NOT do (restraint is a signal)

- **No stratified/factorial maze generation.** At n ≈ 1000, random-uniform sizes already
  give ~150 mazes per size bucket — enough to read the trend. We add the *analysis*
  (bucketed segmentation) now and would add stratified *generation* only if coverage came
  out thin. Don't build sampling machinery to solve what large N already solves. (Guard
  against confounding size with density/style by segmenting within-group, not by design.)
- **No configurable benchmark frame, alternative model classes, or experiment-tracking
  tools** (MLflow / W&B / DVC). Out of scope for a literacy phase. The registry pattern
  means any of these can be added later cheaply, so we don't pay for them now.

## Build order (one unit ≈ one commit)

1. `maze/` — grid + generators (+ tests: bounds, passability, solvable mazes) ✅
2. `search/` — dijkstra / a* / greedy on one instrumented core (+ test: A* path is optimal) ✅
3. `data/labels.py` — backward cost-to-go (+ test: matches search distances) ✅
4. `data/features.py` — features with hypotheses ✅
5. `data/dataset.py` — assembly + whole-maze holdout (+ test: split disjointness) ✅
6. `model/` — Manhattan baseline + trained regressor + learned-heuristic bridge ✅
7. `evaluation/` — heuristic quality, then the search benchmark ✅
8. `experiments/` — full pipeline run + charts ✅  (narrative writeup at the finish line)
9. `evaluation/analysis.py` — within-group segmentation + gap distribution ✅  (story extraction)

*Experiment-driven redesign (see "The turn to experiment-driven design" above):*
10. `ExperimentConfig` + feature/generator registries + prediction-transform hook (+ tests) ✅
11. Run persistence — save config + metrics + git hash + seed per run, no overwrite ✅
12. Size/density tagging + bucketed analysis — *deferred* (style segmentation carried the story; see Open)
13. Feature-importance reporting (permutation importance, MAE units) ✅
14. `--add`/`--drop` feature flags + cross-regime train/test (`--train-style`/`--test-style`) ✅
15. `experiment.run` + report data/charts + narrative PDF (`generate_report.py`) ✅

## Open / revisit later

- Acknowledged gaps deferred to a later phase, not forced here: this is **regression**
  (no class-imbalance practice) and **pure simulation** (no real-world-messy-data
  modeling). Both to be covered later in the journey.
- **Size/density bucketed segmentation: deferred.** Style segmentation carried the story,
  and with random-uniform sizes at n≈1000 each bucket is well-populated — so stratified
  sampling/analysis can wait until a question actually needs it.
- **Findings' open questions** (braided mazes for the structured-optimality caveat;
  admissibility via the transform hook; regional features vs the global ceiling;
  per-regime models vs one-model-with-a-tag) are documented in `pathfinding_report.pdf`.
- 8-connected movement (diagonals) is a possible extension; start 4-connected.
