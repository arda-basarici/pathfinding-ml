# DESIGN — pathfinding-ml

What was built and why — the decisions and their reasoning, kept as a clean snapshot of the
design as it stands. Edited in place, not appended to; the findings and figures live in
[pathfinding_report.pdf](pathfinding_report.pdf), and the build chronology lives in git
history. How the code is structured → [ARCHITECTURE.md](ARCHITECTURE.md); the front door →
[README.md](README.md).

Decisions are anchored **D1, D2, …** where they are made below — the numbering is historical
and cited from the code (e.g. `model/train.py` cites D6), so it is preserved.

*Snapshot of the completed project · last updated 2026-07-05.*

---

## Objective

Can a model learn a better A\* heuristic than Manhattan distance — **and how do you actually
know whether it helped?** A model predicts true cost-to-go and is plugged in as the A\*
heuristic, then measured on two axes never collapsed into one: nodes expanded (speed) and
optimality gap (path quality). The deliberate constraint: a basic model (gradient-boosted
trees, hand-built features, simulated mazes), so that the project's value concentrates in the
**evaluation discipline** — the honest answer to "did it help?" — rather than model
sophistication.

---

## The decisions

**D1 — ML inside the search, not against it.**
The project is a learned *heuristic*, not a learned path-finder competing with A\*. A\* on an
admissible heuristic is already optimal; "beat A\*" is the wrong framing for a solved problem.
The real, open question is the speed/optimality tradeoff of a learned heuristic.

**D2 — Exact labels via backward search.**
The training target is true cost-to-go h\*(cell), computed exactly and cheaply with one
backward uniform-cost sweep from the goal — no noise, no approximation. Perfect labels are a
luxury; they isolate the heuristic's behaviour from label error.

**D3 — Split by whole maze (the leakage guard).**
Cells within a maze are spatially correlated; a random cell-level train/test split leaks
neighbours' answers into the test set. Whole mazes are held out, and disjointness is
*asserted in code* in the dataset assembly — the project's signature honesty lesson, codified
as a test rather than a comment.

**D4 — Beat a *strong* baseline.**
The baseline is the admissible Manhattan heuristic A\* already uses — not "predict the mean."
Beating a fair, already-good baseline is the test with teeth; the honest outcome may be
"barely," and saying so is the point.

**D5 — Two axes, never one score.**
Heuristic quality (accuracy *and* admissibility) is reported separately from downstream search
performance (nodes expanded *vs* path optimality). The deliverable is the tradeoff frontier;
collapsing it to a single number would hide the whole finding.

**D6 — Gradient-boosted trees, not a neural net.**
Small tabular feature set → boosted trees are the strong default and the right-tool choice; a
network here would be the wrong-tool signal. No neural-net comparison in this project — that
belongs to a study where the net is the point (and the successor project,
[blackjack-rl](https://github.com/arda-basarici/blackjack-rl), ran exactly that arc).

**D7 — Two unlike maze distributions, from day one.**
Scattered obstacle fields and structured (perfect corridor) mazes — deliberately *different
environments*, not two versions of one thing — so a heuristic that only works on one style is
exposed as overfit. Chosen on purpose as a generalisation/robustness check, and stated as such
in the write-up, because choosing it deliberately is itself part of the honesty story. This
decision is what made the project's central finding *visible* (see the turn, below).

**D8 — `Grid` is immutable; `Maze` is separate.**
`Grid` (height, width, frozenset of blocked cells) is a frozen dataclass: hashable, cacheable,
impossible to mutate under a running search. `Maze` (grid + start + goal) is a separate type —
the grid is the static *map*, start/goal are the *question* asked of it. Generators return
`Maze` instances already verified solvable; `Grid` holds no search logic on purpose — a data
structure shouldn't know how to explore itself. (A frozenset over a NumPy array is plenty at
≤ 45×45; revisit only if neighbour lookups ever dominate.)

**D9 — Maze generation specifics.**
Structured = recursive backtracker, producing a *perfect* maze (fully connected, no loops) —
its long forced detours are exactly where Manhattan is a poor estimate, i.e. where a learned
heuristic has room to help. Scattered = independent random blocking per cell (an open field,
not a walled maze); disconnected draws are rejected and redrawn until start/goal connect.
Endpoints are random passable cells with a minimum separation (Manhattan ≥ half the larger
side) so no instance is trivially short. Per maze, size (each side in [15, 45]) and — for
scattered — density ([0.20, 0.35]) are randomised, giving a scale/density axis on top of the
two styles. All randomness flows through one injected `numpy.random.Generator`.

**D10 — One instrumented best-first core for all three algorithms.**
Dijkstra, A\*, and Greedy are the *same* search differing only in the frontier priority
(`g`, `g + h`, `h`). They share one core function, so node expansion is counted identically
and the benchmark compares like with like — any difference in work is the algorithm or the
heuristic, never an accounting artefact. The correctness anchor is a test: A\* with an
admissible heuristic must return the same path cost as Dijkstra.

**D11 — Batched heuristic prediction at evaluation.**
Cost-to-go is predicted for *all* passable cells in one vectorised call, then served as
lookups — instead of a `model.predict` per cell during search. Pure speed: identical `h`
values, identical node counts (a test asserts the match with the per-cell version, which is
kept for clarity).

**D12 — Analyse within-group, and report distributions, not just means.**
Every metric is segmented by maze *style*, and the optimality gap is reported as a
distribution (fraction optimal, median, p90, max), never only a mean. Pooling across maze
types hides where a heuristic actually wins or fails; a mean gap hides whether the error is
uniformly small or a heavy tail. The same within-group discipline as the
[steam-reviews](https://github.com/arda-basarici/steam-reviews) study: a pattern must prove
itself inside each subgroup before the pooled number is believed.

---

## The turn to experiment-driven design

Mid-project, the design pivoted from "train one model" to "an apparatus for asking honest
comparative questions." The pivot was driven by a finding, not by gold-plating — and it is the
project's maturity signal.

**The finding that forced it.** The first end-to-end run looked like a non-result: pooled over
all held-out mazes, the learned heuristic was a wash (~0.5% *more* nodes than Manhattan, a
1.48% optimality gap). Segmenting by maze style (D12) reversed it — the pooled average was
hiding two opposite effects that nearly cancelled. In open fields the learned heuristic is
*worse* (Manhattan is already near-exact there; model error only adds noise); in corridor
mazes it is *better* (~7% fewer nodes at a 0% gap — Manhattan barely beats blind Dijkstra
where walls force detours). A textbook Simpson's reversal: the honest result was never "it
didn't work," it was "it helps exactly where the classic heuristic is weak." Two caveats kept
it honest — the structured 0% gap is partly free (perfect mazes have a unique path), and the
first corridor win rested on only 27 mazes.

**What that implied.** A single result was no longer the deliverable; the *comparisons* were.
Claiming "the corridor win holds" or "global features help" takes many controlled variants —
feature sets, maze mixes, transfer directions — compared reliably. That needs variants
expressed as **configuration** (not code edits), runs that are **saved, never overwritten**,
and results **reproducible** from seed + config + code version. Concretely:

- **`ExperimentConfig`** — one dataclass is the single source of truth for a run, saved with
  it; a new knob is a new field, nothing else moves.
- **Registries** (name → function) for the extension points — features and maze generators —
  so adding an option is a registration, not a refactor.
- **A prediction → heuristic transform hook** — default identity; the seam where an
  admissibility experiment (scale down, penalise overestimation) plugs in.
- **Run persistence** — config + metrics + git hash + seed to `experiments/runs/<id>/`, never
  overwriting; the report can cite "config X at commit Y."

**Restraint as part of the design.** No stratified maze generation (at n ≈ 1000,
random-uniform sampling already fills every size bucket — analysis segmentation suffices); no
experiment-tracking stack (MLflow / W&B / DVC) for a project this size; no configurable
benchmark frame or alternative model classes. The registry pattern keeps all of them cheap to
add later, which is exactly why they were not added now.

---

## Outcome

The apparatus earned its keep: adding one feature — global obstacle density — took the learned
heuristic to **~17% fewer nodes than Manhattan at a ~0.2% optimality gap**, ~97% of held-out
mazes solved optimally, confirmed across seeds and a 10,000-maze spot-check. Then the deeper
finding overturned the surface story: trained on open fields *alone*, the plain model already
wins with no global feature. The feature is really a **regime tag** — the outcome is governed
by the *training distribution*, not the model — shown by pure-style runs and by asymmetric
cross-style transfer failure (too timid in one direction, too reckless in the other). Three
features match the full seven. Full analysis:
[pathfinding_report.pdf](pathfinding_report.pdf).

---

## Scope & non-goals

- **Regression only, simulation only.** No class-imbalance practice, no real-world messy-data
  modeling — acknowledged gaps deferred to later projects, not forced in here.
- **4-connected movement.** Diagonals are a possible extension, never scoped in.
- **No neural network** (D6) and **no experiment-tracking tooling** — right-tool and
  right-size calls, stated above.

## Future work (curated)

Open questions documented in the report: braided (loopy) mazes to remove the
perfect-maze optimality freebie; admissibility enforcement via the transform hook; regional
features against the global-feature ceiling; per-regime models vs one model with a regime tag.
Size/density bucketed segmentation stays deferred — style segmentation carried the story, and
random-uniform sampling keeps every bucket populated if a question ever needs it.

The regime-tag lesson itself — *coverage of the training distribution governs what can be
learned* — recurred as a load-bearing diagnostic in
[blackjack-rl](https://github.com/arda-basarici/blackjack-rl).
