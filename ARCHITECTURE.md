# ARCHITECTURE — pathfinding-ml

How the project is built and why that structure — the module graph, the seams, and the
structural stories, kept as a clean snapshot of the code as it stands. Edited in place; the
decisions and their reasoning (anchored D1, D2, …) → [DESIGN.md](DESIGN.md); the front door →
[README.md](README.md).

*Snapshot of the completed project · last updated 2026-07-05.*

---

## Design shape

One-directional data flow through the library, one job per module; `search/` is the shared
measuring instrument consumed from both ends:

```
        pathfinding/
          maze/        grid + the two generators (scattered field · perfect corridor)
             │
             ▼
          data/        labels (exact cost-to-go) → features (registry) → dataset assembly
             │                                                (whole-maze split, asserted)
             ▼
          model/       Manhattan baseline · boosted-tree regressor · heuristic bridge ·
             │         permutation importance
             ▼
          evaluation/  axis 1: heuristic quality (accuracy + admissibility)
             │         axis 2: search benchmark · within-group analysis
             │
          search/  ◄───┘   dijkstra / a* / greedy on ONE instrumented best-first core —
             ▲             the single place node expansion is counted
             └── also drives label generation (backward sweep)

          config.py       ExperimentConfig — the single source of truth for a run
          experiment.py   run(config): the full pipeline, end to end
          persistence.py  save each run (config · metrics · git hash · seed), never overwrite

        experiments/run_experiment.py     the CLI (feature flags, transfer runs)
        generate_report.py · report_data.py · report_charts.py     the narrative PDF
```

The rules, stated once:

- **Data types hold no behaviour of their journey.** `Grid` is a frozen dataclass with no
  search logic (DESIGN D8); rendering is a separate presentation module; search consumes
  grids, never the other way around.
- **Everything measured passes through one core.** All three algorithms share the single
  instrumented best-first function (D10), so every node-expansion count in the project is
  computed by the same code.
- **Extension points are registries.** Features and maze generators are name → function
  registries; the CLI's `--add`/`--drop` and style flags resolve against them. Adding an
  option is a registration, not a refactor.
- **Effects at the shell.** The library is pure computation over injected RNGs; file I/O
  lives in `persistence.py`, the CLI, and the report layer.

### The life of a run

```
  ExperimentConfig + seed
        │
        ▼
  experiment.run(config)      generate mazes → label → featurize → assemble (leakage-guarded)
        │                     → train → bridge into A* → benchmark → within-group analysis
        ▼
  experiments/runs/<id>/      config + metrics + git hash + seed — never overwritten
        │
        ▼
  report_data.py (cached in .report_cache/) → report_charts.py → generate_report.py
        │
        ▼
  pathfinding_report.pdf      (committed)
```

`experiment.run` is shared by the CLI and the report generator — the report's numbers come
from the same code path as any hand-run experiment, cached rather than recomputed.

## Module responsibilities

One line per module; detail lives in the docstrings.

| module | single job |
| --- | --- |
| `maze/grid.py` | `Grid` (frozen: dims + blocked cells) and `Maze` (grid + start + goal) — the map vs the question |
| `maze/generator.py` | the two generators, both returning verified-solvable `Maze`s with style tags |
| `maze/render.py` | ASCII rendering for eyeballing generators — presentation only |
| `search/algorithms.py` | dijkstra / a\* / greedy as one best-first core differing only in frontier priority |
| `search/heuristics.py` | the heuristic contract `(cell, goal) → float` + the classic estimates |
| `search/instrumentation.py` | what a run produced and what it cost — `nodes_expanded` is the headline |
| `data/labels.py` | exact cost-to-go for every cell via one backward sweep from the goal |
| `data/features.py` | the feature registry: named `(grid, cell, goal, window) → float` functions with hypotheses |
| `data/dataset.py` | (features, label) assembly with the whole-maze split asserted — the leakage guard |
| `model/baseline.py` | Manhattan as the *fair* baseline predictor |
| `model/train.py` | the gradient-boosted regressor (D6) |
| `model/predict.py` | the bridge: trained model → search `Heuristic`, including the batched precompute |
| `model/importance.py` | permutation importance in MAE units — what the model actually relies on |
| `evaluation/heuristic_quality.py` | axis 1: is the estimate good? accuracy and admissibility, kept separate |
| `evaluation/search_benchmark.py` | axis 2: does search get better, at what optimality cost? (the deliverable) |
| `evaluation/analysis.py` | within-group segmentation + gap distributions — the numbers-to-story layer |
| `config.py` · `experiment.py` · `persistence.py` | the run contract: one config, one pipeline, one immutable run record |
| `experiments/run_experiment.py` | the CLI: one run per invocation, feature and transfer flags |
| `experiments/inspect_mazes.py` | print a few mazes from the real pipeline — eyeball check for the generators |
| `report_data.py` · `report_charts.py` · `generate_report.py` | compute (cached) → render → assemble the PDF |

## Seams that carried weight

- **The heuristic contract** — any `(cell, goal) → float` plugs into A\*; Manhattan, the
  learned bridge, and the batched lookup are interchangeable to the search code.
- **The feature registry** — every ablation in the study (`--add global_obstacle_density`,
  three-vs-seven features) was a flag, not a code edit.
- **The prediction → heuristic transform hook** — default identity; the designed seam for
  admissibility experiments (documented future work, never needed for the core story).
- **Style tags on mazes and benchmark rows** — what makes within-group analysis (D12) a
  `groupby`, not a re-run.

---

## Structural stories

### One instrumented core, three algorithms

Dijkstra, A\*, and Greedy differ *only* in the priority they pop the frontier on (`g`,
`g + h`, `h`) — so they are one function, not three implementations (D10). Node expansion is
counted in exactly one place, a monotonic counter breaks priority ties deterministically, and
the correctness anchor is a test: A\* with an admissible heuristic must return the same path
cost as Dijkstra. Every comparison in the report inherits this like-for-like guarantee.

### The leakage guard is an assertion, not a convention

The whole-maze train/test split (D3) is enforced where the dataset is assembled — the code
asserts maze-level disjointness, and a test exercises it. The project's most quotable honesty
lesson lives in the type of guarantee only code can give.

### The apparatus, not a script

The mid-project pivot (DESIGN — *the turn to experiment-driven design*) left a specific
structure: `ExperimentConfig` as the single source of truth, `experiment.run` as the one
pipeline both the CLI and the report call, and immutable run records under
`experiments/runs/<id>/` carrying config + metrics + git hash + seed. Every claim in the
report is a citable (config, commit) pair; the transfer experiments and ablations that
produced the regime-tag finding were CLI flags over this one seam.

### The report reads, it never re-runs

`report_data.py` computes the report's numbers through the same `experiment.run` path and
caches them (`.report_cache/`, git-ignored); `generate_report.py` renders from the cache.
Rebuilding the PDF is cheap and cannot silently drift from how experiments are actually run —
the one honest caveat being that bit-exact reproduction requires pinning numpy/scikit-learn
(HistGradientBoostingRegressor internals shift across releases; noted in `requirements.txt`).

---

## Deliberately not done

- **No stratified/factorial maze generation.** Random-uniform sampling at n ≈ 1000 populates
  every size bucket; segmentation in analysis answers what stratified generation would have —
  machinery avoided because large N already solves it.
- **No experiment-tracking stack** (MLflow / W&B / DVC) and **no plugin framework beyond the
  two registries.** Right-sized for a single-question study; the registries keep heavier
  tooling cheap to adopt if ever needed.
- **No neural model and no 8-connected movement** — scope decisions, held (DESIGN D6, scope).
