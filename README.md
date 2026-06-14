# pathfinding-ml

**Can a model learn a better A\* heuristic than Manhattan distance — and how do you actually know whether it helped?**

A* finds shortest paths quickly when its *heuristic* — its estimate of the distance still
to go — is good. Manhattan distance is the classic estimate, but it's blind to walls. This
project trains a model to predict true cost-to-go and plugs it in as the heuristic, then
measures the result honestly, on two axes that are never collapsed into one: **nodes
expanded** (speed) and **optimality gap** (path quality).

The interesting part turned out not to be the model. It was how much work it took to
*know* whether the model helped — and how the answer changed as we looked closer.

> **Scope, stated plainly.** This is a deliberately **basic ML project**: a
> gradient-boosted regressor on a handful of hand-built features, run on *simulated* mazes
> (open obstacle fields and perfect corridor mazes) on a 4-connected grid. It is a
> literacy-phase project — its value is the **evaluation discipline**, not model
> sophistication. It extends the Phase 1 A\* maze-solver.

## What we found

The full narrative, with figures, is in **[`pathfinding_report.pdf`](pathfinding_report.pdf)**. In short:

- **The average lied.** Pooled over all mazes the learned heuristic looked like a wash.
  Split by maze type, it *reversed*: a win in corridor mazes, a loss in open fields — a
  Simpson's-paradox trap. Pool two opposite effects and the average reports nothing.
- **One feature flipped it.** Adding *global obstacle density* took it to **~17% fewer
  nodes than Manhattan at a ~0.2% optimality gap**, with ~97% of held-out mazes solved
  optimally — confirmed across random seeds (and a 10,000-maze spot-check).
- **It was never the feature — it was the data.** Trained on open fields *alone*, the
  plain model already wins, with no global density. That feature is really a **regime
  tag**: the result is governed by the *training distribution*, not the model. Train on
  one maze type and test on the other and it fails — in two opposite ways (too timid, or
  too reckless).
- **Three features do the work of seven.** Most of the feature set was redundant.

## Layout

```
pathfinding/
  maze/        grid + two generators (scattered field, perfect corridor maze)
  search/      dijkstra / a* / greedy on one instrumented core; heuristics
  data/        exact cost-to-go labels, a feature registry, dataset assembly
  model/       Manhattan baseline, gradient-boosted regressor, learned-heuristic bridge, importance
  evaluation/  heuristic quality, search benchmark, within-group analysis
  experiment.py   one run, end to end (shared by the CLI and the report)
  config.py       ExperimentConfig — the single source of truth for a run
  persistence.py  save each run (config + metrics + git hash), never overwriting
experiments/   run_experiment.py (CLI) · saved runs
generate_report.py · report_data.py · report_charts.py   the narrative PDF
tests/         pytest
```

Three honesty guards are built into the structure: a **whole-maze train/test split**
(asserted in code, so correlated cells can't leak), an **admissibility report** (a learned
heuristic can overestimate and break optimality — we measure how often), and **two unlike
maze distributions** (a heuristic that only works on one style is exposed as overfit).

## Run it

```bash
pip install -r requirements.txt
python -m pytest                       # the pipeline is fully tested

# one experiment (saves a reproducible run record + charts)
python experiments/run_experiment.py --n 1000
python experiments/run_experiment.py --add global_obstacle_density          # ablate a feature
python experiments/run_experiment.py --train-style scattered --test-style structured   # transfer

# build the narrative report
python generate_report.py              # writes pathfinding_report.pdf
```

## What it demonstrates

The subject is pathfinding, but the project is really about the discipline of knowing
whether a result is real: refusing to trust a pooled average (it inverted), beating a
*strong* baseline rather than a strawman, separating the metric you can optimise from the
objective you care about, guarding against leakage, and — the turn that mattered most —
being willing to overturn your own explanation when the data demands it. The reasoning
behind every design choice is logged in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

Phase 2 of [AI Journey](../../README.md) — the project where a trained model is the
centrepiece. Code, tests, and the full decision log: github.com/arda-basarici/ai-journey
