"""One experiment run, end to end — the shared orchestration.

`run(config)` executes the full pipeline for a single `ExperimentConfig` and returns a
record (the same structure `persistence.build_record` produces). Both the CLI
(`experiments/run_experiment.py`) and the report data layer build on this, so the report's
numbers come from exactly the code path a normal run uses — and are reproducible from the
seed in the config.
"""

from __future__ import annotations

import numpy as np

from .config import ExperimentConfig
from .data.dataset import assemble, assemble_from_splits
from .evaluation.analysis import gap_distribution, summarize_by_style
from .evaluation.heuristic_quality import evaluate
from .evaluation.search_benchmark import run_benchmark, summarize
from .maze.generator import make_mazes
from .model.baseline import ManhattanBaseline
from .model.importance import feature_importance
from .model.train import train_model
from .persistence import build_record

_STYLE_SF = {"scattered": 0.0, "structured": 1.0, "mixed": 0.5}


def run(config: ExperimentConfig) -> dict:
    """Run the pipeline for one config; return the metrics record (deterministic)."""
    rng = np.random.default_rng(config.seed)
    feats = config.feature_names

    if config.train_style and config.test_style:
        n_test = max(1, round(config.test_fraction * config.n))
        train_mazes = make_mazes(
            config.n, rng, size_range=config.size_range,
            structured_fraction=_STYLE_SF[config.train_style],
        )
        test_mazes = make_mazes(
            n_test, rng, size_range=config.size_range,
            structured_fraction=_STYLE_SF[config.test_style],
        )
        data = assemble_from_splits(train_mazes, test_mazes, feature_names=feats, window=config.window)
    else:
        mazes = make_mazes(
            config.n, rng, size_range=config.size_range,
            structured_fraction=config.structured_fraction,
        )
        data = assemble(
            mazes, test_fraction=config.test_fraction, rng=rng,
            feature_names=feats, window=config.window,
        )
        test_mazes = [mazes[i] for i in data.test_maze_ids]

    model = train_model(
        data.X_train, data.y_train,
        learning_rate=config.learning_rate, max_iter=config.max_iter,
    )
    model_q = evaluate(data.y_test, model.predict(data.X_test))
    base_q = evaluate(data.y_test, ManhattanBaseline(feats).predict(data.X_test))
    importance = feature_importance(model, data.X_test, data.y_test, feats, seed=config.seed)

    rows = run_benchmark(test_mazes, model, feature_names=feats, window=config.window)
    summary = summarize(rows)
    gap_dists = {f"{a}+{h}": gap_distribution(rows, a, h) for (a, h) in summary}

    return build_record(
        config, model_q, base_q, summary, gap_dists, summarize_by_style(rows), importance
    )
