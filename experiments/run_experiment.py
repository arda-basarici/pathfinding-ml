"""Run the full pathfinding-ml pipeline end to end and report the honest results.

    python experiments/run_experiment.py            # defaults
    python experiments/run_experiment.py --n 250    # more mazes

Pipeline: generate mazes -> assemble dataset (whole-maze holdout) -> train model
-> Axis 1: heuristic quality on held-out *cells* (proxy metric + admissibility)
-> Axis 2: search benchmark on held-out *mazes* (nodes expanded vs optimality gap)
-> print a summary and save two charts to experiments/figures/.

Everything is evaluated on held-out data, so the numbers reflect generalisation, not
memorisation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run directly -> put the project root on the import path (see inspect_mazes.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # write files, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pathfinding.config import ExperimentConfig  # noqa: E402
from pathfinding.data.dataset import assemble  # noqa: E402
from pathfinding.data.features import FEATURE_FUNCS  # noqa: E402
from pathfinding.evaluation.analysis import gap_distribution, summarize_by_style  # noqa: E402
from pathfinding.evaluation.heuristic_quality import evaluate  # noqa: E402
from pathfinding.evaluation.search_benchmark import run_benchmark, summarize  # noqa: E402
from pathfinding.maze.generator import make_mazes  # noqa: E402
from pathfinding.model.baseline import ManhattanBaseline  # noqa: E402
from pathfinding.model.train import train_model  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=1000, help="number of mazes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--size-min", type=int, default=15)
    p.add_argument("--size-max", type=int, default=45)
    p.add_argument("--structured-fraction", type=float, default=0.5)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument(
        "--features",
        type=str,
        default=None,
        help="comma-separated feature names (default: the six baseline features)",
    )
    return p.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    feature_names = None
    if args.features:
        feature_names = [f.strip() for f in args.features.split(",") if f.strip()]
        unknown = [f for f in feature_names if f not in FEATURE_FUNCS]
        if unknown:
            raise SystemExit(
                f"unknown feature(s): {unknown}\navailable: {sorted(FEATURE_FUNCS)}"
            )
    kwargs = dict(
        n=args.n,
        seed=args.seed,
        test_fraction=args.test_fraction,
        size_min=args.size_min,
        size_max=args.size_max,
        structured_fraction=args.structured_fraction,
        window=args.window,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
    )
    if feature_names is not None:
        kwargs["feature_names"] = feature_names
    return ExperimentConfig(**kwargs)


def print_quality(model_q, base_q) -> None:
    print("\n--- Axis 1: heuristic quality on held-out cells (proxy metric) ---")
    print(
        f"  learned  : MAE {model_q.mae:6.3f}  RMSE {model_q.rmse:6.3f}  "
        f"overestimates {model_q.frac_overestimated:5.1%} of cells "
        f"(max +{model_q.max_overestimate:.1f})"
    )
    print(
        f"  manhattan: MAE {base_q.mae:6.3f}  RMSE {base_q.rmse:6.3f}  "
        f"overestimates  0.0% (admissible by construction)"
    )


def print_summary(summary) -> None:
    print("\n--- Axis 2: search on held-out mazes ---")
    print(f"  {'algorithm + heuristic':26}{'mean nodes':>12}{'opt. gap':>11}{'found':>8}")
    for (algo, heur), s in sorted(summary.items()):
        print(
            f"  {algo + ' + ' + heur:26}{s['mean_nodes_expanded']:>12.1f}"
            f"{s['mean_optimality_gap'] * 100:>10.2f}%{s['found_rate'] * 100:>7.0f}%"
        )

    am, al = summary.get(("astar", "manhattan")), summary.get(("astar", "learned"))
    if am and al:
        change = al["mean_nodes_expanded"] / am["mean_nodes_expanded"] - 1
        print(
            f"\n  A* learned vs A* manhattan: nodes {change:+.1%}, "
            f"optimality gap {al['mean_optimality_gap']:.2%} "
            f"(manhattan stays optimal at {am['mean_optimality_gap']:.2%})"
        )
        print("  (Read honestly: fewer nodes is only a win if the optimality gap stays small.)")


def print_gap_distribution(rows) -> None:
    print("\n--- Optimality-gap distribution (is it uniform, or a bad tail?) ---")
    for algo, heur in [("astar", "learned"), ("greedy", "learned"), ("greedy", "manhattan")]:
        d = gap_distribution(rows, algo, heur)
        if d["n"]:
            print(
                f"  {algo + ' + ' + heur:24} optimal {d['frac_optimal']:5.0%}  "
                f"median {d['median'] * 100:5.2f}%  p90 {d['p90'] * 100:6.2f}%  "
                f"max {d['max'] * 100:6.2f}%"
            )


def print_by_style(rows) -> None:
    print("\n--- By maze type (within-group: don't pool what you can separate) ---")
    for style, summary in summarize_by_style(rows).items():
        n = next(iter(summary.values()))["n"] if summary else 0
        print(f"  [{style}]  ({int(n)} mazes)")
        for (algo, heur), s in sorted(summary.items()):
            print(
                f"    {algo + ' + ' + heur:24}{s['mean_nodes_expanded']:>10.1f} nodes"
                f"{s['mean_optimality_gap'] * 100:>8.2f}% gap"
            )


def plot_tradeoff(summary, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for (algo, heur), s in summary.items():
        label = "dijkstra (blind)" if algo == "dijkstra" else f"{algo} + {heur}"
        ax.scatter(s["mean_nodes_expanded"], s["mean_optimality_gap"] * 100, s=90)
        ax.annotate(
            label,
            (s["mean_nodes_expanded"], s["mean_optimality_gap"] * 100),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=9,
        )
    ax.set_xlabel("mean nodes expanded  (← fewer = faster)")
    ax.set_ylabel("mean optimality gap (%)  (lower = better paths)")
    ax.set_title("Speed vs optimality tradeoff (held-out mazes)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_quality(model_q, base_q, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    maes = [model_q.mae, base_q.mae]
    bars = ax.bar(["learned model", "manhattan baseline"], maes, color=["#4C72B0", "#999999"])
    ax.set_ylabel("MAE of cost-to-go (held-out cells)")
    ax.set_title(
        "Prediction error (lower = better)\n"
        f"learned overestimates {model_q.frac_overestimated:.0%} of cells "
        f"(admissible heuristics: 0%)"
    )
    for bar, value in zip(bars, maes):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    config = build_config(parse_args())
    rng = np.random.default_rng(config.seed)

    print(
        f"generating {config.n} mazes (sizes {config.size_min}-{config.size_max}, "
        f"structured_fraction={config.structured_fraction})..."
    )
    print(f"features ({len(config.feature_names)}): {', '.join(config.feature_names)}")
    mazes = make_mazes(
        config.n,
        rng,
        size_range=config.size_range,
        structured_fraction=config.structured_fraction,
    )

    print("assembling dataset (whole-maze holdout)...")
    data = assemble(
        mazes,
        test_fraction=config.test_fraction,
        rng=rng,
        feature_names=config.feature_names,
        window=config.window,
    )
    print(
        f"  train: {len(data.y_train)} cells / {len(data.train_maze_ids)} mazes | "
        f"test: {len(data.y_test)} cells / {len(data.test_maze_ids)} mazes"
    )

    print("training model...")
    model = train_model(
        data.X_train,
        data.y_train,
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
    )

    model_q = evaluate(data.y_test, model.predict(data.X_test))
    base_q = evaluate(data.y_test, ManhattanBaseline(config.feature_names).predict(data.X_test))
    print_quality(model_q, base_q)

    test_mazes = [mazes[i] for i in data.test_maze_ids]
    print(f"\nbenchmarking search on {len(test_mazes)} held-out mazes...")
    rows = run_benchmark(
        test_mazes, model, feature_names=config.feature_names, window=config.window
    )
    summary = summarize(rows)
    print_summary(summary)
    print_gap_distribution(rows)
    print_by_style(rows)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(summary, FIG_DIR / "tradeoff.png")
    plot_quality(model_q, base_q, FIG_DIR / "heuristic_quality.png")
    print(f"\ncharts written to {FIG_DIR}")


if __name__ == "__main__":
    main()
