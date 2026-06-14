"""pathfinding-ml CLI.

Thin entry point — argument parsing only; all real work lives in the ``pathfinding``
package. Subcommands mirror the pipeline stages:

    generate   create a population of mazes and cache them
    label      compute exact cost-to-go labels for the cached mazes
    train      build features, assemble the (whole-maze-split) dataset, fit the model
    evaluate   run heuristic-quality + search-benchmark and emit results

Usage (once implemented):
    python main.py generate --n 200 --size 30
    python main.py train
    python main.py evaluate
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser with one subparser per pipeline stage."""
    parser = argparse.ArgumentParser(prog="pathfinding-ml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate and cache mazes")
    gen.add_argument("--n", type=int, default=200, help="number of mazes")
    gen.add_argument("--size", type=int, default=30, help="grid side length")
    gen.add_argument("--seed", type=int, default=0)

    sub.add_parser("label", help="compute exact cost-to-go labels")
    sub.add_parser("train", help="assemble dataset and train the model")
    sub.add_parser("evaluate", help="run heuristic quality + search benchmark")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatch. Wires args to the package; returns a process exit code."""
    args = build_parser().parse_args(argv)
    raise NotImplementedError(f"command {args.command!r} not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
