"""Eyeball what the generators actually produce.

Run:  python experiments/inspect_mazes.py
Generates a few small mazes through the real make_mazes pipeline and prints them, so
you can confirm by eye that structured mazes look like corridor mazes, scattered ones
look scattered, and start (S) / goal (G) land on passable cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run directly (python experiments/inspect_mazes.py), Python puts experiments/ on the
# import path, not the project root — so 'pathfinding' isn't found. Add the root (this
# file's grandparent) so the script works from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402  (import after the path bootstrap, intentional)

from pathfinding.maze.generator import make_mazes  # noqa: E402
from pathfinding.maze.grid import Maze  # noqa: E402
from pathfinding.maze.render import to_ascii  # noqa: E402


def show(title: str, mazes: list[Maze]) -> None:
    for i, maze in enumerate(mazes, start=1):
        g = maze.grid
        print(
            f"\n--- {title} #{i}  "
            f"{g.height}x{g.width}  blocked={len(g.blocked)}  "
            f"start={maze.start}  goal={maze.goal} ---"
        )
        print(to_ascii(maze))


def main() -> None:
    rng = np.random.default_rng(0)
    # Small sizes so they fit a terminal; the pipeline is identical to training.
    show("structured", make_mazes(2, rng, size_range=(11, 15), structured_fraction=1.0))
    show("scattered", make_mazes(2, rng, size_range=(11, 15), structured_fraction=0.0))
    # A couple at default settings to see the real size/style mix.
    show("default mix", make_mazes(2, rng))


if __name__ == "__main__":
    main()
