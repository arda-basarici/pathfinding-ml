"""ExperimentConfig — the single source of truth for one experiment run.

Every knob a variant might change lives here, so a run is fully described by its config
(plus a seed and the code version). A new knob is a new field; nothing else moves. The
config is what gets saved with each run, so results are reproducible and comparable.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from .data.features import DEFAULT_FEATURES


@dataclass
class ExperimentConfig:
    # data / mazes
    n: int = 1000
    seed: int = 0
    test_fraction: float = 0.25
    size_min: int = 15
    size_max: int = 45
    structured_fraction: float = 0.5
    # cross-regime: if both set, train on one maze style and test on another
    # ("scattered" | "structured" | "mixed"). None => normal same-distribution split.
    train_style: str | None = None
    test_style: str | None = None
    # features
    feature_names: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    window: int = 2
    # model
    learning_rate: float = 0.1
    max_iter: int = 300

    @property
    def size_range(self) -> tuple[int, int]:
        return (self.size_min, self.size_max)

    def to_dict(self) -> dict:
        """Plain dict for persisting alongside a run's results."""
        return dataclasses.asdict(self)
