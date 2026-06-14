"""report_data.py — the experiments the report is built on.

`gather()` defines the handful of runs the report needs and returns their records. Each
is computed by `pathfinding.experiment.run` (deterministic from its seed) and cached on
disk by config signature, so the first report build computes them (a few minutes) and
later builds are instant. Delete `.report_cache/` to force a recompute.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pathfinding.config import ExperimentConfig
from pathfinding.experiment import run

CACHE = Path(__file__).resolve().parent / ".report_cache"

# The +global feature set (the six baseline features plus the regime-tag feature).
GLOBAL_FEATURES = [
    "manhattan_to_goal", "euclidean_to_goal", "row_delta", "col_delta",
    "local_obstacle_density", "blocked_neighbors", "global_obstacle_density",
]


def _signature(config: ExperimentConfig) -> str:
    return hashlib.md5(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def get(config: ExperimentConfig) -> dict:
    """Record for one config — from cache if present, else computed and cached."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{_signature(config)}.json"
    if path.exists():
        return json.loads(path.read_text())
    record = run(config)
    path.write_text(json.dumps(record, default=str))
    return record


def gather(n: int = 1000, seed: int = 0) -> dict:
    """All records the report needs, keyed by a short label."""
    def C(**kw) -> ExperimentConfig:
        return ExperimentConfig(**{"n": n, "seed": seed, **kw})

    return {
        "baseline": get(C()),                                   # six features, mixed
        "global": get(C(feature_names=GLOBAL_FEATURES)),        # + regime tag, mixed
        "pure_scattered": get(C(structured_fraction=0.0)),      # baseline, scattered only
        "pure_structured": get(C(structured_fraction=1.0)),     # baseline, structured only
        "transfer_s2t": get(C(train_style="scattered", test_style="structured")),
        "transfer_t2s": get(C(train_style="structured", test_style="scattered")),
        "min2": get(C(feature_names=["manhattan_to_goal", "global_obstacle_density"])),
        "min3": get(C(feature_names=["manhattan_to_goal", "euclidean_to_goal", "global_obstacle_density"])),
        # robustness: the headline configs across seeds (seed 0 reuses the cached runs above)
        "baseline_seeds": [get(C(seed=s)) for s in (0, 1, 2)],
        "global_seeds": [get(C(seed=s, feature_names=GLOBAL_FEATURES)) for s in (0, 1, 2)],
        "mix": {
            sf: get(C(structured_fraction=sf, feature_names=GLOBAL_FEATURES))
            for sf in (0.0, 0.25, 0.5, 0.75, 1.0)
        },
    }
