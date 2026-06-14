"""Persist experiment runs so variants are saved, comparable, and reproducible.

Each run writes one ``record.json`` (its config + metrics + git hash + timestamp) into
its own timestamped directory under ``experiments/runs/`` — never overwriting a previous
run. The report can then cite "config X at commit Y," and comparing variants is reading
two records instead of scrolling terminal output.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime
from pathlib import Path


def git_hash() -> str:
    """Short current commit hash, or 'unknown' if not in a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _key(pair: tuple[str, str]) -> str:
    """Tuple key (algorithm, heuristic) -> JSON-safe 'algorithm+heuristic'."""
    return f"{pair[0]}+{pair[1]}"


def build_record(config, model_q, base_q, summary, gap_dists, by_style) -> dict:
    """Assemble the JSON-serialisable record for a run (no run_id/timestamp yet)."""
    return {
        "git_hash": git_hash(),
        "config": config.to_dict(),
        "axis1_quality": {
            "learned": dataclasses.asdict(model_q),
            "manhattan": dataclasses.asdict(base_q),
        },
        "axis2_summary": {_key(k): v for k, v in summary.items()},
        "gap_distribution": gap_dists,
        "by_style": {
            style: {_key(k): v for k, v in s.items()} for style, s in by_style.items()
        },
    }


def save_run(runs_root: Path, record: dict, run_id: str | None = None) -> Path:
    """Write ``record`` to a fresh run directory; return its path. Never overwrites."""
    now = datetime.now()
    if run_id is None:
        cfg = record.get("config", {})
        run_id = (
            now.strftime("%Y%m%d-%H%M%S")
            + f"_n{cfg.get('n', '?')}_seed{cfg.get('seed', '?')}_{record.get('git_hash', '')}"
        )

    full = {"run_id": run_id, "timestamp": now.isoformat(timespec="seconds"), **record}
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "record.json").write_text(json.dumps(full, indent=2, default=str))
    return run_dir
