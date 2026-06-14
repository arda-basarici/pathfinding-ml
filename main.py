"""Convenience entry point — delegates to the experiment CLI.

The real tools live one level down:
    python experiments/run_experiment.py ...   # run a single experiment
    python generate_report.py                  # build the PDF report
This shim just lets `python main.py ...` run an experiment too.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.run_experiment import main  # noqa: E402

if __name__ == "__main__":
    main()
