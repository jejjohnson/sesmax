"""The console script, which says what to install when the extra is missing."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point of ``sesmax-experiment``."""
    try:
        from sesmax_experiment.sim import main as run
    except ImportError as exc:
        sys.exit(str(exc))
    run()
