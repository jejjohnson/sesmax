"""The console script, which says what to install when the extra is missing."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point of ``sesmax-ocean``."""
    try:
        from sesmax_ocean._src.basins.commands import main as run
    except ImportError as exc:
        sys.exit(f"{exc}\nsesmax-ocean needs: pip install 'sesmax-ocean[data]'")
    run()
