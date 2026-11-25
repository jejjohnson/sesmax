"""Public surface for sesmax-experiment's runs in chunks.

:func:`run_chunks` integrates a model from one time of a :class:`Timeline` to the
next, and after each chunk runs the same stages: the monitors, a sink for the
snapshots and whatever pipekit operators it is given. The schedules say when a
stage is due, and combine with ``|`` and ``&``.
"""

from __future__ import annotations

from sesmax_experiment._src.run.chunks import ChunkedRun, run_chunks
from sesmax_experiment._src.run.frame import ChunkFrame, RunStopped
from sesmax_experiment._src.run.schedules import (
    Due,
    at_snapshots,
    every_chunks,
    every_sim_seconds,
    every_wall_seconds,
)
from sesmax_experiment._src.run.stages import AdvanceChunk, RunMonitors
from sesmax_experiment._src.run.timeline import Timeline


__all__ = [
    "AdvanceChunk",
    "ChunkFrame",
    "ChunkedRun",
    "Due",
    "RunMonitors",
    "RunStopped",
    "Timeline",
    "at_snapshots",
    "every_chunks",
    "every_sim_seconds",
    "every_wall_seconds",
    "run_chunks",
]
