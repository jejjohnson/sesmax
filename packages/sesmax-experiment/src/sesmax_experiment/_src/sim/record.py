"""The record of a run, one line of numbers per chunk."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from sesmax_experiment._src.run.frame import ChunkFrame


@runtime_checkable
class MetricWriter(Protocol):
    """Takes the numbers of a step, for a file or an experiment tracker."""

    def write_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        """Take the numbers of one step."""
        ...

    def flush(self) -> None:
        """Make sure what was written can be read."""
        ...

    def close(self) -> None:
        """Nothing more will be written."""
        ...


class JsonLinesWriter:
    """Writes a line of JSON per step; a number that is not finite is ``null``.

    Args:
        path: The file, which is emptied.
    """

    def __init__(self, path: Path) -> None:
        self._file = path.open("w")

    def write_scalars(self, step: int, scalars: Mapping[str, float]) -> None:
        """Append one line, with the step first."""
        values = {k: v if math.isfinite(v) else None for k, v in scalars.items()}
        self._file.write(json.dumps({"step": step, **values}) + "\n")

    def flush(self) -> None:
        """Write what is buffered."""
        self._file.flush()

    def close(self) -> None:
        """Close the file."""
        self._file.close()


class HealthRecord:
    """The numbers of each chunk: its times, its speed and what the monitors saw.

    ``wall_seconds`` is the integration, ``host_seconds`` everything else since the
    chunk before: the monitors, the snapshots and the checkpoints. ``eta_seconds``
    takes the speed since the end of the first chunk, which holds the compilation.

    Args:
        writer: Where they go.
        t1: The end of the run.
        first_step: The chunks of the whole run before this one.
        clock: The clock, for tests.
    """

    def __init__(
        self,
        writer: MetricWriter,
        *,
        t1: float,
        first_step: int = 0,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._writer = writer
        self._t1 = t1
        self._first_step = first_step
        self._clock = clock
        self._last = clock()
        # The wall time and the simulation time the speed is measured from.
        self._origin: tuple[float, float] | None = None

    def __call__(self, frame: ChunkFrame) -> None:
        """Write the line of one chunk."""
        info, now = frame.info, self._clock()
        steps = info.stats.get("num_steps")
        scalars = {
            "t": info.t1,
            "wall_seconds": info.wall_seconds,
            "host_seconds": max(now - self._last - info.wall_seconds, 0.0),
        }
        if steps is not None and info.wall_seconds > 0.0:
            scalars["steps_per_second"] = steps / info.wall_seconds
        if self._origin is None:
            elapsed, advanced = now - self._last, info.t1 - info.t0
            self._origin = (now, info.t1)
        else:
            elapsed, advanced = now - self._origin[0], info.t1 - self._origin[1]
        scalars["eta_seconds"] = elapsed * (self._t1 - info.t1) / advanced
        scalars |= {k: float(v) for k, v in info.stats.items()}
        scalars |= {k: float(v) for k, v in frame.metrics.items()}
        self._writer.write_scalars(self._first_step + info.index + 1, scalars)
        self._writer.flush()
        self._last = now
