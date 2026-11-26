"""The stages of a chunk that are sesmax's: advance the model, ask the monitors."""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from collections.abc import Sequence
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from pipekit_ops import Operator, StatefulOperator

from sesmax_core import Model
from sesmax_experiment._src.monitor import ChunkInfo, Monitor
from sesmax_experiment._src.run.frame import ChunkFrame, RunStopped
from sesmax_experiment._src.run.timeline import Timeline


# A ratio this close above a whole number of steps is that number.
_STEP_SLACK = 1e-9


def n_steps(t0: float, t1: float, dt: float) -> int:
    """The steps of ``dt`` between two times, the last of which may be shorter.

    ``(t1 - t0) / dt`` rounded up, unless it is a whole number but for rounding.

    Args:
        t0: Start.
        t1: End.
        dt: The step, positive.

    Returns:
        At least one.
    """
    return max(1, math.ceil((t1 - t0) / dt - _STEP_SLACK))


def solver_stats(solution: dfx.Solution) -> dict[str, Any]:
    """A solution's statistics as host numbers, as far as there are any."""
    out: dict[str, Any] = {}
    for key in ("num_steps", "num_accepted_steps", "num_rejected_steps"):
        value = (solution.stats or {}).get(key)
        if value is not None:
            with contextlib.suppress(TypeError, ValueError):
                out[key] = int(np.asarray(value))
    with contextlib.suppress(Exception):
        out["result_successful"] = bool(solution.result == dfx.RESULTS.successful)
    return out


class AdvanceChunk(StatefulOperator):
    """Integrate a model from one edge of a timeline to the next.

    The carried state is the index of the chunk. With a fixed ``dt`` the number of
    steps is stated (:func:`n_steps`), because ``(t1 - t0) / dt`` rounds up in some
    chunks and down in others, and the discretisation would depend on where the
    chunks fall. A ``stepsize_controller`` sets its own steps.

    Args:
        model: The model.
        timeline: The times.
        dt: The step, or the first one of a ``stepsize_controller``.
        integrate_kwargs: Passed to ``Model.integrate``.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml = True

    def __init__(
        self,
        model: Model,
        timeline: Timeline,
        dt: float,
        integrate_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.timeline = timeline
        self.dt = dt
        self.integrate_kwargs = dict(integrate_kwargs or {})

    def _settings(self, t0: float, t1: float) -> dict[str, Any]:
        settings = dict(self.integrate_kwargs)
        if "stepsize_controller" in settings:
            return {"dt": self.dt, **settings}
        steps = n_steps(t0, t1, self.dt)
        settings["stepsize_controller"] = dfx.StepTo(ts=jnp.linspace(t0, t1, steps + 1))
        settings["max_steps"] = steps
        return {"dt": None, **settings}

    def _apply(self, frame: ChunkFrame, index: int) -> tuple[ChunkFrame, int]:
        t0, t1 = (float(t) for t in self.timeline.edges[index : index + 2])
        start = time.perf_counter()
        solution = self.model.integrate(
            frame.state, t0, t1, saveat=dfx.SaveAt(t1=True), **self._settings(t0, t1)
        )
        state = jax.block_until_ready(jtu.tree_map(lambda x: x[-1], solution.ys))
        info = ChunkInfo(
            index=index,
            n_chunks=self.timeline.n_chunks,
            t0=t0,
            t1=t1,
            wall_seconds=time.perf_counter() - start,
            is_snapshot=index + 1 in self.timeline.snapshots,
            stats=solver_stats(solution),
        )
        return ChunkFrame(state=state, info=info), index + 1


class RunMonitors(Operator):
    """Ask every monitor about a frame, and stop the run if one wants to.

    Every monitor is asked before the run stops, so the metrics of a stopped chunk
    are complete. The first reason counts, and the stop is resumable only if every
    monitor that asks for one says so.

    Args:
        model: The model the monitors see.
        monitors: The monitors.

    Raises:
        RunStopped: If a monitor asks to terminate.
    """

    __config_mixin_auto__ = False
    forbid_in_yaml = True

    def __init__(self, model: Model, monitors: Sequence[Monitor]) -> None:
        self.model = model
        self.monitors = tuple(monitors)

    def _apply(self, frame: ChunkFrame) -> ChunkFrame:
        metrics, messages = dict(frame.metrics), list(frame.messages)
        reason: str | None = None
        resumable = True
        for monitor in self.monitors:
            verdict = monitor.on_chunk_end(self.model, frame.state, frame.info)
            metrics |= {f"{monitor.name}.{k}": v for k, v in verdict.metrics.items()}
            messages += [f"[{monitor.name}] {text}" for text in verdict.messages]
            if verdict.terminate:
                reason = reason or f"monitor {monitor.name!r}: {verdict.reason}"
                resumable = resumable and verdict.resumable
        frame = dataclasses.replace(frame, metrics=metrics, messages=tuple(messages))
        if reason is not None:
            raise RunStopped(reason, frame, resumable=resumable)
        return frame
