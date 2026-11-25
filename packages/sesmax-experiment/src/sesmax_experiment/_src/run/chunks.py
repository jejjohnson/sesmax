"""A run of a model in chunks, as a pipeline of pipekit operators."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jaxtyping import PyTree
from pipekit_cycle import Cycle
from pipekit_ops import Branch, Operator, Sequential, Sink, Tap

from sesmax_core import Model
from sesmax_experiment._src.monitor import ChunkInfo, Monitor
from sesmax_experiment._src.run.frame import ChunkFrame
from sesmax_experiment._src.run.schedules import at_snapshots
from sesmax_experiment._src.run.stages import AdvanceChunk, RunMonitors
from sesmax_experiment._src.run.timeline import Timeline


@dataclasses.dataclass(frozen=True)
class ChunkedRun:
    """What a run returns.

    Args:
        frame: The last frame: the final state, and what is known of the last chunk.
        snapshots: The frames at the snapshots, the start included, if no
            ``on_snapshot`` took them.
    """

    frame: ChunkFrame
    snapshots: tuple[ChunkFrame, ...]

    @property
    def state(self) -> PyTree:
        """The final state."""
        return self.frame.state


def run_chunks(
    model: Model,
    state0: PyTree,
    timeline: Timeline,
    *,
    dt: float,
    monitors: Sequence[Monitor] = (),
    on_snapshot: Callable[[ChunkFrame], Any] | None = None,
    stages: Sequence[Operator] = (),
    spec: Any = None,
    **integrate_kwargs: Any,
) -> ChunkedRun:
    """Run a model in chunks, looking at the state between them.

    Each chunk is one compiled integration and then the same stages, in order:

    1. advance the model to the next time of the timeline;
    2. ask the monitors, which may stop the run before anything is written;
    3. at a snapshot, hand the frame to ``on_snapshot``;
    4. the other ``stages``.

    It writes nothing itself. A stage is any pipekit operator on a
    :class:`ChunkFrame`, for example a ``Branch`` of a schedule on a ``Sink``, or a
    ``Retry`` around one.

    Args:
        model: The model.
        state0: Its initial state; the boundary conditions are applied to it.
        timeline: The chunks and the snapshots.
        dt: The time step. Without a ``stepsize_controller`` each chunk takes
            ``(t1 - t0) / dt`` equal steps, rounded up.
        monitors: Asked after each chunk, and at the start and the end of the run.
        on_snapshot: Takes each snapshot, the start included. Without it the run
            keeps them in memory and returns them.
        stages: More operators on each frame.
        spec: Handed to the monitors at the start.
        **integrate_kwargs: Passed to ``Model.integrate`` (``solver``,
            ``stepsize_controller``, …).

    Returns:
        The last frame, and the snapshots unless ``on_snapshot`` took them.

    Raises:
        RunStopped: If a monitor stops the run.

    Examples:
        >>> import jax.numpy as jnp
        >>> from sesmax_core.testing import L96State, Lorenz96
        >>> from sesmax_experiment.monitor import NonFiniteMonitor
        >>> from sesmax_experiment.run import Timeline, run_chunks
        >>> model = Lorenz96.create(F=8.0)
        >>> state = L96State(x=jnp.linspace(-2.0, 2.0, 12))
        >>> run = run_chunks(
        ...     model,
        ...     state,
        ...     Timeline.build(0.0, 1.0, chunk=0.25, snapshot=0.5),
        ...     dt=0.01,
        ...     monitors=[NonFiniteMonitor()],
        ... )
        >>> [frame.info.t1 for frame in run.snapshots]
        [0.0, 0.5, 1.0]
        >>> run.frame.info.stats["num_steps"]
        25
    """
    t0 = float(timeline.edges[0])
    state0 = jax.block_until_ready(model.apply_boundary_conditions(state0))
    first = ChunkFrame(
        state=state0,
        info=ChunkInfo(
            index=-1,
            n_chunks=timeline.n_chunks,
            t0=t0,
            t1=t0,
            wall_seconds=0.0,
            is_snapshot=0 in timeline.snapshots,
        ),
    )
    kept: list[ChunkFrame] = []
    take = kept.append if on_snapshot is None else on_snapshot
    last = [first]
    chunk = Sequential(
        [
            AdvanceChunk(model, timeline, dt, integrate_kwargs),
            Tap(lambda frame: last.__setitem__(0, frame), name="last"),
            RunMonitors(model, monitors),
            Branch(at_snapshots(), Sink(take, name="on_snapshot")),
            *stages,
        ]
    )
    for monitor in monitors:
        monitor.on_run_start(model, state0, spec=spec)
    ok = False
    try:
        if first.info.is_snapshot:
            take(first)
        frame, _ = Cycle(step_op=chunk, n_steps=timeline.n_chunks)(first, 0)
        ok = True
    finally:
        for monitor in monitors:
            monitor.on_run_end(model, last[0].state, ok=ok)
    return ChunkedRun(frame=frame, snapshots=tuple(kept))
