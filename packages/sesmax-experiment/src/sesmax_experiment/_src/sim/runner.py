"""Run a spec and write its artifacts: simulate, spinup and restart."""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import numpy as np
import paramax
from loguru import logger
from pipekit_ops import Operator, Tap

from sesmax_core import Model, io
from sesmax_core.component import BuiltRun, Component
from sesmax_core.fields import values
from sesmax_experiment._src.components import find_model
from sesmax_experiment._src.monitor import Monitor, default_monitors
from sesmax_experiment._src.run.chunks import run_chunks
from sesmax_experiment._src.run.frame import ChunkFrame, RunStopped
from sesmax_experiment._src.run.schedules import Due
from sesmax_experiment._src.run.timeline import Timeline
from sesmax_experiment._src.sim.artifacts import (
    SnapshotStore,
    flat_diagnostics,
    manifest,
    run_attrs,
    write_json,
)
from sesmax_experiment._src.sim.checkpoints import (
    DIRECTORY,
    Checkpointing,
    CheckpointStore,
    checkpoint_stage,
    remove_stale,
    resolve,
)
from sesmax_experiment._src.sim.checks import preflight
from sesmax_experiment._src.sim.progress import (
    start_run_log,
    stop_run_log,
)
from sesmax_experiment._src.sim.record import HealthRecord, JsonLinesWriter
from sesmax_experiment._src.sim.report import chunk_line
from sesmax_experiment._src.sim.spec import RunSpec, dump_yaml
from sesmax_experiment._src.sim.units import (
    field_units,
    format_time_seconds,
    format_wallclock,
)


Mode = Literal["simulate", "spinup", "restart"]
Status = Literal["finished", "stopped"]


class IntegrationDivergedError(RuntimeError):
    """A run was halted, because its state stopped making sense.

    The most common cause is a time step too large for the grid spacing and the
    fastest wave of the model.
    """


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    """What a run wrote.

    Args:
        output_dir: The directory.
        status: ``"finished"``, or ``"stopped"`` if a monitor stopped it resumably.
        snapshots_path: The snapshots, or ``None`` if there are none.
        final_state_path: The final state, or ``None`` if it stopped before the end.
        metrics_path: The metrics, or ``None`` if there are none.
        checkpoint_path: The newest checkpoint, which a restart reads, or ``None``.
        wallclock_seconds: How long the integration took.
        n_steps: The steps the solver took.
        state: The last state.
    """

    output_dir: Path
    status: Status
    snapshots_path: Path | None
    final_state_path: Path | None
    metrics_path: Path | None
    checkpoint_path: Path | None
    wallclock_seconds: float
    n_steps: int
    state: Any


def build_run(
    spec: RunSpec, among: Mapping[str, Component] | None = None
) -> tuple[Component, BuiltRun]:
    """Build the model and the initial state of a spec.

    Args:
        spec: The spec.
        among: The components to look in; the installed ones by default.

    Returns:
        The component that offers the model, and the run it built.
    """
    component, build = find_model(spec.model.name, among)
    data = spec.to_dict()
    return component, build({key: data[key] for key in ("scenario", "model")})


def timeline_of(
    spec: RunSpec, *, only_final: bool = False, diagnostics_per_save: int = 1
) -> Timeline:
    """The times of a spec: a snapshot every ``save_interval``, and the end.

    Args:
        spec: The spec.
        only_final: One snapshot, the end, beside the start.
        diagnostics_per_save: Chunks between two snapshots.

    Returns:
        The timeline.
    """
    times = spec.timestepping
    interval = times.t1 - times.t0 if only_final else times.save_interval
    timeline = Timeline.build(times.t0, times.t1, chunk=interval, snapshot=interval)
    return timeline.subdivided(max(diagnostics_per_save, 1))


def _not_finite(state: Any) -> list[str]:
    """The fields that hold a NaN or an infinity, with how many."""
    bad = []
    for field in dataclasses.fields(state):
        array = np.asarray(values(getattr(state, field.name)))
        if count := int(np.sum(~np.isfinite(array))):
            bad.append(f"{field.name} ({count}/{array.size} non-finite)")
    return bad


def _assert_finite(frame: ChunkFrame) -> None:
    """Refuse a state that is not finite, whatever the monitors are."""
    if bad := _not_finite(frame.state):
        raise RunStopped("non-finite values in " + "; ".join(bad), frame)


def _diverged(mode: Mode, cause: Exception) -> IntegrationDivergedError:
    first = ""
    if isinstance(cause, RunStopped) and cause.frame.info.index < 0:
        first = "  The initial state was already non-finite.\n"
    return IntegrationDivergedError(
        f"{mode} integration halted: {cause}\n{first}"
        "  If the state became non-finite, the most common cause is a CFL "
        "violation: a time step too large for the grid spacing and the fastest "
        "wave. Refusing to write artifacts."
    )


@dataclasses.dataclass
class _Observers:
    """What looks at the frames of a run, beside its monitors."""

    model: Model
    units: dict[str, str]
    log: Any
    record: HealthRecord
    store: SnapshotStore | None
    steps: int = 0
    last: int = -1

    def chunk(self, frame: ChunkFrame) -> None:
        """After every chunk: the log and the record, once."""
        if frame.info.index <= self.last:
            return
        self.last = frame.info.index
        self.steps += int(frame.info.stats.get("num_steps", 0))
        self.line(frame)
        self.record(frame)

    def line(self, frame: ChunkFrame) -> None:
        """One line of the log."""
        self.log.debug(chunk_line(self.model, frame, self.units))

    def snapshot(self, frame: ChunkFrame) -> None:
        """At a snapshot, the start included."""
        _assert_finite(frame)
        if frame.info.index < 0:
            self.line(frame)
        if self.store is not None:
            self.store(frame)


def _integrate(
    spec: RunSpec,
    model: Model,
    state0: Any,
    timeline: Timeline,
    *,
    mode: Mode,
    monitors: Sequence[Monitor],
    seen: _Observers,
    due: Due | None,
    checkpoint: Callable[[ChunkFrame], Any],
) -> tuple[Any, RunStopped | None]:
    """The chunks of a run: its last state, and the stop if it was resumable."""
    stages: list[Operator] = [Tap(seen.chunk)]
    if due is not None:
        stages.append(checkpoint_stage(checkpoint, due, seen.log.warning))
    if 0 not in timeline.snapshots:
        start = float(timeline.edges[0])
        seen.log.debug(f"starting between two snapshots, at t={start:g}")
    try:
        run = run_chunks(
            model,
            state0,
            timeline,
            dt=spec.timestepping.dt,
            monitors=monitors,
            on_snapshot=seen.snapshot,
            stages=stages,
            spec=spec,
        )
    except RunStopped as stop:
        seen.chunk(stop.frame)
        bad = _not_finite(stop.frame.state)
        if bad or not stop.resumable:
            raise _diverged(mode, stop) from stop
        if stop.frame.info.index + 1 == timeline.n_chunks:
            # Its time was up as it ended: there is nothing to resume.
            seen.snapshot(stop.frame)
            return stop.frame.state, None
        checkpoint(stop.frame)
        return stop.frame.state, stop
    except eqx.EquinoxRuntimeError as exc:
        # A guard inside the tendency halted the step that went wrong.
        raise _diverged(mode, exc) from exc
    return run.state, None


def _metrics(
    component: Component, built: BuiltRun, state: Any, run: dict[str, Any]
) -> dict[str, Any]:
    """The numbers about the final state, and about the run."""
    out: dict[str, Any] = {}
    try:
        out |= flat_diagnostics(paramax.unwrap(built.model).diagnose(state))
    except Exception as exc:
        logger.warning("model.diagnose failed: {}", exc)
    if component.evaluate is not None:
        try:
            out |= component.evaluate(built.model, state)
        except Exception as exc:
            logger.warning("the component's metrics failed: {}", exc)
    return out | run


def _clear(output_dir: Path) -> None:
    """Remove what an earlier run left, which would be taken for this one's."""
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_stale(output_dir)
    if (output_dir / "final_state.zarr").exists():
        shutil.rmtree(output_dir / "final_state.zarr")
    for name in ("metrics.json", "manifest.json"):
        (output_dir / name).unlink(missing_ok=True)


def _run(
    spec: RunSpec,
    output_dir: Path,
    *,
    mode: Mode,
    built: BuiltRun,
    component: Component,
    state0: Any,
    whole: Timeline,
    timeline: Timeline,
    monitors: Sequence[Monitor] | None,
    checkpointing: Checkpointing | None,
) -> SimulationResult:
    """Integrate, and write everything a run writes."""
    model, only_final = built.model, mode == "spinup"
    logger.info("{}: {} x {}", mode, spec.scenario.name, spec.model.name)
    logger.info("output_dir={}", output_dir)
    preflight(spec, model, component)
    _clear(output_dir)

    t0, t1 = float(timeline.edges[0]), float(timeline.edges[-1])
    first_step = whole.n_chunks - timeline.n_chunks
    attrs = run_attrs(spec, mode=mode, coordinates=built.coordinates, t0=t0)
    snapshots = SnapshotStore(output_dir, attrs)
    writes_snapshots = spec.output.write_snapshots and not only_final
    writes_metrics = spec.output.write_metrics and not only_final
    keep = checkpointing or Checkpointing(due=Due(lambda frame: False))
    checkpoints = CheckpointStore(
        output_dir / DIRECTORY, keep_last=keep.keep_last, keep_every=keep.keep_every
    )

    def checkpoint(frame: ChunkFrame) -> None:
        _assert_finite(frame)
        step, t = first_step + frame.info.index + 1, frame.info.t1
        path = checkpoints.save(frame.state, step=step, time=t, attrs=attrs)
        run_log.log.debug(f"checkpoint at t={format_time_seconds(t)}: {path.name}")

    run_log = start_run_log(output_dir, label=f"sesmax-experiment/{mode}")
    writer = JsonLinesWriter(output_dir / "metrics.jsonl")
    seen = _Observers(
        model=model,
        units=field_units(
            [field.name for field in dataclasses.fields(state0)],
            component.field_units.get(spec.model.name, {}),
            nondimensional=built.coordinates == "nondimensional",
        ),
        log=run_log.log,
        record=HealthRecord(writer, t1=t1, first_step=first_step),
        store=snapshots if writes_snapshots else None,
    )
    run_log.log.debug(
        f"integration starting: scenario={spec.scenario.name} "
        f"model={spec.model.name} t0={format_time_seconds(t0)} "
        f"t1={format_time_seconds(t1)} dt={spec.timestepping.dt} s "
        f"chunks={timeline.n_chunks} snapshots={len(timeline.snapshots)}"
    )
    message = "finished cleanly"
    try:
        start = time.perf_counter()
        state, stop = _integrate(
            spec,
            model,
            state0,
            timeline,
            mode=mode,
            monitors=default_monitors() if monitors is None else monitors,
            seen=seen,
            due=None if checkpointing is None else checkpointing.due,
            checkpoint=checkpoint,
        )
        wall_seconds = time.perf_counter() - start
        status: Status = "finished" if stop is None else "stopped"
        if stop is None:
            logger.info("integration finished in {}", format_wallclock(wall_seconds))
        else:
            message = f"stopped, and can be resumed: {stop}"
            logger.warning(message)

        metrics_path = final_state_path = None
        if writes_metrics and stop is None:
            run = {
                "wallclock_seconds": wall_seconds,
                "n_steps": seen.steps,
                "t0": t0,
                "t1": t1,
                "save_interval": spec.timestepping.save_interval,
                "mode": mode,
            }
            metrics_path = write_json(
                output_dir / "metrics.json", _metrics(component, built, state, run)
            )
        snapshots_path = snapshots.finish() if writes_snapshots else None
        if stop is None:
            final_state_path = output_dir / "final_state.zarr"
            io.save_dataset(
                io.state_to_dataset(state, time=t1, attrs=attrs),
                final_state_path,
                mode="w",
            )
        dump_yaml(spec, str(output_dir / "resolved.yaml"))
        describe = component.describe_numerics
        write_json(
            output_dir / "manifest.json",
            manifest(
                spec,
                mode=mode,
                status=status,
                reason=None if stop is None else stop.reason,
                t0=t0,
                t1=t1 if stop is None else stop.frame.info.t1,
                wall_seconds=wall_seconds,
                numerics={} if describe is None else describe(model),
            ),
        )
        logger.info("wrote {}", output_dir)
    except BaseException as exc:
        message = f"FAILED: {type(exc).__name__}: {exc}"
        raise
    finally:
        writer.close()
        stop_run_log(run_log, final_message=message)
    return SimulationResult(
        output_dir=output_dir,
        status=status,
        snapshots_path=snapshots_path,
        final_state_path=final_state_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoints.latest(),
        wallclock_seconds=wall_seconds,
        n_steps=seen.steps,
        state=state,
    )


def simulate(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
    monitors: Sequence[Monitor] | None = None,
    checkpointing: Checkpointing | None = None,
    among: Mapping[str, Component] | None = None,
) -> SimulationResult:
    """Run a spec from its initial condition, and write its artifacts.

    ``run.log`` and ``metrics.jsonl``, a line of numbers per chunk, are written as
    the run goes, and snapshots are streamed to ``snapshots.partial.zarr``. A run
    that succeeds leaves ``snapshots.zarr``, ``final_state.zarr``, ``metrics.json``,
    ``resolved.yaml`` and ``manifest.json``; one that fails leaves none of these. One
    that a monitor stops resumably, as ``WatchdogMonitor`` does, leaves a checkpoint
    in place of ``final_state.zarr`` and ``metrics.json``.

    Args:
        spec: The spec, with its debug block merged.
        output_dir: Where to write. Artifacts of an earlier run there are removed.
        diagnostics_per_save: Chunks between two snapshots, each a line in
            ``run.log``.
        monitors: The monitors; ``default_monitors()`` by default.
        checkpointing: When checkpoints are written, and which are kept; by default
            only where a monitor stops the run resumably.
        among: The components to look in; the installed ones by default.

    Returns:
        What was written.

    Raises:
        IntegrationDivergedError: If a monitor or a guard halted the run.
    """
    component, built = build_run(spec, among)
    timeline = timeline_of(
        spec,
        diagnostics_per_save=diagnostics_per_save,
    )
    return _run(
        spec,
        Path(output_dir),
        mode="simulate",
        built=built,
        component=component,
        state0=built.state0,
        whole=timeline,
        timeline=timeline,
        monitors=monitors,
        checkpointing=checkpointing,
    )


def spinup(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
    monitors: Sequence[Monitor] | None = None,
    checkpointing: Checkpointing | None = None,
    among: Mapping[str, Component] | None = None,
) -> SimulationResult:
    """Run a spec for its final state alone, which a restart then starts from.

    Args:
        spec: The spec, with its debug block merged.
        output_dir: Where to write.
        diagnostics_per_save: Chunks of the run, each a line in ``run.log``.
        monitors: The monitors; ``default_monitors()`` by default.
        checkpointing: When checkpoints are written, and which are kept; by default
            only where a monitor stops the run resumably.
        among: The components to look in; the installed ones by default.

    Returns:
        What was written: no snapshots and no metrics.

    Raises:
        IntegrationDivergedError: If a monitor or a guard halted the run.
    """
    component, built = build_run(spec, among)
    timeline = timeline_of(
        spec,
        only_final=True,
        diagnostics_per_save=diagnostics_per_save,
    )
    return _run(
        spec,
        Path(output_dir),
        mode="spinup",
        built=built,
        component=component,
        state0=built.state0,
        whole=timeline,
        timeline=timeline,
        monitors=monitors,
        checkpointing=checkpointing,
    )


def _attr(attrs: Mapping[str, Any], name: str) -> Any:
    """An attribute of a store, in sesmax's spelling or in somax's."""
    return attrs.get(f"sesmax_{name}", attrs.get(f"somax_{name}"))


def _require_coordinates(wanted: str, attrs: Mapping[str, Any], path: Path) -> None:
    """Refuse a state in other units than the run's.

    The class of a state is the same in both, so nothing else would notice, and the
    numbers would be advanced in the wrong units. A store without a mark is older
    than nondimensional runs, so it is in SI.
    """
    found = _attr(attrs, "coordinates") or "si"
    if found != wanted:
        raise ValueError(
            f"the state in {path} is in {found} coordinates and this run is "
            f"{wanted}; its numbers would be advanced in the wrong units. Restart "
            "from a matching state, or convert it with a StateAffine built from "
            "the run's Scales."
        )


def _rest(timeline: Timeline, attrs: Mapping[str, Any]) -> Timeline:
    """What is left of a run's timeline after a checkpoint."""
    raw = _attr(attrs, "sim_t")
    if raw is None:
        return timeline
    try:
        t = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"the state to restart from has a non-numeric sim_t={raw!r}; it must be "
            "a number of seconds"
        ) from exc
    try:
        rest = timeline.after(t)
    except ValueError as exc:
        raise ValueError(
            f"the state to restart from is at t={t}, and the run ends at "
            f"t1={timeline.edges[-1]}: there is nothing left to run. Extend t1."
        ) from exc
    if rest is not timeline:
        logger.info("restart: continuing at t={}", t)
    return rest


def restart(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    restart_from: str | Path,
    diagnostics_per_save: int = 1,
    monitors: Sequence[Monitor] | None = None,
    checkpointing: Checkpointing | None = None,
    among: Mapping[str, Component] | None = None,
) -> SimulationResult:
    """Run a spec from a saved state.

    The model is built from the spec, so a viscosity or a forcing may differ from
    the earlier run's; only the state is loaded. A state that records its time, as
    a checkpoint does, starts the run there, and the chunks and the snapshots after
    it are those of the spec's whole window. A chunk starts from the state alone, so
    with the same spec the run takes the steps it would have taken. A state without
    a time, a ``final_state.zarr``, starts the whole window. Stores written by
    somax load too.

    Args:
        spec: The spec, with its debug block merged.
        output_dir: Where to write, not the directory of ``restart_from``.
        restart_from: A zarr store with a state, a directory of checkpoints, or the
            directory of a run. The newest complete checkpoint is taken.
        diagnostics_per_save: Chunks between two snapshots.
        monitors: The monitors; ``default_monitors()`` by default.
        checkpointing: When checkpoints are written, and which are kept; by default
            only where a monitor stops the run resumably.
        among: The components to look in; the installed ones by default.

    Returns:
        What was written.

    Raises:
        FileNotFoundError: If there is no store or no complete checkpoint.
        TypeError: If the store holds another kind of state than the model's.
        ValueError: If its units or its grid are not the run's, there is nothing
            left to run, or the store is inside ``output_dir``.
        IntegrationDivergedError: If a monitor or a guard halted the run.
    """
    path = resolve(Path(restart_from))
    if path.resolve().is_relative_to(Path(output_dir).resolve()):
        raise ValueError(
            f"{path} is inside the output directory {output_dir}, whose artifacts "
            "this run would replace; restart into another directory"
        )
    component, built = build_run(spec, among)
    ds = io.load_dataset(path)
    expected = type(built.state0)
    recorded = ds.attrs.get("state_class")
    if recorded is not None and recorded != expected.__name__:
        raise TypeError(
            f"{path} holds a {recorded}, and {spec.scenario.name!r} x "
            f"{spec.model.name!r} needs a {expected.__name__}"
        )
    state0 = io.dataset_to_state(ds, state_class=expected)
    shapes = [x.shape for x in jax.tree.leaves(state0)]
    wanted = [x.shape for x in jax.tree.leaves(built.state0)]
    if shapes != wanted:
        raise ValueError(f"{path} has fields of shape {shapes}, the run {wanted}")
    _require_coordinates(built.coordinates, ds.attrs, path)
    whole = timeline_of(spec, diagnostics_per_save=diagnostics_per_save)
    return _run(
        spec,
        Path(output_dir),
        mode="restart",
        built=built,
        component=component,
        state0=state0,
        whole=whole,
        timeline=_rest(whole, ds.attrs),
        monitors=monitors,
        checkpointing=checkpointing,
    )
