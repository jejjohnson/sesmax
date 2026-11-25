"""Run a spec and write its artifacts: simulate, spinup and restart."""

from __future__ import annotations

import dataclasses
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import numpy as np
import paramax
from loguru import logger
from pipekit_ops import Tap

from sesmax_core import io
from sesmax_core.component import BuiltRun, Component
from sesmax_core.fields import values
from sesmax_experiment._src.components import find_model
from sesmax_experiment._src.monitor import Monitor, default_monitors
from sesmax_experiment._src.run.chunks import ChunkedRun, run_chunks
from sesmax_experiment._src.run.frame import ChunkFrame, RunStopped
from sesmax_experiment._src.run.timeline import Timeline
from sesmax_experiment._src.sim.artifacts import (
    SnapshotStore,
    flat_diagnostics,
    manifest,
    run_attrs,
    write_json,
)
from sesmax_experiment._src.sim.checks import preflight
from sesmax_experiment._src.sim.progress import (
    RunLogContext,
    start_run_log,
    stop_run_log,
)
from sesmax_experiment._src.sim.report import chunk_line
from sesmax_experiment._src.sim.spec import RunSpec, TimesteppingSpec, dump_yaml
from sesmax_experiment._src.sim.units import (
    field_units,
    format_time_seconds,
    format_wallclock,
)


Mode = Literal["simulate", "spinup", "restart"]


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
        snapshots_path: The snapshots, or ``None`` if there are none.
        final_state_path: The final state, which a restart reads.
        metrics_path: The metrics, or ``None`` if there are none.
        wallclock_seconds: How long the integration took.
        n_steps: The steps the solver took.
        state: The final state.
    """

    output_dir: Path
    snapshots_path: Path | None
    final_state_path: Path
    metrics_path: Path | None
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


def _integrate(
    spec: RunSpec,
    built: BuiltRun,
    state0: Any,
    timeline: Timeline,
    *,
    mode: Mode,
    monitors: Sequence[Monitor],
    store: SnapshotStore | None,
    units: dict[str, str],
    run_log: RunLogContext,
    steps: list[int],
) -> ChunkedRun:
    """The chunks of a run, with its log and its snapshots."""
    model = built.model

    def log_chunk(frame: ChunkFrame) -> None:
        steps.append(int(frame.info.stats.get("num_steps", 0)))
        run_log.log.debug(chunk_line(model, frame, units))

    def snapshot(frame: ChunkFrame) -> None:
        _assert_finite(frame)
        if frame.info.index < 0:
            run_log.log.debug(chunk_line(model, frame, units))
        if store is not None:
            store(frame)

    try:
        return run_chunks(
            model,
            state0,
            timeline,
            dt=spec.timestepping.dt,
            monitors=monitors,
            on_snapshot=snapshot,
            stages=[Tap(log_chunk)],
            spec=spec,
        )
    except RunStopped as exc:
        run_log.log.debug(chunk_line(model, exc.frame, units))
        raise _diverged(mode, exc) from exc
    except eqx.EquinoxRuntimeError as exc:
        # A guard inside the tendency halted the step that went wrong.
        raise _diverged(mode, exc) from exc


def _metrics(
    spec: RunSpec,
    component: Component,
    built: BuiltRun,
    state: Any,
    *,
    mode: Mode,
    wall_seconds: float,
    n_steps: int,
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
    times = spec.timestepping
    return out | {
        "wallclock_seconds": wall_seconds,
        "n_steps": n_steps,
        "t0": times.t0,
        "t1": times.t1,
        "save_interval": times.save_interval,
        "mode": mode,
    }


def _run(
    spec: RunSpec,
    output_dir: Path,
    *,
    mode: Mode,
    built: BuiltRun,
    component: Component,
    state0: Any,
    diagnostics_per_save: int,
    monitors: Sequence[Monitor] | None,
) -> SimulationResult:
    """Integrate, and write everything a successful run writes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    only_final = mode == "spinup"
    logger.info("{}: {} x {}", mode, spec.scenario.name, spec.model.name)
    logger.info("output_dir={}", output_dir)
    preflight(spec, built.model, component)

    timeline = timeline_of(
        spec, only_final=only_final, diagnostics_per_save=diagnostics_per_save
    )
    attrs = run_attrs(spec, mode=mode, coordinates=built.coordinates)
    store = SnapshotStore(output_dir, attrs)
    for stale in ("final_state.zarr",):
        if (output_dir / stale).exists():
            shutil.rmtree(output_dir / stale)
    for stale in ("metrics.json", "manifest.json"):
        (output_dir / stale).unlink(missing_ok=True)
    units = field_units(
        [field.name for field in dataclasses.fields(state0)],
        component.field_units.get(spec.model.name, {}),
        nondimensional=built.coordinates == "nondimensional",
    )
    times = spec.timestepping
    writes_snapshots = spec.output.write_snapshots and not only_final
    writes_metrics = spec.output.write_metrics and not only_final
    run_log = start_run_log(output_dir, label=f"sesmax-experiment/{mode}")
    run_log.log.debug(
        f"integration starting: scenario={spec.scenario.name} "
        f"model={spec.model.name} t0={format_time_seconds(times.t0)} "
        f"t1={format_time_seconds(times.t1)} dt={times.dt} s "
        f"chunks={timeline.n_chunks} snapshots={len(timeline.snapshots)}"
    )
    message = "finished cleanly"
    try:
        steps: list[int] = []
        start = time.perf_counter()
        run = _integrate(
            spec,
            built,
            state0,
            timeline,
            mode=mode,
            monitors=default_monitors() if monitors is None else monitors,
            store=store if writes_snapshots else None,
            units=units,
            run_log=run_log,
            steps=steps,
        )
        wall_seconds = time.perf_counter() - start
        logger.info("integration finished in {}", format_wallclock(wall_seconds))
        state = run.state

        metrics_path = None
        if writes_metrics:
            metrics = _metrics(
                spec,
                component,
                built,
                state,
                mode=mode,
                wall_seconds=wall_seconds,
                n_steps=sum(steps),
            )
            metrics_path = write_json(output_dir / "metrics.json", metrics)
        snapshots_path = store.finish() if writes_snapshots else None
        final_state_path = output_dir / "final_state.zarr"
        io.save_dataset(
            io.state_to_dataset(state, time=float(times.t1), attrs=attrs),
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
                wall_seconds=wall_seconds,
                numerics={} if describe is None else describe(built.model),
            ),
        )
        logger.info("wrote {}", output_dir)
    except BaseException as exc:
        message = f"FAILED: {type(exc).__name__}: {exc}"
        raise
    finally:
        stop_run_log(run_log, final_message=message)
    return SimulationResult(
        output_dir=output_dir,
        snapshots_path=snapshots_path,
        final_state_path=final_state_path,
        metrics_path=metrics_path,
        wallclock_seconds=wall_seconds,
        n_steps=sum(steps),
        state=state,
    )


def simulate(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
    monitors: Sequence[Monitor] | None = None,
    among: Mapping[str, Component] | None = None,
) -> SimulationResult:
    """Run a spec from its initial condition, and write its artifacts.

    ``run.log`` is written as the run goes, and snapshots are streamed to
    ``snapshots.partial.zarr``. A run that succeeds leaves ``snapshots.zarr``,
    ``final_state.zarr``, ``metrics.json``, ``resolved.yaml`` and
    ``manifest.json``; one that fails leaves none of these.

    Args:
        spec: The spec, with its debug block merged.
        output_dir: Where to write. Artifacts of an earlier run there are removed.
        diagnostics_per_save: Chunks between two snapshots, each a line in
            ``run.log``.
        monitors: The monitors; ``default_monitors()`` by default.
        among: The components to look in; the installed ones by default.

    Returns:
        What was written.

    Raises:
        IntegrationDivergedError: If a monitor or a guard halted the run.
    """
    component, built = build_run(spec, among)
    return _run(
        spec,
        Path(output_dir),
        mode="simulate",
        built=built,
        component=component,
        state0=built.state0,
        diagnostics_per_save=diagnostics_per_save,
        monitors=monitors,
    )


def spinup(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    diagnostics_per_save: int = 1,
    monitors: Sequence[Monitor] | None = None,
    among: Mapping[str, Component] | None = None,
) -> SimulationResult:
    """Run a spec for its final state alone, which a restart then starts from.

    Args:
        spec: The spec, with its debug block merged.
        output_dir: Where to write.
        diagnostics_per_save: Chunks of the run, each a line in ``run.log``.
        monitors: The monitors; ``default_monitors()`` by default.
        among: The components to look in; the installed ones by default.

    Returns:
        What was written: no snapshots and no metrics.

    Raises:
        IntegrationDivergedError: If a monitor or a guard halted the run.
    """
    component, built = build_run(spec, among)
    return _run(
        spec,
        Path(output_dir),
        mode="spinup",
        built=built,
        component=component,
        state0=built.state0,
        diagnostics_per_save=diagnostics_per_save,
        monitors=monitors,
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


def _continued(spec: RunSpec, attrs: Mapping[str, Any]) -> RunSpec:
    """The spec, starting where a checkpoint stopped."""
    raw = _attr(attrs, "sim_t")
    if raw is None:
        return spec
    try:
        t = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"the state to restart from has a non-numeric sim_t={raw!r}; it must be "
            "a number of seconds"
        ) from exc
    times = spec.timestepping
    if t <= times.t0:
        return spec
    if t >= times.t1:
        raise ValueError(
            f"the state to restart from is at t={t}, and the run ends at "
            f"t1={times.t1}: there is nothing left to run. Extend t1."
        )
    logger.info("restart: continuing at t={}", t)
    rest = TimesteppingSpec(
        t0=t,
        t1=times.t1,
        dt=times.dt,
        save_interval=min(times.save_interval, times.t1 - t),
    )
    spec = dataclasses.replace(spec, timestepping=rest)
    spec.validate()
    return spec


def restart(
    spec: RunSpec,
    output_dir: str | Path,
    *,
    restart_from: str | Path,
    diagnostics_per_save: int = 1,
    monitors: Sequence[Monitor] | None = None,
    among: Mapping[str, Component] | None = None,
) -> SimulationResult:
    """Run a spec from a saved state.

    The model is built from the spec, so a viscosity or a forcing may differ from
    the earlier run's; only the state is loaded. A state that records its time, as
    a checkpoint does, moves the start of the run there. One that does not, a
    ``final_state.zarr``, starts the spec's whole window. Stores written by somax
    load too.

    Args:
        spec: The spec, with its debug block merged.
        output_dir: Where to write, not the directory of ``restart_from``.
        restart_from: A zarr store with a state.
        diagnostics_per_save: Chunks between two snapshots.
        monitors: The monitors; ``default_monitors()`` by default.
        among: The components to look in; the installed ones by default.

    Returns:
        What was written.

    Raises:
        TypeError: If the store holds another kind of state than the model's.
        ValueError: If its units or its grid are not the run's, there is nothing
            left to run, or the store is inside ``output_dir``.
        IntegrationDivergedError: If a monitor or a guard halted the run.
    """
    path = Path(restart_from)
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
    return _run(
        _continued(spec, ds.attrs),
        Path(output_dir),
        mode="restart",
        built=built,
        component=component,
        state0=state0,
        diagnostics_per_save=diagnostics_per_save,
        monitors=monitors,
    )
