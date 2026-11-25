"""Runs in chunks, on Lorenz-96 and on linear shallow water."""

from __future__ import annotations

import warnings
from typing import Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pipekit_ops import Branch, Retry, Sink, Tap, Try

from sesmax_core import Model
from sesmax_core.testing import (
    L96State,
    LinearShallowWater2D,
    LinearSW2DState,
    Lorenz96,
)
from sesmax_experiment.monitor import (
    BaseMonitor,
    MonitorVerdict,
    NonFiniteMonitor,
    default_monitors,
)
from sesmax_experiment.run import (
    RunStopped,
    Timeline,
    every_chunks,
    run_chunks,
)


@pytest.fixture(autouse=True)
def _float64():
    with jax.enable_x64(True):
        yield


def _lorenz() -> tuple[Model, L96State]:
    rng = np.random.default_rng(0)
    return Lorenz96.create(F=8.0), L96State(x=jnp.asarray(8.0 + rng.normal(size=40)))


TIMELINE = {"chunk": 0.1, "snapshot": 0.5}


def test_chunks_of_an_inexact_ratio_give_one_solve() -> None:
    """0.1 / 0.01 is 10 steps in every chunk, so the answer is that of one solve."""
    model, state = _lorenz()
    run = run_chunks(model, state, Timeline.build(0.0, 4.0, **TIMELINE), dt=0.01)
    times = [frame.info.t1 for frame in run.snapshots]
    assert times == pytest.approx([0.5 * k for k in range(9)])
    one = model.integrate(
        state,
        t0=0.0,
        t1=4.0,
        dt=None,
        stepsize_controller=dfx.StepTo(ts=jnp.linspace(0.0, 4.0, 401)),
        saveat=dfx.SaveAt(ts=jnp.asarray(times)),
        max_steps=400,
    ).ys.x
    got = jnp.stack([frame.state.x for frame in run.snapshots])
    assert float(jnp.abs(got - one).max()) < 1e-8 * float(jnp.abs(one).max())


def test_every_chunk_takes_the_stated_steps() -> None:
    model, state = _lorenz()
    steps: list[int] = []
    run_chunks(
        model,
        state,
        Timeline.build(0.0, 4.0, chunk=0.1),
        dt=0.01,
        stages=[Tap(lambda frame: steps.append(frame.info.stats["num_steps"]))],
    )
    assert steps == [10] * 40


def test_a_shorter_last_chunk_takes_fewer_steps() -> None:
    model, state = _lorenz()
    run = run_chunks(model, state, Timeline.build(0.0, 0.25, chunk=0.1), dt=0.01)
    assert run.frame.info.stats["num_steps"] == 5


def test_linear_shallow_water_in_chunks() -> None:
    model = LinearShallowWater2D.create(nx=16, ny=12)
    shape = model.grid.shape
    rng = np.random.default_rng(0)
    state = LinearSW2DState(
        h=jnp.asarray(rng.normal(size=shape)), u=jnp.zeros(shape), v=jnp.zeros(shape)
    )
    timeline = Timeline.build(0.0, 1000.0, chunk=100.0)
    run = run_chunks(model, state, timeline, dt=10.0)
    one = model.integrate(state, t0=0.0, t1=1000.0, dt=10.0).ys
    inner = (slice(1, -1), slice(1, -1))
    for field in ("h", "u", "v"):
        expected = getattr(one, field)[-1][inner]
        got = getattr(run.state, field)[inner]
        assert float(jnp.abs(got - expected).max()) < 1e-12


def test_a_run_compiles_once() -> None:
    traces: list[int] = []

    class Counting(Lorenz96):
        def vector_field(self, t: Any, state: Any, args: Any = None) -> Any:
            traces.append(1)
            return super().vector_field(t, state, args)

    plain, state = _lorenz()
    model = Counting(params=plain.params)
    run_chunks(model, state, Timeline.build(0.0, 0.1, chunk=0.1), dt=0.01)
    first = len(traces)
    run_chunks(model, state, Timeline.build(0.0, 4.0, chunk=0.1), dt=0.01)
    assert first > 0
    assert len(traces) == first


def test_a_restart_from_a_snapshot_is_the_uninterrupted_run() -> None:
    model, state = _lorenz()
    whole = run_chunks(model, state, Timeline.build(0.0, 4.0, **TIMELINE), dt=0.01)
    middle = whole.snapshots[4]
    assert middle.info.t1 == pytest.approx(2.0)
    again = run_chunks(
        model, middle.state, Timeline.build(2.0, 4.0, **TIMELINE), dt=0.01
    )
    np.testing.assert_allclose(again.state.x, whole.state.x, rtol=1e-9)


def test_a_sink_takes_the_snapshots() -> None:
    """And then the run keeps none of them."""
    model, state = _lorenz()
    timeline = Timeline.build(0.0, 1.0, **TIMELINE)
    taken = []
    streamed = run_chunks(
        model, state, timeline, dt=0.01, on_snapshot=lambda f: taken.append(f.state.x)
    )
    kept = run_chunks(model, state, timeline, dt=0.01)
    assert streamed.snapshots == ()
    for a, b in zip(taken, kept.snapshots, strict=True):
        np.testing.assert_array_equal(a, b.state.x)


class _Writer:
    """A checkpoint write that fails the first ``failures`` times."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.written: list[float] = []

    def __call__(self, frame: Any) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise OSError("no space")
        self.written.append(frame.info.t1)


def _checkpoint(writer: _Writer) -> Branch:
    sturdy = Try(
        Retry(Sink(writer), attempts=2, base_delay=0.0, on=(OSError,)),
        Tap(lambda frame: warnings.warn("checkpoint skipped", stacklevel=1)),
        on=(OSError,),
    )
    return Branch(every_chunks(2), sturdy)


def test_a_write_that_fails_once_is_retried() -> None:
    model, state = _lorenz()
    timeline = Timeline.build(0.0, 0.8, chunk=0.1)
    writer = _Writer(failures=1)
    run = run_chunks(model, state, timeline, dt=0.01, stages=[_checkpoint(writer)])
    assert writer.calls == 5
    assert writer.written == pytest.approx([0.2, 0.4, 0.6, 0.8])
    plain = run_chunks(model, state, timeline, dt=0.01)
    np.testing.assert_array_equal(run.state.x, plain.state.x)


def test_a_write_that_always_fails_does_not_stop_the_run() -> None:
    model, state = _lorenz()
    timeline = Timeline.build(0.0, 0.8, chunk=0.1)
    writer = _Writer(failures=100)
    with pytest.warns(UserWarning, match="checkpoint skipped") as caught:
        run = run_chunks(model, state, timeline, dt=0.01, stages=[_checkpoint(writer)])
    assert (writer.calls, len(caught)) == (8, 4)
    plain = run_chunks(model, state, timeline, dt=0.01)
    np.testing.assert_array_equal(run.state.x, plain.state.x)


class _Recorder(BaseMonitor):
    name = "recorder"

    def __init__(self, stop_at: int | None = None) -> None:
        self.stop_at = stop_at
        self.log: list[Any] = []

    def on_run_start(self, model: Any, state0: Any, *, spec: Any) -> None:
        self.log.append(("start", spec))

    def on_chunk_end(self, model: Any, state: Any, info: Any) -> MonitorVerdict:
        self.log.append(info.index)
        if info.index == self.stop_at:
            return MonitorVerdict(terminate=True, reason="enough")
        return MonitorVerdict(metrics={"t": info.t1}, messages=("here",))

    def on_run_end(self, model: Any, final_state: Any, *, ok: bool) -> None:
        self.log.append(("end", ok))


def test_the_monitors_see_the_run() -> None:
    model, state = _lorenz()
    recorder = _Recorder()
    run = run_chunks(
        model,
        state,
        Timeline.build(0.0, 0.3, chunk=0.1),
        dt=0.01,
        monitors=[*default_monitors(), recorder],
        spec="the spec",
    )
    assert recorder.log == [("start", "the spec"), 0, 1, 2, ("end", True)]
    assert run.frame.metrics["recorder.t"] == pytest.approx(0.3)
    assert "[recorder] here" in run.frame.messages


def test_a_monitor_stops_the_run_before_anything_is_written() -> None:
    model, state = _lorenz()
    recorder, taken, staged = _Recorder(stop_at=1), [], []
    with pytest.raises(RunStopped, match=r"chunk 2/4.*'recorder': enough") as caught:
        run_chunks(
            model,
            state,
            Timeline.build(0.0, 0.4, chunk=0.1, snapshot=0.1),
            dt=0.01,
            monitors=[recorder],
            on_snapshot=lambda frame: taken.append(frame.info.index),
            stages=[Tap(lambda frame: staged.append(frame.info.index))],
        )
    # the start and the first chunk
    assert taken == [-1, 0]
    assert staged == [0]
    assert recorder.log[-1] == ("end", False)
    assert caught.value.frame.info.index == 1


def test_a_state_that_is_not_finite_stops_the_run() -> None:
    model, state = _lorenz()
    bad = L96State(x=state.x.at[0].set(jnp.nan))
    with pytest.raises(RunStopped, match="chunk 1/1"):
        run_chunks(
            model,
            bad,
            Timeline.build(0.0, 0.1, chunk=0.1),
            dt=0.01,
            monitors=[NonFiniteMonitor()],
        )


def test_a_controller_sets_its_own_steps() -> None:
    model, state = _lorenz()
    run = run_chunks(
        model,
        state,
        Timeline.build(0.0, 0.5, chunk=0.1),
        dt=0.01,
        solver=dfx.Dopri5(),
        stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-8),
    )
    one = model.integrate(
        state,
        t0=0.0,
        t1=0.5,
        dt=0.01,
        solver=dfx.Dopri8(),
        stepsize_controller=dfx.PIDController(rtol=1e-11, atol=1e-11),
    ).ys.x[-1]
    np.testing.assert_allclose(run.state.x, one, rtol=1e-5)
    assert run.frame.info.stats["num_steps"] != 10
