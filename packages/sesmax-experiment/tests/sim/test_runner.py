"""The runner: its artifacts, its log, its failures, and restarts.

Ported from somax 9cb19b3 (``tests/test_cli_run.py``) to the testing component. The
tests of checkpoints are in ``test_checkpoints.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import PyTree
from run_specs import lorenz96, shallow_water

from sesmax_core import Model, io
from sesmax_core.component import AssertionFailedError, BuiltRun, Component
from sesmax_core.testing import L96State, Lorenz96
from sesmax_experiment.monitor import BaseMonitor, MonitorVerdict
from sesmax_experiment.sim import (
    IntegrationDivergedError,
    ModelSpec,
    OutputSpec,
    RunSpec,
    ScenarioSpec,
    TimesteppingSpec,
    load_yaml,
    restart,
    simulate,
    spinup,
    timeline_of,
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _chunk_lines(directory: Path) -> int:
    text = (directory / "run.log").read_text()
    return sum("| chunk " in line for line in text.splitlines())


class _NaNAfter(Model):
    """A ring that becomes NaN at a time, or is so from the start."""

    t: float = eqx.field(static=True)

    def vector_field(self, t: float, state: PyTree, args: PyTree | None = None):
        return L96State(x=jnp.where(t > self.t, jnp.nan, 0.0) * state.x)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        return state


class _Guarded(_NaNAfter):
    """The same, with a guard inside the tendency."""

    def vector_field(self, t: float, state: PyTree, args: PyTree | None = None):
        tendency = super().vector_field(t, state, args)
        bad = jnp.any(~jnp.isfinite(tendency.x))
        return L96State(x=eqx.error_if(tendency.x, bad, "non-finite tendency"))


def _component(model: Model, **kwargs) -> dict[str, Component]:
    def build(spec) -> BuiltRun:
        start = jnp.nan if spec["scenario"]["nondim"] else 1.0
        return BuiltRun(model=model, state0=L96State(x=jnp.full(4, start)))

    return {"toys": Component(name="toys", models={"toy": build}, **kwargs)}


def _toy_spec(**nondim) -> RunSpec:
    return RunSpec(
        scenario=ScenarioSpec(name="toy", nondim=nondim),
        model=ModelSpec(name="toy"),
        timestepping=TimesteppingSpec(t0=0.0, t1=1.0, dt=0.1, save_interval=0.5),
    )


class TestSimulate:
    def test_a_stable_run_writes_every_artifact(self, tmp_path: Path) -> None:
        result = simulate(shallow_water(save_interval=200.0), tmp_path)
        assert {p.name for p in tmp_path.iterdir()} == {
            "snapshots.zarr",
            "final_state.zarr",
            "metrics.json",
            "resolved.yaml",
            "manifest.json",
            "run.log",
            "metrics.jsonl",
        }
        assert result.output_dir == tmp_path
        assert result.metrics_path is not None
        for key, value in _json(result.metrics_path).items():
            if isinstance(value, int | float):
                assert np.isfinite(value), key

    def test_the_snapshots_are_the_states_at_the_saves(self, tmp_path: Path) -> None:
        result = simulate(shallow_water(save_interval=200.0), tmp_path)
        assert result.snapshots_path is not None
        ds = io.load_dataset(result.snapshots_path)
        np.testing.assert_array_equal(ds["time"], [0.0, 200.0, 400.0, 600.0])
        last = io.load_dataset(result.final_state_path)
        np.testing.assert_array_equal(ds["h"][-1], last["h"][0])
        np.testing.assert_array_equal(last["h"][0], np.asarray(result.state.h))

    def test_the_run_is_the_model_s_own(self, tmp_path: Path) -> None:
        result = simulate(lorenz96(), tmp_path)
        model = Lorenz96.create(F=8.0)
        state = L96State(x=jnp.full(12, 8.0).at[0].add(0.01))
        for start in (0.0, 0.2):
            run = model.integrate(state, t0=start, t1=start + 0.2, dt=0.01)
            state = L96State(x=run.ys.x[-1])
        np.testing.assert_allclose(result.state.x, state.x, rtol=1e-5)
        assert result.n_steps == 40

    def test_metrics_hold_diagnostics_and_the_component_s(self, tmp_path: Path) -> None:
        result = simulate(shallow_water(), tmp_path)
        metrics = _json(tmp_path / "metrics.json")
        # A scalar keeps its name; a field gives its mean, maximum and minimum.
        assert metrics["energy"] > 0.0
        assert metrics["relative_vorticity_max"] >= metrics["relative_vorticity_mean"]
        # The component's ``evaluate``, and the run's own numbers.
        assert metrics["largest_h"] > 0.5
        assert metrics["n_steps"] == result.n_steps == 60
        assert metrics["mode"] == "simulate"
        assert (metrics["t0"], metrics["t1"]) == (0.0, 600.0)

    def test_the_manifest_says_how_to_repeat_the_run(self, tmp_path: Path) -> None:
        simulate(shallow_water(), tmp_path)
        data = _json(tmp_path / "manifest.json")
        assert data["numerics"] == {
            "model_class": "LinearShallowWater2D",
            "bc": "periodic",
        }
        assert data["versions"]["sesmax-experiment"]
        assert re.fullmatch(r"[0-9a-f]{64}", data["config_sha256"])
        assert load_yaml(str(tmp_path / "resolved.yaml")) == shallow_water()

    def test_the_config_hash_sees_every_block(self, tmp_path: Path) -> None:
        simulate(lorenz96(), tmp_path / "si")
        simulate(lorenz96(nondim=True), tmp_path / "nd")
        hashes = {
            _json(tmp_path / d / "manifest.json")["config_sha256"] for d in ("si", "nd")
        }
        assert len(hashes) == 2

    def test_outputs_can_be_switched_off(self, tmp_path: Path) -> None:
        output = OutputSpec(write_snapshots=False, write_metrics=False)
        result = simulate(shallow_water(output=output), tmp_path)
        assert result.snapshots_path is None
        assert result.metrics_path is None
        assert not (tmp_path / "snapshots.zarr").exists()
        assert not (tmp_path / "metrics.json").exists()

    def test_the_artifacts_of_an_earlier_run_are_replaced(self, tmp_path: Path) -> None:
        simulate(shallow_water(t1=600.0, save_interval=200.0), tmp_path)
        simulate(shallow_water(t1=300.0), tmp_path)
        ds = io.load_dataset(tmp_path / "snapshots.zarr")
        np.testing.assert_array_equal(ds["time"], [0.0, 300.0])

    def test_a_monitor_can_stop_it(self, tmp_path: Path) -> None:
        class Stop(BaseMonitor):
            name = "stop"

            def on_chunk_end(self, model, state, info) -> MonitorVerdict:
                return MonitorVerdict(terminate=True, reason="enough")

        with pytest.raises(IntegrationDivergedError, match="enough"):
            simulate(shallow_water(), tmp_path, monitors=[Stop()])


class TestDivergence:
    def test_a_step_too_large_raises_and_writes_nothing(self, tmp_path: Path) -> None:
        """The Courant number is about 4, which blows up within the run."""
        spec = shallow_water(t1=48 * 2000.0, dt=2000.0, nx=64, save_interval=9600.0)
        with pytest.raises(IntegrationDivergedError) as error:
            simulate(spec, tmp_path)
        assert "non-finite" in str(error.value)
        assert "CFL" in str(error.value)
        for name in ("snapshots.zarr", "final_state.zarr", "metrics.json"):
            assert not (tmp_path / name).exists()
        assert "FAILED: IntegrationDivergedError" in (tmp_path / "run.log").read_text()

    def test_a_state_that_is_not_finite_stops_the_run(self, tmp_path: Path) -> None:
        among = _component(_NaNAfter(t=0.6))
        with pytest.raises(IntegrationDivergedError, match="non-finite"):
            simulate(_toy_spec(), tmp_path, among=among)
        # The snapshots taken before are kept aside.
        ds = io.load_dataset(tmp_path / "snapshots.partial.zarr")
        np.testing.assert_array_equal(ds["time"], [0.0, 0.5])

    def test_it_is_found_without_monitors(self, tmp_path: Path) -> None:
        among = _component(_NaNAfter(t=0.6))
        with pytest.raises(IntegrationDivergedError, match=r"x \(4/4 non-finite\)"):
            simulate(_toy_spec(), tmp_path, among=among, monitors=[])

    def test_an_initial_state_that_is_not_finite_is_named(self, tmp_path: Path) -> None:
        among = _component(_NaNAfter(t=2.0))
        with pytest.raises(IntegrationDivergedError, match="initial state was already"):
            simulate(_toy_spec(start=1.0), tmp_path, among=among)

    def test_a_guard_in_the_tendency_is_the_same_error(self, tmp_path: Path) -> None:
        among = _component(_Guarded(t=0.6))
        with pytest.raises(IntegrationDivergedError, match="non-finite tendency"):
            simulate(_toy_spec(), tmp_path, among=among)

    def test_the_mode_is_in_the_message(self, tmp_path: Path) -> None:
        among = _component(_NaNAfter(t=0.6))
        with pytest.raises(IntegrationDivergedError, match="spinup integration"):
            spinup(_toy_spec(), tmp_path, among=among)


class TestRunLog:
    def test_every_run_writes_one(self, tmp_path: Path) -> None:
        simulate(shallow_water(), tmp_path)
        log = (tmp_path / "run.log").read_text()
        for part in (
            "sesmax-experiment/simulate",
            "integration starting",
            "chunk 0/1",
            "chunk 1/1",
            "finished cleanly",
        ):
            assert part in log

    def test_a_chunk_s_line_has_its_fields_and_its_energy(self, tmp_path: Path) -> None:
        simulate(shallow_water(), tmp_path)
        line = [
            x for x in (tmp_path / "run.log").read_text().splitlines() if "1/1" in x
        ]
        assert "energy=" in line[0]
        assert "h=" in line[0] or "h[" in line[0]
        assert "m/s" in line[0]

    def test_more_chunks_between_saves_give_more_lines(self, tmp_path: Path) -> None:
        simulate(shallow_water(), tmp_path / "one")
        simulate(shallow_water(), tmp_path / "four", diagnostics_per_save=4)
        assert _chunk_lines(tmp_path / "one") == 2
        assert _chunk_lines(tmp_path / "four") == 5
        times = [
            io.load_dataset(tmp_path / d / "snapshots.zarr")["time"].values
            for d in ("one", "four")
        ]
        np.testing.assert_array_equal(*times)


class TestChecks:
    def test_a_check_runs_before_the_run(self, tmp_path: Path) -> None:
        spec = shallow_water(dt=2000.0, t1=4000.0, assertions={"gravity_wave_cfl": {}})
        with pytest.raises(AssertionFailedError, match="Courant number"):
            simulate(spec, tmp_path)
        assert not (tmp_path / "run.log").exists()

    def test_its_parameters_are_the_spec_s(self, tmp_path: Path) -> None:
        spec = shallow_water(assertions={"gravity_wave_cfl": {"limit": 1e-6}})
        with pytest.raises(AssertionFailedError, match="not below 1e-06"):
            simulate(spec, tmp_path)
        spec = shallow_water(assertions={"gravity_wave_cfl": {"limit": 0.5}})
        simulate(spec, tmp_path)

    def test_an_unknown_check_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionFailedError, match="unknown assertion"):
            simulate(shallow_water(assertions={"no_such_check": {}}), tmp_path)


def _checkpoint(directory: Path, attrs: dict, t: float = 200.0) -> Path:
    """A state with the attributes a checkpoint carries."""
    result = simulate(shallow_water(t1=t), directory / "stage1")
    path = directory / "checkpoint.zarr"
    io.save_dataset(io.state_to_dataset(result.state, time=t, attrs=attrs), path)
    return path


class TestRestart:
    def test_spinup_then_restart(self, tmp_path: Path) -> None:
        spun = spinup(shallow_water(), tmp_path / "spinup")
        assert {p.name for p in (tmp_path / "spinup").iterdir()} == {
            "final_state.zarr",
            "resolved.yaml",
            "manifest.json",
            "run.log",
            "metrics.jsonl",
        }
        result = restart(
            shallow_water(), tmp_path / "prod", restart_from=spun.final_state_path
        )
        assert result.snapshots_path is not None
        ds = io.load_dataset(result.snapshots_path)
        np.testing.assert_array_equal(ds["h"][0], np.asarray(spun.state.h))
        # A final state records no time, so the spec's whole window is run.
        np.testing.assert_array_equal(ds["time"], [0.0, 600.0])

    def test_a_run_in_two_parts_is_the_run(self, tmp_path: Path) -> None:
        whole = simulate(shallow_water(t1=400.0, save_interval=200.0), tmp_path / "all")
        path = _checkpoint(tmp_path, {"sesmax_sim_t": 200.0})
        second = restart(
            shallow_water(t1=400.0, save_interval=200.0),
            tmp_path / "second",
            restart_from=path,
        )
        np.testing.assert_array_equal(whole.state.h, second.state.h)

    @pytest.mark.parametrize("prefix", ["sesmax", "somax"])
    def test_the_start_moves_to_the_checkpoint(
        self, tmp_path: Path, prefix: str
    ) -> None:
        path = _checkpoint(tmp_path, {f"{prefix}_sim_t": 200.0})
        restart(
            shallow_water(save_interval=200.0), tmp_path / "next", restart_from=path
        )
        log = (tmp_path / "next" / "run.log").read_text()
        assert re.search(r"integration starting:[^\n]*\bt0=200 s\b", log)
        ds = io.load_dataset(tmp_path / "next" / "snapshots.zarr")
        np.testing.assert_array_equal(ds["time"], [200.0, 400.0, 600.0])

    def test_a_time_that_is_not_a_number_raises(self, tmp_path: Path) -> None:
        path = _checkpoint(tmp_path, {"sesmax_sim_t": "not-a-number"})
        with pytest.raises(ValueError, match="non-numeric sim_t"):
            restart(shallow_water(), tmp_path / "next", restart_from=path)

    def test_nothing_left_to_run_raises(self, tmp_path: Path) -> None:
        path = _checkpoint(tmp_path, {"sesmax_sim_t": 600.0}, t=600.0)
        with pytest.raises(ValueError, match="nothing left"):
            restart(shallow_water(), tmp_path / "next", restart_from=path)

    def test_another_model_s_state_raises(self, tmp_path: Path) -> None:
        spun = spinup(lorenz96(), tmp_path / "spinup")
        with pytest.raises(TypeError, match="holds a L96State"):
            restart(
                shallow_water(), tmp_path / "next", restart_from=spun.final_state_path
            )

    def test_a_restart_into_its_own_directory_raises(self, tmp_path: Path) -> None:
        """It would remove the state it starts from."""
        spun = spinup(shallow_water(), tmp_path)
        with pytest.raises(ValueError, match="another directory"):
            restart(shallow_water(), tmp_path, restart_from=spun.final_state_path)
        assert spun.final_state_path.is_dir()

    def test_another_grid_s_state_raises(self, tmp_path: Path) -> None:
        spun = spinup(shallow_water(nx=8), tmp_path / "spinup")
        with pytest.raises(ValueError, match="shape"):
            restart(
                shallow_water(), tmp_path / "next", restart_from=spun.final_state_path
            )

    @pytest.mark.parametrize("nondim", [False, True])
    def test_other_units_raise(self, tmp_path: Path, nondim: bool) -> None:
        spun = spinup(lorenz96(nondim=nondim), tmp_path / "spinup")
        with pytest.raises(ValueError, match="wrong units"):
            restart(
                lorenz96(nondim=not nondim),
                tmp_path / "next",
                restart_from=spun.final_state_path,
            )

    def test_a_store_from_somax_loads(self, tmp_path: Path) -> None:
        """somax marks the units ``somax_coordinates``."""
        result = simulate(lorenz96(nondim=True), tmp_path / "stage1")
        path = tmp_path / "final_state.zarr"
        attrs = {"somax_coordinates": "nondimensional", "somax_sim_mode": "run"}
        io.save_dataset(io.state_to_dataset(result.state, time=0.4, attrs=attrs), path)
        restart(lorenz96(nondim=True), tmp_path / "next", restart_from=path)
        with pytest.raises(ValueError, match="wrong units"):
            restart(lorenz96(), tmp_path / "si", restart_from=path)


class TestTimeline:
    """The saves are exactly ``save_interval`` apart, and the end closes them."""

    def _saves(self, t1: float, interval: float, **kwargs) -> np.ndarray:
        timeline = timeline_of(
            shallow_water(t1=t1, dt=0.1, save_interval=interval), **kwargs
        )
        return timeline.edges[sorted(timeline.snapshots)]

    def test_an_interval_that_divides_the_window(self) -> None:
        np.testing.assert_array_equal(self._saves(10.0, 1.0), np.arange(11.0))

    def test_one_that_does_not_is_closed_by_the_end(self) -> None:
        np.testing.assert_array_equal(self._saves(10.0, 6.0), [0.0, 6.0, 10.0])

    def test_months_of_a_year(self) -> None:
        year, month = 365.25 * 86400.0, 30 * 86400.0
        saves = self._saves(year, month)
        np.testing.assert_array_equal(saves[:13], month * np.arange(13))
        assert saves.shape == (14,) and saves[-1] == year

    def test_only_the_final_state(self) -> None:
        np.testing.assert_array_equal(
            self._saves(10.0, 1.0, only_final=True), [0.0, 10.0]
        )

    def test_chunks_between_the_saves(self) -> None:
        timeline = timeline_of(
            shallow_water(t1=10.0, dt=0.1, save_interval=6.0), diagnostics_per_save=2
        )
        np.testing.assert_array_equal(timeline.edges, [0.0, 3.0, 6.0, 8.0, 10.0])
        assert sorted(timeline.snapshots) == [0, 2, 4]
