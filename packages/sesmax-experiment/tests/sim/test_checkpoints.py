"""Checkpoints: what is written, what survives a crash, and what restarts."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from run_specs import shallow_water

from sesmax_core import io
from sesmax_core.testing import L96State
from sesmax_experiment.monitor import (
    BaseMonitor,
    MonitorVerdict,
    WatchdogMonitor,
    default_monitors,
)
from sesmax_experiment.run import every_chunks
from sesmax_experiment.sim import (
    Checkpointing,
    CheckpointStore,
    IntegrationDivergedError,
    restart,
    simulate,
)


EVERY_CHUNK = Checkpointing(every_chunks(1))


def _spec(**kwargs):
    return shallow_water(**({"t1": 600.0, "save_interval": 100.0} | kwargs))


def _state(x: float) -> L96State:
    return L96State(x=jnp.full(4, x))


class _StopAfter(BaseMonitor):
    """Asks to stop after a chunk, resumably or not."""

    name = "stop_after"

    def __init__(self, index: int, *, resumable: bool = True) -> None:
        self.index, self.resumable = index, resumable

    def on_chunk_end(self, model, state, info) -> MonitorVerdict:
        due = info.index == self.index
        return MonitorVerdict(
            terminate=due,
            reason="enough" if due else None,
            resumable=self.resumable,
        )


class TestStore:
    def _save(self, store: CheckpointStore, *steps: int) -> None:
        for step in steps:
            store.save(_state(step), step=step, time=10.0 * step, attrs={"a": 1})

    def test_each_is_a_store_of_its_own(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path / "c")
        assert store.steps() == [] and store.latest() is None
        self._save(store, 1, 2)
        assert store.steps() == [1, 2]
        ds = io.load_dataset(store.path(2))
        assert ds.attrs["sesmax_sim_t"] == 20.0
        assert (ds.attrs["sesmax_step"], ds.attrs["a"]) == (2, 1)

    def test_the_newest_are_kept(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path, keep_last=2)
        self._save(store, *range(1, 8))
        assert store.steps() == [6, 7]

    def test_an_archive_is_kept(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path, keep_last=1, keep_every=3)
        self._save(store, *range(1, 9))
        assert store.steps() == [3, 6, 8]

    def test_a_partial_one_is_not_one(self, tmp_path: Path) -> None:
        store = CheckpointStore(tmp_path)
        self._save(store, 1)
        (tmp_path / "00000002.partial.zarr").mkdir()
        assert store.latest() == store.path(1)
        # The next write of that step replaces it.
        self._save(store, 2)
        assert store.latest() == store.path(2)
        assert not (tmp_path / "00000002.partial.zarr").exists()

    def test_a_write_that_dies_leaves_the_older_ones(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = CheckpointStore(tmp_path, keep_last=1)
        self._save(store, 1)
        save = io.save_dataset

        def dies(ds, path, **kwargs):
            save(ds, path, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(io, "save_dataset", dies)
        with pytest.raises(KeyboardInterrupt):
            self._save(store, 2)
        assert store.steps() == [1]

    @pytest.mark.parametrize("bad", [{"keep_last": 0}, {"keep_every": 0}])
    def test_numbers_are_positive(self, bad: dict[str, int]) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            Checkpointing(every_chunks(1), **bad)


class TestWhileRunning:
    def test_none_by_default(self, tmp_path: Path) -> None:
        result = simulate(_spec(), tmp_path)
        assert result.status == "finished"
        assert result.checkpoint_path is None
        assert not (tmp_path / "checkpoints").exists()

    def test_the_newest_is_the_end(self, tmp_path: Path) -> None:
        result = simulate(
            _spec(), tmp_path, checkpointing=Checkpointing(every_chunks(2), keep_last=2)
        )
        store = CheckpointStore(tmp_path / "checkpoints")
        assert store.steps() == [4, 6]
        assert result.checkpoint_path == store.path(6)
        ds = io.load_dataset(store.path(6))
        assert ds.attrs["sesmax_sim_t"] == 600.0
        np.testing.assert_array_equal(ds["h"][0], result.state.h)

    def test_those_of_an_earlier_run_are_removed(self, tmp_path: Path) -> None:
        simulate(_spec(), tmp_path, checkpointing=EVERY_CHUNK)
        (tmp_path / "checkpoint.zarr").mkdir()
        simulate(_spec(), tmp_path)
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "checkpoint.zarr").exists()

    @pytest.mark.parametrize("failures", [1, 2])
    def test_a_failed_write_does_not_stop_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failures: int
    ) -> None:
        save, calls = CheckpointStore.save, []

        def flaky(self, state, *, step, **kwargs):
            calls.append(step)
            if step == 2 and calls.count(2) <= failures:
                raise OSError("the mount is away")
            return save(self, state, step=step, **kwargs)

        monkeypatch.setattr(CheckpointStore, "save", flaky)
        simulate(_spec(), tmp_path, checkpointing=Checkpointing(every_chunks(1), 9))
        steps = CheckpointStore(tmp_path / "checkpoints").steps()
        assert steps == ([1, 2, 3, 4, 5, 6] if failures == 1 else [1, 3, 4, 5, 6])
        warned = "could not be written" in (tmp_path / "run.log").read_text()
        assert warned == (failures == 2)

    def test_a_state_that_is_not_finite_is_not_written(self, tmp_path: Path) -> None:
        spec = shallow_water(t1=48 * 2000.0, dt=2000.0, nx=64, save_interval=96e3)
        with pytest.raises(IntegrationDivergedError, match="non-finite"):
            simulate(
                spec,
                tmp_path,
                monitors=[],
                diagnostics_per_save=24,
                checkpointing=EVERY_CHUNK,
            )
        for step in (store := CheckpointStore(tmp_path / "checkpoints")).steps():
            assert np.isfinite(io.load_dataset(store.path(step))["h"]).all()


class TestRestart:
    @pytest.mark.parametrize("per_save", [1, 3])
    def test_the_rest_of_the_run_is_the_run(
        self, tmp_path: Path, per_save: int
    ) -> None:
        """From any checkpoint, with one between two snapshots among them."""
        whole = simulate(_spec(), tmp_path / "all", diagnostics_per_save=per_save)
        simulate(
            _spec(),
            tmp_path / "some",
            diagnostics_per_save=per_save,
            checkpointing=Checkpointing(every_chunks(1), keep_last=99),
        )
        ds = io.load_dataset(tmp_path / "all" / "snapshots.zarr")
        for step in (2, 4):
            rest = restart(
                _spec(),
                tmp_path / f"rest{step}",
                restart_from=tmp_path / "some" / "checkpoints" / f"{step:08d}.zarr",
                diagnostics_per_save=per_save,
            )
            np.testing.assert_array_equal(rest.state.h, whole.state.h)
            assert rest.snapshots_path is not None
            part = io.load_dataset(rest.snapshots_path)
            # The snapshots are at the times, and the states, of the whole run.
            assert set(part["time"].values) <= set(ds["time"].values)
            same = ds.sel(time=part["time"])
            np.testing.assert_array_equal(part["h"], same["h"])

    def test_the_steps_go_on_counting(self, tmp_path: Path) -> None:
        spec = _spec()
        simulate(spec, tmp_path / "a", monitors=[_StopAfter(2)])
        restart(
            spec,
            tmp_path / "b",
            restart_from=tmp_path / "a",
            checkpointing=Checkpointing(every_chunks(1), keep_last=99),
        )
        store = CheckpointStore(tmp_path / "b" / "checkpoints")
        assert store.steps() == [4, 5, 6]
        lines = (tmp_path / "b" / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line)["step"] for line in lines] == [4, 5, 6]

    @pytest.mark.parametrize("inside", ["", "checkpoints"])
    def test_a_directory_gives_its_newest(self, tmp_path: Path, inside: str) -> None:
        simulate(_spec(), tmp_path / "a", monitors=[_StopAfter(1)])
        (tmp_path / "a" / "checkpoints" / "00000009.partial.zarr").mkdir()
        restart(_spec(), tmp_path / "b", restart_from=tmp_path / "a" / inside)
        ds = io.load_dataset(tmp_path / "b" / "snapshots.zarr")
        np.testing.assert_array_equal(ds["time"], [200, 300, 400, 500, 600])

    def test_the_one_store_of_a_somax_run(self, tmp_path: Path) -> None:
        result = simulate(_spec(t1=300.0), tmp_path / "a")
        attrs = {"somax_sim_t": 300.0, "somax_coordinates": "si"}
        ds = io.state_to_dataset(result.state, time=300.0, attrs=attrs)
        io.save_dataset(ds, tmp_path / "a" / "checkpoint.zarr")
        restart(_spec(), tmp_path / "b", restart_from=tmp_path / "a")
        ds = io.load_dataset(tmp_path / "b" / "snapshots.zarr")
        np.testing.assert_array_equal(ds["time"], [300, 400, 500, 600])

    def test_a_directory_without_one_raises(self, tmp_path: Path) -> None:
        simulate(_spec(), tmp_path / "a")
        with pytest.raises(FileNotFoundError, match="no complete checkpoint"):
            restart(_spec(), tmp_path / "b", restart_from=tmp_path / "a")

    def test_a_finished_run_has_nothing_left(self, tmp_path: Path) -> None:
        simulate(_spec(), tmp_path / "a", checkpointing=EVERY_CHUNK)
        with pytest.raises(ValueError, match="nothing left"):
            restart(_spec(), tmp_path / "b", restart_from=tmp_path / "a")


class TestStopping:
    def test_a_resumable_stop(self, tmp_path: Path) -> None:
        whole = simulate(_spec(), tmp_path / "all")
        result = simulate(
            _spec(), tmp_path / "a", monitors=[*default_monitors(), _StopAfter(2)]
        )
        assert result.status == "stopped"
        assert result.final_state_path is None and result.metrics_path is None
        assert result.checkpoint_path == tmp_path / "a" / "checkpoints/00000003.zarr"
        assert not (tmp_path / "a" / "final_state.zarr").exists()
        # What it had is there, and says what it is.
        ds = io.load_dataset(tmp_path / "a" / "snapshots.zarr")
        np.testing.assert_array_equal(ds["time"], [0.0, 100.0, 200.0])
        data = json.loads((tmp_path / "a" / "manifest.json").read_text())
        assert (data["status"], data["t1"]) == ("stopped", 300.0)
        assert "enough" in data["reason"]
        assert "can be resumed" in (tmp_path / "a" / "run.log").read_text()

        rest = restart(_spec(), tmp_path / "b", restart_from=tmp_path / "a")
        assert rest.status == "finished"
        np.testing.assert_array_equal(rest.state.h, whole.state.h)
        ds = io.load_dataset(tmp_path / "b" / "snapshots.zarr")
        np.testing.assert_array_equal(ds["time"], [300, 400, 500, 600])
        data = json.loads((tmp_path / "b" / "manifest.json").read_text())
        assert (data["status"], data["t0"]) == ("finished", 300.0)

    def test_the_wall_time_is_one(self, tmp_path: Path) -> None:
        result = simulate(_spec(), tmp_path, monitors=[WatchdogMonitor(max_wall_s=0.0)])
        assert result.status == "stopped"
        assert CheckpointStore(tmp_path / "checkpoints").steps() == [1]

    def test_a_stop_at_the_end_is_the_end(self, tmp_path: Path) -> None:
        result = simulate(_spec(), tmp_path, monitors=[_StopAfter(5)])
        assert result.status == "finished"
        assert result.checkpoint_path is None
        ds = io.load_dataset(tmp_path / "snapshots.zarr")
        assert ds["time"][-1] == 600.0 and ds.sizes["time"] == 7
        assert (tmp_path / "final_state.zarr").is_dir()

    def test_one_that_is_not_resumable_is_a_failure(self, tmp_path: Path) -> None:
        monitors = [_StopAfter(2), _StopAfter(2, resumable=False)]
        with pytest.raises(IntegrationDivergedError, match="enough"):
            simulate(_spec(), tmp_path, monitors=monitors)
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "manifest.json").exists()


@pytest.mark.integration
def test_a_killed_run_restarts(tmp_path: Path) -> None:
    """SIGKILL in the middle of a write; the newest complete checkpoint restarts."""
    code = """
    import os, signal
    from sesmax_core import io
    from sesmax_experiment.run import every_chunks
    from sesmax_experiment.sim import Checkpointing, simulate
    from run_specs import shallow_water

    save = io.save_dataset

    def killed(ds, path, **kwargs):
        save(ds, path, **kwargs)
        if path.name == "00000004.partial.zarr":
            os.kill(os.getpid(), signal.SIGKILL)

    io.save_dataset = killed
    spec = shallow_water(t1=600.0, save_interval=100.0)
    simulate(spec, "run", checkpointing=Checkpointing(every_chunks(1)))
    """
    env = os.environ | {"PYTHONPATH": str(Path(__file__).parents[1] / "support")}
    process = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert process.returncode == -signal.SIGKILL, process.stderr
    store = CheckpointStore(tmp_path / "run" / "checkpoints")
    assert (store.directory / "00000004.partial.zarr").exists()
    assert store.steps() == [1, 2, 3]

    whole = simulate(_spec(), tmp_path / "all")
    rest = restart(_spec(), tmp_path / "rest", restart_from=tmp_path / "run")
    np.testing.assert_array_equal(rest.state.h, whole.state.h)
