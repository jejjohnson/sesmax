"""The command line: somax's commands and arguments."""

from __future__ import annotations

from pathlib import Path

import pytest
from run_specs import shallow_water

from sesmax_experiment.sim import DebugSpec, app, dump_yaml, load_yaml


@pytest.fixture
def config(tmp_path: Path) -> Path:
    spec = shallow_water(debug=DebugSpec(scenario={"grid": {"nx": 8, "ny": 8}}))
    path = tmp_path / "spec.yaml"
    dump_yaml(spec, str(path))
    return path


def _run(*args: object) -> None:
    app([str(a) for a in args], result_action="return_value")


def test_simulate_and_restart(config: Path, tmp_path: Path) -> None:
    _run("spinup", "--config", config, "--output-dir", tmp_path / "spinup")
    state = tmp_path / "spinup" / "final_state.zarr"
    _run(
        "restart",
        "--config",
        config,
        "--from",
        state,
        "--output-dir",
        tmp_path / "prod",
    )
    _run("simulate", "--config", config, "--output-dir", tmp_path / "run", "--debug")
    assert (tmp_path / "prod" / "snapshots.zarr").is_dir()
    resolved = load_yaml(str(tmp_path / "run" / "resolved.yaml"))
    assert resolved.scenario.grid["nx"] == 8


def test_diagnostics_per_save_reaches_the_run(config: Path, tmp_path: Path) -> None:
    _run(
        "simulate",
        *("--config", config, "--output-dir", tmp_path / "run"),
        *("--diagnostics-per-save", 3),
    )
    assert "chunk 3/3" in (tmp_path / "run" / "run.log").read_text()


def test_list_models(capsys: pytest.CaptureFixture[str]) -> None:
    _run("list-models")
    assert "testing:\n  - linear_swm\n  - lorenz96" in capsys.readouterr().out


def test_show_config(config: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run("show-config", config)
    assert "name: linear_swm" in capsys.readouterr().out
