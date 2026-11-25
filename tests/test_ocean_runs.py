"""The ocean's run configs through the run loop, which finds it as a component.

Here, and not with either package: neither imports the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sesmax_experiment.components import components, find_model
from sesmax_experiment.sim import app, build_run, load_yaml, simulate


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "packages" / "sesmax-ocean" / "configs").glob("*.yaml"))
if not CONFIGS:
    # A built distribution ships tests/ without the workspace.
    pytest.skip("the configs are not present", allow_module_level=True)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda path: path.stem)
def test_a_config_builds(path: Path) -> None:
    spec = load_yaml(str(path))
    spec.validate()
    short = spec.with_debug_applied()
    short.validate()
    component, run = build_run(short)
    assert component.name == "ocean"
    assert run.coordinates == ("nondimensional" if spec.scenario.nondim else "si")
    assert set(spec.assertions) <= set(component.checks) | {"cfl", "bounded_metric"}
    # Written out when the spec says so: a spinup is for its final state alone.
    spinup = path.stem.startswith("spinup")
    assert spec.output.write_snapshots is spec.output.write_metrics is not spinup


@pytest.mark.parametrize(
    "name", ["doublegyre_bt_qg", "doublegyre_bt_qg_nondim", "swm_jet"]
)
def test_a_config_runs(name: str, tmp_path: Path) -> None:
    """Ten steps on its debug grid, with the checks it asks for."""
    (path,) = (path for path in CONFIGS if path.stem == name)
    spec = load_yaml(str(path)).with_debug_applied()
    spec.timestepping.t1 = 10 * spec.timestepping.dt
    spec.timestepping.save_interval = 5 * spec.timestepping.dt
    result = simulate(spec, tmp_path / name)
    assert result.status == "finished"
    metrics = json.loads((tmp_path / name / "metrics.json").read_text("utf-8"))
    assert metrics
    manifest = json.loads((tmp_path / name / "manifest.json").read_text("utf-8"))
    assert manifest["numerics"]["model_class"]


def test_no_two_components_offer_one_model() -> None:
    for component in components().values():
        for model in component.models:
            assert find_model(model)[0].name == component.name


def test_the_scenarios_are_listed(capsys: pytest.CaptureFixture[str]) -> None:
    app(["list-scenarios"], result_action="return_value")
    out = capsys.readouterr().out
    assert "ocean:\n  - double_gyre: barotropic_qg, multilayer_nonlinear_swm" in out
