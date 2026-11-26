"""The ocean is a component a run can name."""

from __future__ import annotations

import dataclasses
import json
from importlib.metadata import entry_points
from typing import Any

import pytest

from sesmax_core.component import Component
from sesmax_ocean.scenarios import MODELS, SCENARIOS, build_ocean_run, compatible_models


def _component() -> Component:
    (entry,) = entry_points(group="sesmax.components", name="ocean")
    return entry.load()


def _spec(name: str) -> dict[str, Any]:
    layers = {"H": [500.0, 4500.0], "g_prime": [9.81, 0.025]}
    return {
        "scenario": {"name": "double_gyre", "grid": {"nx": 16, "ny": 16}},
        "model": {
            "name": name,
            "stratification": layers if "multilayer" in name else {},
            "params": {"lateral_viscosity": 500.0},
        },
    }


def test_the_ocean_is_installed_as_a_component() -> None:
    ocean = _component()
    assert ocean.name == "ocean"
    assert set(ocean.models) == set(ocean.field_units) == set(MODELS)
    assert set(ocean.checks) == {
        "deformation_radius",
        "equatorial_deformation_radius",
        "munk_width",
        "pv_inversion",
        "static_stability",
        "stommel_width",
    }


def test_it_offers_the_scenarios_that_can_be_built() -> None:
    ocean = _component()
    built = {name for name, entry in SCENARIOS.items() if entry.build is not None}
    assert set(ocean.scenarios) == built == {"double_gyre"}
    assert "barotropic_qg" in ocean.scenarios["double_gyre"]
    assert "spherical_qg" not in ocean.scenarios["double_gyre"]


@pytest.mark.parametrize("name", compatible_models("double_gyre"))
def test_every_field_has_a_unit(name: str) -> None:
    state = build_ocean_run(_spec(name)).state0
    fields = {field.name for field in dataclasses.fields(state)}
    assert set(_component().field_units[name]) == fields


@pytest.mark.parametrize("name", compatible_models("double_gyre"))
def test_the_numerics_are_for_a_manifest(name: str) -> None:
    model = build_ocean_run(_spec(name)).model
    describe = _component().describe_numerics
    assert describe is not None
    numerics = describe(model)
    assert numerics["model_class"] == type(model).__name__
    assert numerics["lateral_viscosity"] == pytest.approx(500.0)
    if "swm" in name or name.startswith("reparam"):
        assert numerics["scheme"].startswith("Upwind(order=1)")
    json.dumps(numerics)


def test_the_metrics_are_the_ocean_s() -> None:
    run = build_ocean_run(_spec("barotropic_qg"))
    evaluate = _component().evaluate
    assert evaluate is not None
    assert evaluate(run.model, run.state0)["qg_balance_residual"] == 0.0
