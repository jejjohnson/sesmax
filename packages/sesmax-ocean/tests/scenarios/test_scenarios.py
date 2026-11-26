"""The scenarios: what they build, and what they refuse."""

from __future__ import annotations

from typing import Any, get_args

import pytest

from sesmax_ocean._src.scenarios.types import GeometryKind
from sesmax_ocean.scenarios import (
    SCENARIOS,
    Scenario,
    build_scenario,
    get_scenario,
)


def _double_gyre(**blocks: Any) -> Scenario:
    grid = {"nx": 8, "ny": 6, "Lx": 1.0e6, "Ly": 8.0e5}
    return build_scenario({"name": "double_gyre", "grid": grid, **blocks})


def test_the_scenarios_are_somax_s() -> None:
    assert {name: entry.geometry_kind for name, entry in SCENARIOS.items()} == {
        "double_gyre": "rectangular",
        "global_ocean": "spherical_cap",
        "gulf_stream": "real_basin",
        "med_sea": "real_basin",
        "north_atlantic": "real_basin",
        "southern_ocean": "spherical_cap",
    }
    for name, entry in SCENARIOS.items():
        assert entry.name == name
        assert entry.geometry_kind in get_args(GeometryKind)


def test_the_double_gyre_is_the_block() -> None:
    built = _double_gyre(
        consts={"f0": 9.0e-5, "beta": 2.0e-11},
        forcing={"wind_amplitude": "1e-12", "wind_profile": "single"},
        initial_condition={"type": "jet", "params": {"jet_speed": 0.5}},
    )
    assert (built.geometry.kind, built.geometry.nx, built.geometry.ny) == (
        "rectangular",
        8,
        6,
    )
    assert (built.geometry.Lx, built.geometry.Ly) == (1.0e6, 8.0e5)
    assert (built.constants.f0, built.constants.beta) == (9.0e-5, 2.0e-11)
    assert built.constants.g == 9.81
    # A number YAML reads as a string, because it has no decimal point.
    assert built.forcing_params == {
        "wind_amplitude": 1.0e-12,
        "wind_profile": "single",
    }
    assert built.initial_condition.kind == "jet"
    assert built.initial_condition.params == {"jet_speed": 0.5}


def test_the_double_gyre_has_defaults() -> None:
    built = build_scenario({"name": "double_gyre", "grid": {"nx": 8, "ny": 8}})
    assert (built.geometry.Lx, built.geometry.Ly) == (1.0e6, 1.0e6)
    assert (built.constants.f0, built.constants.beta) == (1.0e-4, 1.6e-11)
    assert built.initial_condition.kind == "at_rest"
    assert dict(built.forcing_params) == {}


@pytest.mark.parametrize("missing", ["nx", "ny"])
def test_the_double_gyre_needs_its_cells(missing: str) -> None:
    grid = {"nx": 8, "ny": 8}
    del grid[missing]
    with pytest.raises(ValueError, match=missing):
        build_scenario({"name": "double_gyre", "grid": grid})


@pytest.mark.parametrize(
    "block",
    [
        {"forcing": {"wing_amplitude": 1.0e-12}},
        {"consts": {"f_0": 1.0e-4}},
        {"grid": {"nx": 8, "ny": 8, "Lz": 1.0}},
        {"initial_condition": {"kind": "jet"}},
        {"initial_condition": {"type": "vortex"}},
    ],
    ids=str,
)
def test_a_misspelt_block_is_refused(block: dict[str, Any]) -> None:
    """It would leave a default in place, and the run would be another one."""
    with pytest.raises(ValueError, match="unknown"):
        _double_gyre(**block)


@pytest.mark.parametrize(
    "name", [name for name, entry in SCENARIOS.items() if entry.build is None]
)
def test_a_planned_scenario_says_what_it_needs(name: str) -> None:
    with pytest.raises(NotImplementedError, match=rf"{name}.* needs .*sesmax#115"):
        build_scenario({"name": name})


def test_an_unknown_scenario_lists_those_there_are() -> None:
    with pytest.raises(KeyError, match=r"double_gyre.*southern_ocean"):
        get_scenario("no_such_scenario")


def test_the_forcing_of_a_scenario_is_copied_and_read_only() -> None:
    source = {"wind_amplitude": 1.0e-12}
    built = _double_gyre(forcing=source)
    source["wind_amplitude"] = 999.0
    assert built.forcing_params["wind_amplitude"] == 1.0e-12
    with pytest.raises(TypeError):
        built.forcing_params["wind_amplitude"] = 2.0e-12  # type: ignore[index]
