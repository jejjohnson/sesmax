"""Which scenario goes with which model, and the run the two make."""

from __future__ import annotations

from typing import Any

import pytest

from sesmax_ocean._src.scenarios.pairs import why_not
from sesmax_ocean.scenarios import (
    MODELS,
    SCENARIOS,
    IncompatiblePairError,
    ModelEntry,
    build_ocean_run,
    check_compatible,
    compatible_models,
    get_model,
)


PLANE = {
    "barotropic_qg",
    "multilayer_nonlinear_swm",
    "multilayer_qg",
    "nonlinear_swm",
    "reparam_multilayer_qg",
}
SPHERE = {"spherical_qg", "spherical_swm"}
PAIRS = {
    "double_gyre": PLANE,
    "global_ocean": set(),
    "gulf_stream": PLANE,
    "med_sea": PLANE,
    "north_atlantic": PLANE,
    "southern_ocean": SPHERE,
}


def _spec(**blocks: Any) -> dict[str, Any]:
    scenario = {"name": "double_gyre", "grid": {"nx": 64, "ny": 64, "Lx": 1, "Ly": 1}}
    nondim = {"rossby": 0.02, "beta_hat": 50.0, "munk": 0.05}
    return {
        "scenario": {**scenario, "nondim": nondim, **blocks},
        "model": {"name": "barotropic_qg"},
    }


def test_the_models_are_somax_s() -> None:
    """All but the linear shallow-water model, which is the testing component's."""
    assert set(MODELS) == PLANE | SPHERE
    for name, entry in MODELS.items():
        assert entry.name == name
        assert entry.coordinates == ("spherical" if name in SPHERE else "cartesian")
        assert entry.masks


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_the_pairs(scenario: str) -> None:
    assert set(compatible_models(scenario)) == PAIRS[scenario]
    for model in MODELS:
        if model in PAIRS[scenario]:
            check_compatible(scenario, model)
        else:
            with pytest.raises(IncompatiblePairError, match=f"{scenario}.*{model}"):
                check_compatible(scenario, model)


def test_the_reason_is_in_the_message() -> None:
    with pytest.raises(IncompatiblePairError, match="needs cartesian coordinates"):
        check_compatible("double_gyre", "spherical_qg")
    with pytest.raises(IncompatiblePairError, match="needs spherical coordinates"):
        check_compatible("southern_ocean", "nonlinear_swm")
    with pytest.raises(IncompatiblePairError, match="reaches a pole"):
        check_compatible("global_ocean", "spherical_swm")


def test_a_model_without_masks_takes_no_land() -> None:
    open_water = ModelEntry("open", "cartesian", False, build=lambda *a: None)
    assert why_not(SCENARIOS["double_gyre"], open_water) is None
    assert "land" in (why_not(SCENARIOS["med_sea"], open_water) or "")


def test_an_unknown_name_is_an_error() -> None:
    with pytest.raises(KeyError, match="no_such_scenario"):
        check_compatible("no_such_scenario", "barotropic_qg")
    with pytest.raises(KeyError, match=r"barotropic_qg.*spherical_swm"):
        get_model("no_such_model")


def test_a_run_of_a_pair_that_does_not_go_is_refused() -> None:
    spec = _spec()
    spec["model"]["name"] = "spherical_qg"
    with pytest.raises(IncompatiblePairError):
        build_ocean_run(spec)


def test_dimensionless_numbers_make_a_dimensionless_run() -> None:
    run = build_ocean_run(_spec(forcing={"wind_profile": "single"}))
    assert run.coordinates == "nondimensional"
    assert run.model.consts.f0 == pytest.approx(50.0)
    assert float(run.model.params.lateral_viscosity) == pytest.approx(0.05**3 * 50.0)
    assert build_ocean_run(_spec(nondim={})).coordinates == "si"


def test_the_shape_of_a_dimensionless_basin_is_the_geometry_s() -> None:
    spec = _spec(grid={"nx": 64, "ny": 32, "Lx": 1.0, "Ly": 0.5})
    grid = build_ocean_run(spec).model.grid
    assert float(grid.spacing("y")) == pytest.approx(float(grid.spacing("x")))
    spec["scenario"]["nondim"]["aspect"] = 1.0
    with pytest.raises(ValueError, match="contradicts"):
        build_ocean_run(spec)


@pytest.mark.parametrize(
    ("blocks", "message"),
    [
        ({"nondim": {"rosby": 0.02}}, "rosby"),
        ({"nondim": {"rossby": 0.02}}, "beta_hat.*munk"),
        ({"forcing": {"wind_amplitude": 1.0}}, "wind_amplitude"),
    ],
)
def test_a_wrong_dimensionless_block_is_refused(
    blocks: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_ocean_run(_spec(**blocks))


@pytest.mark.parametrize(
    "block", [{"params": {"lateral_viscosity": 1.0}}, {"stratification": {"H": [1.0]}}]
)
def test_dimensionless_numbers_go_with_no_others(block: dict[str, Any]) -> None:
    spec = _spec()
    spec["model"].update(block)
    with pytest.raises(ValueError, match="cannot be combined"):
        build_ocean_run(spec)


def test_a_model_without_dimensionless_numbers_refuses_them() -> None:
    spec = _spec()
    spec["model"]["name"] = "nonlinear_swm"
    with pytest.raises(ValueError, match="not built from dimensionless numbers"):
        build_ocean_run(spec)
