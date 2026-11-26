"""The builders of the models on a plane."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from gridax.schemes import WENO, Upwind

from sesmax_core.component import BuiltRun
from sesmax_core.fields import values
from sesmax_ocean.scenarios import (
    MODELS,
    ForcingFields,
    build_ocean_run,
    build_scenario,
)


LAYERED = ["multilayer_nonlinear_swm", "multilayer_qg", "reparam_multilayer_qg"]
ONE_LAYER = ["barotropic_qg", "nonlinear_swm"]
SHALLOW_WATER = ["multilayer_nonlinear_swm", "nonlinear_swm", "reparam_multilayer_qg"]
FRICTION = {"lateral_viscosity": 100.0, "bottom_drag": 1.0e-7}
LAYERS = {"H": [500.0, 4500.0], "g_prime": [9.81, 0.025]}


def _scenario(**blocks: Any) -> dict[str, Any]:
    return {
        "name": "double_gyre",
        "grid": {"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
        "consts": {"f0": 1.0e-4, "beta": 1.6e-11},
        **blocks,
    }


def _model(name: str, **blocks: Any) -> dict[str, Any]:
    layers = {"stratification": LAYERS} if name in LAYERED else {}
    return {"name": name, **layers, "params": FRICTION, **blocks}


def _run(name: str, scenario: dict[str, Any], **blocks: Any) -> BuiltRun:
    return build_ocean_run({"scenario": scenario, "model": _model(name, **blocks)})


@pytest.mark.parametrize("name", [*LAYERED, *ONE_LAYER])
def test_a_model_is_built_from_its_blocks(name: str) -> None:
    run = _run(name, _scenario(forcing={"wind_amplitude": 1.0e-12}))
    model = run.model
    assert run.coordinates == "si"
    assert model.consts.f0 == 1.0e-4
    assert float(model.params.lateral_viscosity) == pytest.approx(100.0)
    assert float(model.params.bottom_drag) == pytest.approx(1.0e-7)
    assert float(model.params.wind_amplitude) == pytest.approx(1.0e-12)
    if name in LAYERED:
        np.testing.assert_allclose(np.asarray(model.strat.H), LAYERS["H"])
    tendency = model.vector_field(0.0, run.state0)
    for field in dataclasses.fields(tendency):
        assert np.isfinite(values(getattr(tendency, field.name))).all()


@pytest.mark.parametrize("name", [*LAYERED, *ONE_LAYER])
def test_a_misspelt_parameter_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="lateral_viscocity"):
        _run(name, _scenario(), params={"lateral_viscocity": 10.0})


@pytest.mark.parametrize("name", SHALLOW_WATER)
def test_the_scheme_has_somax_s_name(name: str) -> None:
    def scheme(**params: str) -> Any:
        model = _run(name, _scenario(), params=params).model
        return getattr(model, "swm", model).scheme

    assert scheme() == Upwind(1)
    assert scheme(method="wenoz5") == WENO(5, variant="z")
    with pytest.raises(ValueError, match="wenoz5"):
        scheme(method="weno-z")


@pytest.mark.parametrize("name", LAYERED)
@pytest.mark.parametrize(
    ("stratification", "message"),
    [
        ({}, "'H', 'g_prime'"),
        ({"H": [500.0, 4500.0]}, "'g_prime'"),
        ({"H": [500.0, 4500.0], "g_prime": [9.81]}, "match the number of layers"),
        ({**LAYERS, "rho": [1.0, 2.0]}, "rho"),
    ],
    ids=["none", "no g_prime", "lengths", "unknown"],
)
def test_layers_need_their_numbers(
    name: str, stratification: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=rf"{name}.*{message}"):
        _run(name, _scenario(), stratification=stratification)


@pytest.mark.parametrize("name", ONE_LAYER)
def test_one_layer_reads_no_stratification(name: str) -> None:
    with pytest.raises(ValueError, match="one layer"):
        _run(name, _scenario(), stratification=LAYERS)


@pytest.mark.parametrize("name", [*LAYERED, "barotropic_qg"])
def test_an_initial_condition_a_model_has_not_is_refused(name: str) -> None:
    with pytest.raises(NotImplementedError, match="gaussian_eddy"):
        _run(name, _scenario(initial_condition={"type": "gaussian_eddy"}))


@pytest.mark.parametrize(
    "initial",
    [
        {"type": "at_rest", "params": {"speed": 1.0}},
        {"type": "jet", "params": {"u": 1}},
    ],
    ids=["at rest", "jet"],
)
def test_a_misspelt_number_of_an_initial_condition_is_refused(
    initial: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"initial_condition\.params"):
        _run("nonlinear_swm", _scenario(initial_condition=initial))


def test_an_eddy_is_a_bump_on_the_resting_layer() -> None:
    initial = {"type": "gaussian_eddy", "params": {"amplitude": 0.5, "sigma": 1.0e5}}
    state = _run(
        "nonlinear_swm", _scenario(initial_condition=initial), params={"H0": 100.0}
    ).state0
    h = np.asarray(state.h)
    assert h.max() == pytest.approx(100.5, abs=1e-2)
    assert h.min() == pytest.approx(100.0, abs=1e-3)
    assert float(jnp.abs(state.u).max()) == float(jnp.abs(state.v).max()) == 0.0


def test_the_jets_of_two_layers_are_opposite() -> None:
    initial = {"type": "jet", "params": {"jet_speed": 0.5, "perturbation": 0.0}}
    state = _run(
        "multilayer_nonlinear_swm", _scenario(initial_condition=initial)
    ).state0
    assert float(state.u.max()) == pytest.approx(0.5, rel=0.1)
    np.testing.assert_array_equal(state.u[0], -state.u[1])
    assert float(jnp.abs(state.v).max()) == 0.0


@pytest.mark.parametrize("name", [*LAYERED, *ONE_LAYER])
def test_a_field_no_builder_reads_is_refused(name: str) -> None:
    """Rather than a run without the land or the wind it was given."""
    scenario = build_scenario(_scenario())
    ones = jnp.ones((16, 16))
    geometry = dataclasses.replace(scenario.geometry, mask=ones)
    for changed in (
        dataclasses.replace(scenario, geometry=geometry),
        dataclasses.replace(scenario, forcing=ForcingFields(tau_x=ones)),
    ):
        with pytest.raises(NotImplementedError, match=r"mask|tau_x"):
            MODELS[name].build(changed, _model(name))


@pytest.mark.parametrize("name", ["spherical_qg", "spherical_swm"])
def test_a_model_on_the_sphere_needs_its_bounds(name: str) -> None:
    with pytest.raises(ValueError, match="lon_bounds"):
        MODELS[name].build(build_scenario(_scenario()), {"name": name})
