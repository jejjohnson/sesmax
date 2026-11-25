"""The builders of the models on the sphere."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from gridax import Mask2D
from gridax.schemes import Upwind

from sesmax_ocean.scenarios import (
    MODELS,
    Constants,
    ForcingFields,
    Geometry,
    InitialCondition,
    Scenario,
)


BOTH = ["spherical_qg", "spherical_swm"]
#: A field of a scenario is on the interior cells; a model has a ring of ghosts.
INTERIOR, WITH_GHOSTS = (8, 16), (10, 18)


def _wet() -> np.ndarray:
    wet = np.ones(INTERIOR)
    wet[2:4, 3:6] = 0.0
    return wet


def _scenario(
    *, mask: Any = None, g: float = 9.81, kind: Any = "at_rest", **forcing: Any
) -> Scenario:
    fields = {k: v for k, v in forcing.items() if k in ("tau_x", "tau_y", "heat")}
    return Scenario(
        name="band",
        geometry=Geometry(
            kind="spherical_cap",
            nx=16,
            ny=8,
            lon_bounds=(0.0, 360.0),
            lat_bounds=(-70.0, -40.0),
            mask=mask,
        ),
        constants=Constants(g=g),
        forcing=ForcingFields(**fields),
        initial_condition=InitialCondition(kind),
        forcing_params={k: v for k, v in forcing.items() if k not in fields},
    )


def _build(name: str, scenario: Scenario | None = None, **model: Any) -> Any:
    return MODELS[name].build(scenario or _scenario(), {"name": name, **model})


@pytest.mark.parametrize("name", BOTH)
def test_a_model_is_built_from_its_blocks(name: str) -> None:
    params = {"lateral_viscosity": 1e4, "bottom_drag": 1e-7, "method": "upwind3"}
    model, state = _build(name, params=params)
    assert float(model.params.lateral_viscosity) == pytest.approx(1e4)
    assert model.scheme == Upwind(3)
    assert model.mask is None
    model.vector_field(0.0, state)
    with pytest.raises(ValueError, match="drag"):
        _build(name, params={"drag": 1.0})
    with pytest.raises(NotImplementedError, match="jet"):
        _build(name, _scenario(kind="jet"))


def test_the_gravity_is_the_scenario_s() -> None:
    model, _ = _build("spherical_swm", _scenario(g=3.71))
    assert float(model.consts.gravity) == pytest.approx(3.71)


@pytest.mark.parametrize("block", ["params", "stratification"])
def test_the_depth_is_in_either_block(block: str) -> None:
    model, state = _build("spherical_swm", **{block: {"H0": 250.0}})
    assert float(model.consts.H0) == pytest.approx(250.0)
    assert float(jnp.max(state.h)) == pytest.approx(250.0)


@pytest.mark.parametrize(("key", "value"), [("cg_tol", 1e-8), ("cg_max_steps", 123)])
def test_the_settings_of_the_solver_reach_the_model(key: str, value: float) -> None:
    model, _ = _build("spherical_qg", params={key: value})
    assert getattr(model, key) == value


@pytest.mark.parametrize("name", BOTH)
def test_the_mask_of_a_scenario_gets_its_ghosts(name: str) -> None:
    model, state = _build(name, _scenario(mask=_wet()))
    assert isinstance(model.mask, Mask2D)
    wet = np.asarray(model.mask.wet(at="T"))
    assert wet.shape == WITH_GHOSTS
    np.testing.assert_array_equal(wet[1:-1, 1:-1], _wet() > 0.5)
    tendency = model.vector_field(0.0, state)
    field = tendency.q if name == "spherical_qg" else tendency.h
    assert np.isfinite(np.asarray(field)).all()


def test_a_resting_layer_is_dry_on_land() -> None:
    """What stands there would stay in the saved states, and count as mass."""
    model, state = _build("spherical_swm", _scenario(mask=_wet()))
    wet = np.asarray(model.mask.wet(at="T"))
    assert not wet.all()
    h = np.asarray(state.h)
    np.testing.assert_array_equal(h[~wet], 0.0)
    np.testing.assert_allclose(h[wet], model.consts.H0)
    open_model, open_state = _build("spherical_swm")
    mass = float(model.diagnose(state).mass)
    assert mass < float(open_model.diagnose(open_state).mass)


@pytest.mark.parametrize("shape", [INTERIOR, WITH_GHOSTS], ids=["interior", "ghosts"])
@pytest.mark.parametrize(
    ("name", "field", "argument"),
    [
        ("spherical_swm", "tau_x", "wind_stress_x"),
        ("spherical_swm", "tau_y", "wind_stress_y"),
        ("spherical_qg", "tau_x", "wind_forcing"),
    ],
)
def test_the_wind_of_a_scenario_reaches_the_model(
    name: str, field: str, argument: str, shape: tuple[int, int]
) -> None:
    model, state = _build(name, _scenario(**{field: np.full(shape, 0.25)}))
    wind = np.asarray(getattr(model, argument))
    assert wind.shape == WITH_GHOSTS
    np.testing.assert_allclose(wind, 0.25)
    # A field carries its own magnitude.
    assert float(model.params.wind_amplitude) == 1.0
    model.vector_field(0.0, state)


def test_a_component_without_a_field_is_zero() -> None:
    """Not the zonal profile of the model, which the field replaces."""
    model, _ = _build("spherical_swm", _scenario(tau_y=np.full(INTERIOR, 0.25)))
    np.testing.assert_array_equal(np.asarray(model.wind_stress_x), 0.0)
    model, _ = _build("spherical_swm")
    assert np.abs(np.asarray(model.wind_stress_x)).max() > 0.5


@pytest.mark.parametrize("name", BOTH)
def test_the_amplitude_of_the_wind(name: str) -> None:
    def amplitude(**forcing: Any) -> float:
        model, _ = _build(name, _scenario(**forcing))
        return float(model.params.wind_amplitude)

    assert amplitude() == 0.0
    assert amplitude(wind_amplitude=3.0) == 3.0
    assert amplitude(wind_amplitude=3.0, tau_x=np.ones(INTERIOR)) == 3.0


@pytest.mark.parametrize("name", BOTH)
def test_a_field_of_another_shape_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="expected the physical grid"):
        _build(name, _scenario(tau_x=np.zeros((5, 5))))


@pytest.mark.parametrize("name", BOTH)
def test_a_field_no_builder_reads_is_refused(name: str) -> None:
    with pytest.raises(NotImplementedError, match="heat"):
        _build(name, _scenario(heat=np.zeros(INTERIOR)))


def test_the_vorticity_takes_one_field_of_wind() -> None:
    with pytest.raises(ValueError, match="no use for 'tau_y'"):
        _build("spherical_qg", _scenario(tau_y=np.zeros(INTERIOR)))


@pytest.mark.parametrize(
    ("name", "numbers"),
    [
        ("spherical_qg", {"rossby": 0.1, "ekman": 0.01}),
        ("spherical_swm", {"rossby": 0.1, "burger": 0.1, "check_resolution": False}),
    ],
)
def test_dimensionless_numbers(name: str, numbers: dict[str, Any]) -> None:
    from_nondimensional = MODELS[name].from_nondimensional
    assert from_nondimensional is not None
    model, state = from_nondimensional(_scenario(mask=_wet()), numbers)
    assert model.consts.radius == 1.0
    assert isinstance(model.mask, Mask2D)
    if name == "spherical_swm":
        assert float(state.h.min()) == 0.0
    with pytest.raises(ValueError, match="rosby"):
        from_nondimensional(_scenario(), {**numbers, "rosby": 1.0})
    without = {key: value for key, value in numbers.items() if key != "rossby"}
    with pytest.raises(ValueError, match="rossby"):
        from_nondimensional(_scenario(), without)
    with pytest.raises(NotImplementedError, match="wind"):
        from_nondimensional(_scenario(tau_x=np.ones(INTERIOR)), numbers)


def test_dimensionless_shallow_water_needs_its_stratification() -> None:
    from_nondimensional = MODELS["spherical_swm"].from_nondimensional
    assert from_nondimensional is not None
    with pytest.raises(ValueError, match="'burger', 'froude', 'lamb'"):
        from_nondimensional(_scenario(), {"rossby": 0.1})
