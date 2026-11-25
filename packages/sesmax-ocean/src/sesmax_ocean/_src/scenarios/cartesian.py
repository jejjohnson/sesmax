"""The builders of the models on a plane.

Moved from somax 9cb19b3: ``somax/_src/cli/models_registry/``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from sesmax_core import Model, State
from sesmax_ocean._src.qg.baroclinic import BaroclinicQG, BaroclinicQGState
from sesmax_ocean._src.qg.barotropic import BarotropicQG, BarotropicQGState
from sesmax_ocean._src.qg.reparameterized import ReparameterizedQG
from sesmax_ocean._src.scenarios.blocks import (
    initial_params,
    known,
    layers,
    model_params,
    no_layers,
    scheme,
    starts_from,
    wind_profile,
)
from sesmax_ocean._src.scenarios.types import ModelEntry, Scenario
from sesmax_ocean._src.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
)
from sesmax_ocean._src.swm.nonlinear_2d import (
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


_FRICTION: dict[str, Callable[[Any], Any]] = {
    "lateral_viscosity": float,
    "bottom_drag": float,
}
_SHALLOW_WATER = {**_FRICTION, "bc": str, "method": scheme}


def _box(name: str, scenario: Scenario) -> dict[str, Any]:
    """The arguments every builder here takes from a scenario.

    Raises:
        ValueError: If the geometry has no lengths.
        NotImplementedError: If the scenario has a field no builder here reads.
    """
    geometry, forcing = scenario.geometry, scenario.forcing
    if geometry.Lx is None or geometry.Ly is None:
        raise ValueError(
            f"{name} requires a Cartesian geometry, with Lx and Ly; got "
            f"kind={geometry.kind!r}"
        )
    fields = {
        "geometry.mask": geometry.mask,
        "geometry.bathymetry": geometry.bathymetry,
        "forcing.tau_x": forcing.tau_x,
        "forcing.tau_y": forcing.tau_y,
        "forcing.heat": forcing.heat,
    }
    if unread := [key for key, value in fields.items() if value is not None]:
        raise NotImplementedError(f"{name} reads no {unread} of a scenario yet")
    return {
        "nx": geometry.nx,
        "ny": geometry.ny,
        "Lx": geometry.Lx,
        "Ly": geometry.Ly,
        "f0": scenario.constants.f0,
        "beta": scenario.constants.beta,
        **scenario.forcing_params,
    }


def _points(model: Any) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """``x`` and ``y``, ``i Δx`` and ``j Δy``, on every point of the array."""
    x = jnp.arange(model.grid.Nx) * model.grid.spacing("x")
    y = jnp.arange(model.grid.Ny) * model.grid.spacing("y")
    return tuple(jnp.meshgrid(x, y))


def _jet(
    name: str, scenario: Scenario, model: Any, box: Mapping[str, Any], *, speed: float
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """A Gaussian jet along the middle of the basin, and the wave that breaks it."""
    number = initial_params(
        name,
        scenario,
        {"jet_speed": speed, "jet_width": 5.0e4, "perturbation": 0.01},
    )
    Lx, Ly = box["Lx"], box["Ly"]
    X, Y = _points(model)
    profile = jnp.exp(-0.5 * ((Y - Ly / 2.0) / number["jet_width"]) ** 2)
    wave = number["perturbation"] * jnp.sin(4.0 * jnp.pi * X / Lx) * profile
    return number["jet_speed"] * profile, wave


def _nonlinear_swm(scenario: Scenario, spec: Mapping[str, Any]) -> tuple[Model, State]:
    name = "nonlinear_swm"
    no_layers(name, spec)
    params = model_params(name, spec, {"H0": float, **_SHALLOW_WATER})
    H0 = params.setdefault("H0", 1000.0)
    box = _box(name, scenario)
    model = NonlinearShallowWater2D.create(**box, g=scenario.constants.g, **params)
    rest = jnp.zeros((model.grid.Ny, model.grid.Nx))
    match starts_from(name, scenario, "at_rest", "jet", "gaussian_eddy"):
        case "at_rest":
            h, u, v = rest + H0, rest, rest
        case "jet":
            u, v = _jet(name, scenario, model, box, speed=1.0)
            h = rest + H0
        case _:
            Lx, Ly = box["Lx"], box["Ly"]
            number = initial_params(
                name,
                scenario,
                {"amplitude": 1.0, "sigma": 5.0e4, "x0": Lx / 2.0, "y0": Ly / 2.0},
            )
            X, Y = _points(model)
            distance = ((X - number["x0"]) / number["sigma"]) ** 2
            distance = distance + ((Y - number["y0"]) / number["sigma"]) ** 2
            h, u, v = H0 + number["amplitude"] * jnp.exp(-0.5 * distance), rest, rest
    return model, NonlinearSW2DState(h=h, u=u, v=v)


def _resting_layers(model: Any) -> MultilayerSW2DState:
    shape = (model.consts.n_layers, model.grid.Ny, model.grid.Nx)
    h = jnp.broadcast_to(model.strat.H[:, None, None], shape)
    return MultilayerSW2DState(h=h, u=jnp.zeros(shape), v=jnp.zeros(shape))


def _multilayer_nonlinear_swm(
    scenario: Scenario, spec: Mapping[str, Any]
) -> tuple[Model, State]:
    name = "multilayer_nonlinear_swm"
    H, g_prime = layers(name, spec.get("stratification", {}))
    box = _box(name, scenario)
    model = MultilayerShallowWater2D.create(
        **box,
        g=scenario.constants.g,
        n_layers=len(H),
        H=H,
        g_prime=g_prime,
        **model_params(name, spec, _SHALLOW_WATER),
    )
    state = _resting_layers(model)
    if starts_from(name, scenario, "at_rest", "jet") == "jet":
        # The jets of two layers are opposite, which makes a vertical shear.
        jet, wave = _jet(name, scenario, model, box, speed=0.5)
        signs = jnp.asarray([1.0, -1.0])[jnp.arange(len(H)) % 2]
        u = signs[:, None, None] * jet
        v = jnp.broadcast_to(wave, u.shape)
        state = MultilayerSW2DState(h=state.h, u=u, v=v)
    return model, state


def _multilayer_qg(scenario: Scenario, spec: Mapping[str, Any]) -> tuple[Model, State]:
    name = "multilayer_qg"
    H, g_prime = layers(name, spec.get("stratification", {}))
    starts_from(name, scenario, "at_rest")
    model = BaroclinicQG.create(
        **_box(name, scenario),
        n_layers=len(H),
        H=H,
        g_prime=g_prime,
        **model_params(name, spec, {**_FRICTION, "poisson_bc": str}),
    )
    q = jnp.zeros((len(H), model.grid.Ny, model.grid.Nx))
    return model, BaroclinicQGState(q=q)


def _reparam_multilayer_qg(
    scenario: Scenario, spec: Mapping[str, Any]
) -> tuple[Model, State]:
    name = "reparam_multilayer_qg"
    H, g_prime = layers(name, spec.get("stratification", {}))
    starts_from(name, scenario, "at_rest")
    model = ReparameterizedQG.create(
        **_box(name, scenario),
        g=scenario.constants.g,
        n_layers=len(H),
        H=H,
        g_prime=g_prime,
        **model_params(name, spec, {**_SHALLOW_WATER, "poisson_bc": str}),
    )
    return model, _resting_layers(model)


def _barotropic_qg(scenario: Scenario, spec: Mapping[str, Any]) -> tuple[Model, State]:
    name = "barotropic_qg"
    no_layers(name, spec)
    starts_from(name, scenario, "at_rest")
    model = BarotropicQG.create(
        **_box(name, scenario), **model_params(name, spec, _FRICTION)
    )
    return model, _no_vorticity(model)


def _no_vorticity(model: BarotropicQG) -> BarotropicQGState:
    return BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))


#: The keys of ``scenario.nondim``, and the arguments they are. ``munk`` and
#: ``stommel`` read better in a file than ``delta_M`` and ``delta_S``.
_NONDIM = {
    "rossby": "rossby",
    "beta_hat": "beta_hat",
    "munk": "delta_M",
    "stommel": "delta_S",
    "inertial": "delta_I",
    "aspect": "aspect",
    "check_resolution": "check_resolution",
}


def _barotropic_qg_nondimensional(
    scenario: Scenario, nondim: Mapping[str, Any]
) -> tuple[Model, State]:
    name = "barotropic_qg"
    known(f"{name}: scenario.nondim", nondim, _NONDIM)
    if missing := sorted({"rossby", "beta_hat", "munk"} - set(nondim)):
        raise ValueError(f"{name}: scenario.nondim missing required key(s) {missing}")
    starts_from(name, scenario, "at_rest")
    box = _box(name, scenario)
    numbers = {_NONDIM[key]: value for key, value in nondim.items()}
    # The shape of the basin is the geometry's; a second one has to agree.
    aspect = box["Ly"] / box["Lx"]
    if abs(numbers.setdefault("aspect", aspect) - aspect) > 1e-9 * max(1.0, aspect):
        raise ValueError(
            f"{name}: scenario.nondim.aspect={numbers['aspect']!r} contradicts the "
            f"geometry, whose Ly / Lx is {aspect!r}"
        )
    model, _ = BarotropicQG.from_nondimensional(
        nx=box["nx"], ny=box["ny"], **wind_profile(scenario), **numbers
    )
    return model, _no_vorticity(model)


#: The models on a plane.
CARTESIAN = (
    ModelEntry(
        "barotropic_qg",
        "cartesian",
        True,
        _barotropic_qg,
        _barotropic_qg_nondimensional,
    ),
    ModelEntry(
        "multilayer_nonlinear_swm", "cartesian", True, _multilayer_nonlinear_swm
    ),
    ModelEntry("multilayer_qg", "cartesian", True, _multilayer_qg),
    ModelEntry("nonlinear_swm", "cartesian", True, _nonlinear_swm),
    ModelEntry("reparam_multilayer_qg", "cartesian", True, _reparam_multilayer_qg),
)
