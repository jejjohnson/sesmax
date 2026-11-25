"""The builders of the models on the sphere.

Moved from somax 9cb19b3: ``somax/_src/cli/models_registry/spherical_swm.py`` and
``spherical_qg.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from gridax import Mask2D, apply_mask
from jaxtyping import Float

from sesmax_core import Model, State
from sesmax_ocean._src.scenarios.blocks import (
    known,
    model_params,
    no_layers,
    read,
    scheme,
    starts_from,
    wind_profile,
)
from sesmax_ocean._src.scenarios.types import ModelEntry, Scenario
from sesmax_ocean._src.spherical.qg import SphericalQG, SphericalQGState
from sesmax_ocean._src.spherical.swm import SphericalSWM, SphericalSWMState


_FRICTION = {"lateral_viscosity": float, "bottom_drag": float, "method": scheme}


def _with_ghosts(
    name: str, field: Any, scenario: Scenario
) -> Float[np.ndarray, "Ny Nx"]:
    """A field of the interior cells with its ghost ring, as the models fill it.

    It wraps in longitude and repeats the edge row in latitude. A field that has its
    ring already is left alone.

    Raises:
        ValueError: If the field has another shape.
    """
    array = np.asarray(field)
    ny, nx = scenario.geometry.ny, scenario.geometry.nx
    if array.shape == (ny + 2, nx + 2):
        return array
    if array.shape != (ny, nx):
        raise ValueError(
            f"{name} has shape {array.shape}; expected the physical grid "
            f"{(ny, nx)} or the ghosted grid {(ny + 2, nx + 2)}"
        )
    array = np.pad(array, ((0, 0), (1, 1)), mode="wrap")
    return np.pad(array, ((1, 1), (0, 0)), mode="edge")


def _band(name: str, scenario: Scenario) -> dict[str, Any]:
    """The grid and the mask of a scenario, as arguments of ``create``.

    Raises:
        ValueError: If the geometry has no bounds.
        NotImplementedError: If the scenario has a field no builder here reads.
    """
    geometry = scenario.geometry
    if geometry.lon_bounds is None or geometry.lat_bounds is None:
        raise ValueError(
            f"{name} requires a spherical geometry, with lon_bounds and lat_bounds; "
            f"got kind={geometry.kind!r}"
        )
    fields = {
        "geometry.bathymetry": geometry.bathymetry,
        "forcing.heat": scenario.forcing.heat,
    }
    if unread := [key for key, value in fields.items() if value is not None]:
        raise NotImplementedError(f"{name} reads no {unread} of a scenario yet")
    mask = None
    if geometry.mask is not None:
        wet = _with_ghosts(f"{name}: geometry.mask", geometry.mask, scenario) > 0.5
        mask = Mask2D.from_mask(wet)
    return {
        "nx": geometry.nx,
        "ny": geometry.ny,
        "lon_range": geometry.lon_bounds,
        "lat_range": geometry.lat_bounds,
        "mask": mask,
    }


def _wind(name: str, scenario: Scenario, names: Mapping[str, str]) -> dict[str, Any]:
    """The wind of a scenario: its own fields, or else a profile of the model.

    A field carries its own magnitude, so the amplitude that multiplies it is one
    unless the scenario gives it.

    Args:
        name: The model.
        scenario: The scenario.
        names: ``field of ForcingFields -> argument of create``.

    Returns:
        Arguments of ``create``.
    """
    fields = {
        argument: jnp.asarray(_with_ghosts(f"{name}: forcing.{field}", value, scenario))
        for field, argument in names.items()
        if (value := getattr(scenario.forcing, field)) is not None
    }
    amplitude = {"wind_amplitude": 1.0} if fields else {}
    return {**fields, **amplitude, **scenario.forcing_params}


def _numbers(
    name: str, scenario: Scenario, nondim: Mapping[str, Any], allowed: set[str]
) -> dict[str, Any]:
    """The arguments of ``from_nondimensional``."""
    known(f"{name}: scenario.nondim", nondim, allowed)
    if "rossby" not in nondim:
        raise ValueError(f"{name}: scenario.nondim requires 'rossby'")
    starts_from(name, scenario, "at_rest")
    if scenario.forcing.tau_x is not None or scenario.forcing.tau_y is not None:
        raise NotImplementedError(
            f"{name}: a dimensionless run has no wind of a scenario's own"
        )
    return {**_band(name, scenario), **wind_profile(scenario), **nondim}


def _resting_layer(model: SphericalSWM) -> SphericalSWMState:
    """A state at rest, dry on land: what stands there would count as mass."""
    depth = jnp.full((model.grid.Ny, model.grid.Nx), model.consts.H0)
    if model.mask is not None:
        depth = apply_mask(depth, model.mask, at="T")
    return SphericalSWMState(h=depth, u=jnp.zeros_like(depth), v=jnp.zeros_like(depth))


def _no_vorticity(model: SphericalQG) -> SphericalQGState:
    return SphericalQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))


def _spherical_swm(scenario: Scenario, spec: Mapping[str, Any]) -> tuple[Model, State]:
    name = "spherical_swm"
    starts_from(name, scenario, "at_rest")
    # One layer: its depth is a parameter, or the stratification block's.
    params = model_params(name, spec, {"H0": float, **_FRICTION})
    depth = read(
        f"{name}: model.stratification", spec.get("stratification", {}), {"H0": float}
    )
    model = SphericalSWM.create(
        **_band(name, scenario),
        **_wind(name, scenario, {"tau_x": "wind_stress_x", "tau_y": "wind_stress_y"}),
        g=scenario.constants.g,
        **{**depth, **params},
    )
    return model, _resting_layer(model)


def _spherical_swm_nondimensional(
    scenario: Scenario, nondim: Mapping[str, Any]
) -> tuple[Model, State]:
    name = "spherical_swm"
    stratification = {"burger", "froude", "lamb"}
    if not stratification & set(nondim):
        raise ValueError(
            f"{name}: scenario.nondim requires one of {sorted(stratification)}"
        )
    others = {"rossby", "ekman", "ekman_lateral", "wind_hat", "check_resolution"}
    numbers = _numbers(name, scenario, nondim, stratification | others)
    model, _ = SphericalSWM.from_nondimensional(**numbers)
    return model, _resting_layer(model)


def _spherical_qg(scenario: Scenario, spec: Mapping[str, Any]) -> tuple[Model, State]:
    name = "spherical_qg"
    no_layers(name, spec)
    starts_from(name, scenario, "at_rest")
    if scenario.forcing.tau_y is not None:
        raise ValueError(
            "spherical_qg is forced by the curl of the wind stress, one field, which "
            "a scenario gives as 'tau_x'; it has no use for 'tau_y'"
        )
    readers = {**_FRICTION, "cg_tol": float, "cg_max_steps": int}
    params = model_params(name, spec, readers)
    model = SphericalQG.create(
        **_band(name, scenario),
        **_wind(name, scenario, {"tau_x": "wind_forcing"}),
        **params,
    )
    return model, _no_vorticity(model)


def _spherical_qg_nondimensional(
    scenario: Scenario, nondim: Mapping[str, Any]
) -> tuple[Model, State]:
    allowed = {"rossby", "ekman", "ekman_lateral", "wind_hat"}
    numbers = _numbers("spherical_qg", scenario, nondim, allowed)
    model, _ = SphericalQG.from_nondimensional(**numbers)
    return model, _no_vorticity(model)


#: The models on the sphere.
SPHERICAL = (
    ModelEntry(
        "spherical_qg", "spherical", True, _spherical_qg, _spherical_qg_nondimensional
    ),
    ModelEntry(
        "spherical_swm",
        "spherical",
        True,
        _spherical_swm,
        _spherical_swm_nondimensional,
    ),
)
