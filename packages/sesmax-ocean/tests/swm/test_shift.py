"""A shifted state has the shifted tendency, where the box is periodic.

So the edge of the box is not a place of its own: every ghost that a tendency
reads has been filled.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from gridax import Mask2D
from gridax.schemes import WENO, Upwind
from somax_reference import float64

from sesmax_ocean.spherical import SphericalSWM, SphericalSWMState
from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


NY, NX = 12, 16
#: ``name -> (model, state, layers, arguments of create, shift in y and x)``
MODELS: dict[str, tuple[Any, Any, tuple[int, ...], dict[str, Any], tuple[int, int]]] = {
    "nonlinear": (
        NonlinearShallowWater2D,
        NonlinearSW2DState,
        (),
        {"Lx": 1e6, "Ly": 8e5},
        (5, 7),
    ),
    "multilayer": (
        MultilayerShallowWater2D,
        MultilayerSW2DState,
        (3,),
        {"Lx": 1e6, "Ly": 8e5, "beta": 0.0},
        (5, 7),
    ),
    "spherical": (
        SphericalSWM,
        SphericalSWMState,
        (),
        {"wind_profile": "none"},
        (0, 7),
    ),
}


def _padded(interior: np.ndarray) -> np.ndarray:
    return np.pad(interior, [(0, 0)] * (interior.ndim - 2) + [(1, 1)] * 2, mode="wrap")


def _tendency(name: str, scheme: Any, interiors: dict[str, np.ndarray], wet: Any):
    model_class, state_class, _, kwargs, _ = MODELS[name]
    mask = None if wet is None else Mask2D.from_center(_padded(wet))
    model = model_class.create(nx=NX, ny=NY, scheme=scheme, mask=mask, **kwargs)
    state = state_class(**{k: jnp.asarray(_padded(x)) for k, x in interiors.items()})
    tendency = model.vector_field(0.0, model.apply_boundary_conditions(state))
    return {
        name: np.asarray(getattr(tendency, name))[..., 1:-1, 1:-1] for name in "huv"
    }


@pytest.mark.parametrize("has_island", [False, True], ids=["nomask", "island"])
@pytest.mark.parametrize("scheme", [Upwind(1), WENO(5)], ids=["upwind1", "weno5"])
@pytest.mark.parametrize("name", list(MODELS))
def test_a_shifted_state_has_the_shifted_tendency(name, scheme, has_island) -> None:
    _, _, layers, _, shift = MODELS[name]
    rng = np.random.default_rng(0)
    interiors = {
        "h": 500.0 + rng.normal(size=(*layers, NY, NX)),
        "u": 0.1 * rng.normal(size=(*layers, NY, NX)),
        "v": 0.1 * rng.normal(size=(*layers, NY, NX)),
    }
    wet = None
    if has_island:
        wet = np.ones((NY, NX), dtype=bool)
        wet[4:7, 5:8] = False
    if name == "spherical":
        # The walls of the band: no flow through the northern one.
        interiors["v"][-1] = 0.0

    def rolled(x: np.ndarray) -> np.ndarray:
        return np.roll(x, shift, axis=(-2, -1))

    with float64():
        here = _tendency(name, scheme, interiors, wet)
        there = _tendency(
            name,
            scheme,
            {k: rolled(x) for k, x in interiors.items()},
            None if wet is None else rolled(wet),
        )
    for field, tendency in here.items():
        scale = np.abs(tendency).max()
        difference = np.abs(rolled(tendency) - there[field]).max()
        assert difference <= 1e-13 * scale, (field, difference / scale)


@pytest.mark.parametrize("name", ["nonlinear", "multilayer"])
def test_a_uniform_flow_has_a_uniform_coriolis_term(name) -> None:
    _, _, layers, _, _ = MODELS[name]
    shape = (*layers, NY, NX)
    interiors = {
        "h": np.full(shape, 500.0),
        "u": np.full(shape, 0.1),
        "v": np.zeros(shape),
    }
    with float64():
        dv_dt = _tendency(name, Upwind(1), interiors, None)["v"]
    # f is 1e-4 in one model and 9.375e-5 in the other
    expected = -0.1 * (1e-4 if name == "nonlinear" else 9.375e-5)
    np.testing.assert_allclose(dv_dt, expected, rtol=1e-12)
