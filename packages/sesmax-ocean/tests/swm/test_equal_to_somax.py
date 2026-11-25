"""The shallow-water models give somax's numbers, on arrays and located fields.

What somax computed is in ``tests/reference``: from one random thickness and
velocity, the tendency, the state after 20 Heun steps and the diagnostics, in a
periodic box, between walls and with an island. `ReparameterizedQG` is here too,
because its state and its tendency are the multilayer model's.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from equality import (
    CASES,
    SIZE,
    assert_reproduces_somax,
    float64,
    initial_state,
    island_mask,
)
from gridax import CField, StaggerMismatchError
from gridax.schemes import WENO, Upwind
from somax_reference import SHALLOW_WATER

from sesmax_ocean.qg import ReparameterizedQG
from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


MODELS = {
    "nonlinear_sw": (NonlinearShallowWater2D, NonlinearSW2DState, ("y", "x")),
    "multilayer_sw": (
        MultilayerShallowWater2D,
        MultilayerSW2DState,
        ("layer", "y", "x"),
    ),
    "reparameterized_qg": (
        ReparameterizedQG,
        MultilayerSW2DState,
        ("layer", "y", "x"),
    ),
}
SCHEMES = {"upwind1": Upwind(1), "weno5": WENO(5)}
POINTS = {"h": "T", "u": "U", "v": "V"}


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize(
    ("family", "case"),
    [(family, case) for family, cases in SHALLOW_WATER.items() for case in cases],
)
def test_equal_to_somax(family: str, case: str, located: bool) -> None:
    model_class, state_class, dims = MODELS[family]
    bc, scheme, has_island = SHALLOW_WATER[family][case]
    with float64():
        model = model_class.create(
            **SIZE,
            **CASES[family][0],
            bc=bc,
            scheme=SCHEMES[scheme],
            mask=island_mask() if has_island else None,
        )
        given, state = (
            initial_state(
                state_class,
                family,
                case,
                POINTS,
                dims if located else None,
                kind=kind,
            )
            for kind in ("given", "init")
        )
        assert_reproduces_somax(model, given, state, family, case)


@pytest.mark.parametrize("family", list(MODELS))
def test_a_velocity_at_the_wrong_point_is_rejected(family: str) -> None:
    model_class, state_class, dims = MODELS[family]
    model = model_class.create(**SIZE)
    shape = (*((3,) if len(dims) == 3 else ()), model.grid.Ny, model.grid.Nx)
    state = state_class(
        h=CField(jnp.ones(shape), dims, loc="T"),
        u=CField(jnp.zeros(shape), dims, loc="V"),
        v=CField(jnp.zeros(shape), dims, loc="V"),
    )
    with pytest.raises(StaggerMismatchError):
        model.vector_field(0.0, state)
