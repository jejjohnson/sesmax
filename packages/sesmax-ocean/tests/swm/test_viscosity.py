"""The viscous term of a velocity between walls, with no normal flow and free slip.

`sin(πx/Lx)` is an eigenvector of the discrete Laplacian of the velocity normal to
the walls, so the term is the same multiple of the velocity in every cell, the
ones beside a wall too.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from somax_reference import float64

from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


NY, NX = 12, 16
NU = 1e3
MODELS = {
    "nonlinear": (NonlinearShallowWater2D, NonlinearSW2DState, ()),
    "multilayer": (MultilayerShallowWater2D, MultilayerSW2DState, (3,)),
}


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("name", list(MODELS))
def test_the_viscous_term_beside_a_wall(name: str, axis: str) -> None:
    model_class, state_class, layers = MODELS[name]
    with float64():
        viscous, inviscid = (
            model_class.create(
                nx=NX, ny=NY, Lx=1e6, Ly=8e5, bc="wall", lateral_viscosity=nu
            )
            for nu in (NU, 0.0)
        )
        y, x = np.meshgrid(np.arange(NY + 2), np.arange(NX + 2), indexing="ij")
        n, i = (NX, x) if axis == "x" else (NY, y)
        normal = np.broadcast_to(np.sin(np.pi * i / n), (*layers, NY + 2, NX + 2))
        zeros = jnp.zeros(normal.shape)
        field = "u" if axis == "x" else "v"
        state = state_class(
            **{"h": zeros + 500.0, "u": zeros, "v": zeros, field: jnp.asarray(normal)}
        )
        state = viscous.apply_boundary_conditions(state)
        difference = np.asarray(
            getattr(viscous.vector_field(0.0, state), field)
            - getattr(inviscid.vector_field(0.0, state), field)
        )
        spacing = float(viscous.grid.spacing(axis))
    eigenvalue = -4.0 / spacing**2 * np.sin(0.5 * np.pi / n) ** 2
    inner = (..., slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(
        difference[inner],
        NU * eigenvalue * normal[inner],
        atol=1e-10 * abs(NU * eigenvalue),
        rtol=0.0,
    )
