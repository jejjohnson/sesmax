"""The states a run between walls returns have no flow through them."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gridax import CField

from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


MODELS = {
    "nonlinear": (NonlinearShallowWater2D, NonlinearSW2DState, ()),
    "multilayer": (MultilayerShallowWater2D, MultilayerSW2DState, (3,)),
}


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize("name", list(MODELS))
def test_no_flow_through_the_walls(name: str, located: bool) -> None:
    model_class, state_class, layers = MODELS[name]
    model = model_class.create(
        nx=16, ny=12, Lx=1e6, Ly=8e5, bc="wall", wind_amplitude=1e-3
    )
    shape = (*layers, 14, 18)
    depth = 100.0 if name == "nonlinear" else model.strat.H[:, None, None]
    fields = {
        "h": depth + jnp.zeros(shape),
        "u": jnp.zeros(shape),
        "v": jnp.zeros(shape),
    }
    if located:
        dims = ("layer", "y", "x")[-len(shape) :]
        fields = {
            k: CField(x, dims, loc=at)
            for (k, x), at in zip(fields.items(), "TUV", strict=True)
        }
    end = model.integrate(state_class(**fields), t0=0.0, t1=3600.0, dt=60.0).ys
    u, v = (np.asarray(x.data if located else x)[-1] for x in (end.u, end.v))
    assert np.abs(u).max() > 0.0
    for face in (0, -2):
        np.testing.assert_array_equal(u[..., face], 0.0)
        np.testing.assert_array_equal(v[..., face, :], 0.0)
    # The face beyond the wall is a ghost, the odd image of the one inside.
    np.testing.assert_array_equal(u[..., -1], -u[..., -3])
    np.testing.assert_array_equal(v[..., -1, :], -v[..., -3, :])
