"""The shallow-water models conserve mass to rounding.

The mass equation is in flux form, so the area integral of its tendency is the
flux through the outer faces: none in a periodic box or behind walls.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gridax import Mask2D, area_sum
from gridax.schemes import WENO, Upwind
from somax_reference import float64

from sesmax_ocean.spherical import SphericalSWM, SphericalSWMState
from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


SIZE = {"nx": 32, "ny": 24}
SHAPE = (26, 34)
MODELS = {
    "nonlinear": (NonlinearShallowWater2D, NonlinearSW2DState, ()),
    "multilayer": (MultilayerShallowWater2D, MultilayerSW2DState, (3,)),
    "spherical": (SphericalSWM, SphericalSWMState, ()),
}


def _mask(periodic: bool) -> Mask2D:
    """An island; the ghost ring is wet where the box is periodic."""
    wet = np.full(SHAPE, periodic)
    wet[1:-1, 1:-1] = True
    wet[8:13, 14:20] = False
    return Mask2D.from_center(wet)


@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "island"])
@pytest.mark.parametrize("scheme", [Upwind(1), WENO(5)], ids=["upwind1", "weno5"])
@pytest.mark.parametrize(
    ("name", "bc"),
    [
        ("nonlinear", "periodic"),
        ("nonlinear", "wall"),
        ("multilayer", "periodic"),
        ("multilayer", "wall"),
        ("spherical", None),
    ],
)
def test_the_net_mass_tendency_is_zero(name, bc, scheme, masked) -> None:
    model_class, state_class, layers = MODELS[name]
    kwargs = {} if bc is None else {"bc": bc}
    rng = np.random.default_rng(0)
    with float64():
        model = model_class.create(
            **SIZE,
            **kwargs,
            scheme=scheme,
            mask=_mask(bc != "wall") if masked else None,
        )
        h = 500.0 + rng.normal(size=(*layers, *SHAPE))
        u, v = 0.1 * rng.normal(size=(2, *layers, *SHAPE))
        state = state_class(h=jnp.asarray(h), u=jnp.asarray(u), v=jnp.asarray(v))
        state = model.apply_boundary_conditions(state)
        dh_dt = model.vector_field(0.0, state).h
        net = np.asarray(area_sum(dh_dt, model.grid))
        total = np.asarray(area_sum(jnp.abs(dh_dt), model.grid))
    assert (np.abs(net) < 1e-14 * total).all()


def test_mass_is_conserved_through_a_run() -> None:
    """A bump that crosses the edge of a periodic box keeps its mass."""
    with float64():
        model = NonlinearShallowWater2D.create(nx=32, ny=8, Lx=1e6, Ly=2.5e5, f0=0.0)
        x = jnp.arange(model.grid.Nx) * model.grid.spacing("x")
        bump = 100.0 + 2.0 * jnp.exp(-(((x - 5e5) / 1e5) ** 2))
        h = jnp.broadcast_to(bump, (model.grid.Ny, model.grid.Nx))
        zeros = jnp.zeros_like(h)
        state = model.apply_boundary_conditions(
            NonlinearSW2DState(h=h, u=zeros, v=zeros)
        )
        # 31 m/s for 16,000 s: half the box, so the two waves meet at the edge
        end = model.integrate(state, t0=0.0, t1=1.6e4, dt=200.0).ys
        mass = [float(area_sum(f, model.grid)) for f in (state.h, end.h[-1])]
        moved = float(jnp.abs(end.h[-1][4, 1] - 100.0))
    assert abs(mass[1] - mass[0]) < 1e-12 * mass[0]
    assert moved > 0.1
