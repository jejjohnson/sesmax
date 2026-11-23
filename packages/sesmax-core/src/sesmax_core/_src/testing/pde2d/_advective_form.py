"""The advective form shared by the 2D nonlinear models."""

from __future__ import annotations

import jax.numpy as jnp
from gridax import (
    BC,
    CField,
    GridOps,
    Periodic,
    fill_ghosts,
)
from gridax.schemes import Scheme
from jaxtyping import Array, Float

from sesmax_core._src.fields import each


#: The boundary condition of a doubly periodic box.
PERIODIC = BC(x=Periodic(), y=Periodic())


def advective_tendency(
    u: Float[Array, "Ny Nx"] | CField,
    v: Float[Array, "Ny Nx"] | CField,
    ops: GridOps,
    *,
    scheme: Scheme,
) -> tuple[Float[Array, "Ny Nx"] | CField, Float[Array, "Ny Nx"] | CField]:
    """Advective tendencies of both velocity components on a periodic box.

    du/dt = -(u d/dx + v d/dy) u = -div(u w) + u div(w), with w = (u, v);
    the same for v

    gridax's advection operator computes the flux form ``-div(h u)``. The
    equations are in advective form and this flow is not divergence-free, so
    the ``u div(w)`` term is added back. The result is not conservative, so a
    sharp front's speed is only as good as the viscosity that resolves it.

    A periodic box needs ``wall="open"``, because a closed wall writes no
    tendency on the outermost cells, and the interpolated face velocities need
    their ghosts filled, because the interpolations write only the interior. A
    masked domain keeps closed walls, which gridax requires.

    Args:
        u: x-velocity at T-points, with its ghost cells filled.
        v: y-velocity at T-points, with its ghost cells filled.
        ops: gridax's operations on the grid and the mask.
        scheme: The scheme of the face values.

    Returns:
        The tendencies of ``u`` and ``v`` at T-points.
    """
    u_on_U = ops.interpolate(u, src="T", dst="U")
    v_on_V = ops.interpolate(v, src="T", dst="V")
    wall = "closed"
    if ops.mask is None:
        grid = ops.grid
        u_on_U = fill_ghosts(u_on_U, grid, PERIODIC, stagger="U")
        v_on_V = fill_ghosts(v_on_V, grid, PERIODIC, stagger="V")
        wall = "open"
    divergence = ops.divergence(u_on_U, v_on_V)
    if wall == "closed":
        # A closed wall leaves the ring [1] without a tendency; keep it so.
        divergence = each(
            lambda d: d * jnp.zeros_like(d).at[2:-2, 2:-2].set(1.0), divergence
        )
    flux_u = ops.advect(u, u_on_U, v_on_V, scheme=scheme, wall=wall)
    flux_v = ops.advect(v, u_on_U, v_on_V, scheme=scheme, wall=wall)

    def add_back(flux: Array, q: Array, div: Array) -> Array:
        return flux + q * div

    return each(add_back, flux_u, u, divergence), each(add_back, flux_v, v, divergence)
