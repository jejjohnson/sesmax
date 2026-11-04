"""The advective form shared by the 2D nonlinear models."""

from __future__ import annotations

import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    Difference2D,
    Interpolation2D,
    enforce_periodic,
)
from jaxtyping import Array, Float


def advective_tendency(
    u: Float[Array, "Ny Nx"],
    v: Float[Array, "Ny Nx"],
    advection: FVXAdvection2D,
    interp: Interpolation2D,
    diff: Difference2D,
    *,
    method: str,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Advective tendencies of both velocity components on a periodic box.

    ∂u/∂t = −(u ∂/∂x + v ∂/∂y) u = −∇·(u 𝐮) + u ∇·𝐮, and the same for v

    finitevolX's advection operator computes the flux form ``-div(h u)``. The
    equations are in advective form and this flow is not divergence-free, so
    the ``u ∇·𝐮`` term is added back. The result is not conservative, so a
    sharp front's speed is only as good as the viscosity that resolves it.

    A periodic box needs ``wall="open"``, because a closed wall writes no
    tendency on the outermost cells, and the interpolated face velocities need
    their ghosts filled, because the interpolations write only the interior. A
    masked domain keeps closed walls, which finitevolX requires.

    Args:
        u: x-velocity at T-points, with its ghost cells filled.
        v: y-velocity at T-points, with its ghost cells filled.
        advection: finitevolX advection operator.
        interp: finitevolX interpolation operator.
        diff: finitevolX difference operator.
        method: Reconstruction method.

    Returns:
        The tendencies of ``u`` and ``v`` at T-points.
    """
    u_on_U = interp.T_to_U(u)
    v_on_V = interp.T_to_V(v)
    wall = "closed"
    if advection.mask is None:
        u_on_U = enforce_periodic(u_on_U)
        v_on_V = enforce_periodic(v_on_V)
        wall = "open"
    divergence = diff.divergence(u_on_U, v_on_V)
    if wall == "closed":
        # A closed wall leaves the ring [1] without a tendency; keep it so.
        ring_free = jnp.zeros_like(divergence).at[2:-2, 2:-2].set(1.0)
        divergence = divergence * ring_free
    flux_u = advection(u, u_on_U, v_on_V, method=method, wall=wall)
    flux_v = advection(v, u_on_U, v_on_V, method=method, wall=wall)
    return flux_u + u * divergence, flux_v + v * divergence
