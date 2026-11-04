"""The Burgers flux shared by the 1D nonlinear models."""

from __future__ import annotations

from finitevolx import Advection1D, Interpolation1D
from jaxtyping import Array, Float


def burgers_tendency(
    u: Float[Array, " Nx"],
    advection: Advection1D,
    interp: Interpolation1D,
    *,
    periodic: bool,
    method: str,
) -> Float[Array, " Nx"]:
    """Advective tendency of Burgers' equation in flux form.

    du/dt = -d(u^2 / 2)/dx

    finitevolX's advection operator computes ``-d(h v)/dx`` with ``v`` at the
    faces, so the flux ``u^2 / 2`` is half of ``h = u`` carried by ``v = u``
    interpolated to the faces.

    A periodic domain needs ``wall="open"``, because a closed wall writes no
    tendency on the outermost cells, and the interpolated face velocity needs
    its ghosts filled, because the interpolation writes only the interior. A
    masked domain keeps closed walls, which finitevolX requires.

    Args:
        u: Velocity at T-points, with its ghost cells filled.
        advection: finitevolX advection operator.
        interp: finitevolX interpolation operator.
        periodic: Whether the domain is periodic.
        method: Reconstruction method.

    Returns:
        The tendency at T-points.
    """
    face_velocity = interp.T_to_U(u)
    wall = "closed"
    if periodic and advection.mask is None:
        face_velocity = face_velocity.at[0].set(face_velocity[-2])
        face_velocity = face_velocity.at[-1].set(face_velocity[1])
        wall = "open"
    return 0.5 * advection(u, face_velocity, method=method, wall=wall)
