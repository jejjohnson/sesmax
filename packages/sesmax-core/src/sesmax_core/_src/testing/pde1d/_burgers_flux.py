"""The Burgers flux shared by the 1D nonlinear models."""

from __future__ import annotations

from gridax import (
    Advection1D,
    CField,
    Interpolation1D,
)
from gridax.schemes import Scheme
from jaxtyping import Array, Float

from sesmax_core._src.fields import each
from sesmax_core._src.testing.pde1d._ghosts import fill_ends


def burgers_tendency(
    u: Float[Array, " Nx"] | CField,
    advection: Advection1D,
    interp: Interpolation1D,
    *,
    periodic: bool,
    scheme: Scheme,
) -> Float[Array, " Nx"] | CField:
    """Advective tendency of Burgers' equation in flux form.

    du/dt = -d(u^2 / 2)/dx

    gridax's advection operator computes ``-d(h v)/dx`` with ``v`` at the
    faces, so the flux ``u^2 / 2`` is half of ``h = u`` carried by ``v = u``
    interpolated to the faces.

    A periodic domain needs ``wall="open"``, because a closed wall writes no
    tendency on the outermost cells, and the interpolated face velocity needs
    its ghosts filled, because the interpolation writes only the interior. A
    masked domain keeps closed walls, which gridax requires.

    Args:
        u: Velocity at T-points, with its ghost cells filled.
        advection: gridax advection operator.
        interp: gridax interpolation operator.
        periodic: Whether the domain is periodic.
        scheme: The scheme of the face values.

    Returns:
        The tendency at T-points.
    """
    face_velocity = interp.T_to_U(u)
    wall = "closed"
    if periodic and advection.mask is None:
        face_velocity = fill_ends(face_velocity, advection.grid, periodic=True, at="U")
        wall = "open"
    flux_form = advection(u, face_velocity, scheme=scheme, wall=wall)
    return each(lambda x: 0.5 * x, flux_form)
