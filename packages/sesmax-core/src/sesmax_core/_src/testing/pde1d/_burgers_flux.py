"""The Burgers flux shared by the 1D nonlinear models."""

from __future__ import annotations

from gridax import (
    BC,
    CField,
    GridOps,
)
from gridax.schemes import Scheme
from jaxtyping import Array, Float

from sesmax_core._src.fields import each


def burgers_tendency(
    u: Float[Array, " Nx"] | CField,
    ops: GridOps,
    *,
    periodic: bool,
    scheme: Scheme,
) -> Float[Array, " Nx"] | CField:
    """Advective tendency of Burgers' equation in flux form.

    du/dt = -d(u^2 / 2)/dx

    gridax's advection operator computes ``-d(h v)/dx`` with ``v`` at the
    faces, so the flux ``u^2 / 2`` is half of ``h = u`` carried by ``v = u``
    interpolated to the faces.

    A periodic domain is said so with ``bc=``: a closed wall would write no
    tendency on the outermost cells. A masked domain keeps closed walls.

    Args:
        u: Velocity at T-points, with its ghost cells filled.
        ops: gridax's operations on the grid and the mask.
        periodic: Whether the domain is periodic.
        scheme: The scheme of the face values.

    Returns:
        The tendency at T-points.
    """
    face_velocity = ops.interpolate(u, src="T", dst="U")
    bc = BC.periodic("x") if periodic and ops.mask is None else None
    flux_form = ops.advect(u, face_velocity, None, scheme=scheme, bc=bc)
    return each(lambda x: 0.5 * x, flux_form)
