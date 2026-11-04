"""The ghost-cell fill shared by the 1D test problems."""

from __future__ import annotations

from jaxtyping import Array, Float


def fill_ghosts(u: Float[Array, " Nx"], *, periodic: bool) -> Float[Array, " Nx"]:
    """Fill the two ghost cells of a 1D field at T-points.

    Args:
        u: The field, with one ghost cell at each end.
        periodic: Wrap the domain if True. Otherwise the boundary value is zero,
            which reflects the first interior value with its sign flipped.

    Returns:
        The field with its ghost cells filled.
    """
    if periodic:
        return u.at[0].set(u[-2]).at[-1].set(u[1])
    return u.at[0].set(-u[1]).at[-1].set(-u[-2])
