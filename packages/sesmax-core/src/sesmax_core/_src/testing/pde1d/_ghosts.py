"""The ghost-cell fill shared by the 1D test problems."""

from __future__ import annotations

from gridax import (
    BC,
    CartesianCGrid1D,
    CField,
    Dirichlet,
    Periodic,
    fill_ghosts,
)
from jaxtyping import Array, Float


def fill_ends(
    u: Float[Array, " Nx"] | CField,
    grid: CartesianCGrid1D,
    *,
    periodic: bool,
    at: str = "T",
) -> Float[Array, " Nx"] | CField:
    """Fill the two ghost cells of a 1D field.

    Args:
        u: The field, with one ghost cell at each end.
        grid: The grid.
        periodic: Wrap the domain if True. Otherwise the boundary value is zero,
            which reflects the first interior value with its sign flipped.
        at: The point of the field.

    Returns:
        The field with its ghost cells filled.
    """
    bc = BC(x=Periodic() if periodic else Dirichlet(0.0))
    return fill_ghosts(u, grid, bc, stagger=at)
