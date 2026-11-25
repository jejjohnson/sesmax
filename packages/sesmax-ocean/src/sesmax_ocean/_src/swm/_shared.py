"""What the shallow-water models share: ghosts, free slip, the default scheme."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from gridax import (
    BC,
    CartesianCGrid2D,
    Dirichlet,
    Mask2D,
    Neumann,
    Periodic,
    SphericalCGrid2D,
    avg_xfwd_yfwd_2d,
    difference,
    fill_ghosts,
    laplacian,
)
from gridax.schemes import Upwind
from jaxtyping import Array, Bool, Float

from sesmax_core.fields import each, placed, values


#: The default scheme of the face values in the mass equation.
UPWIND = Upwind(1)

#: The point of each field of a shallow-water state.
POINTS = {"h": "T", "u": "U", "v": "V"}
_BOX = BC(x=Periodic(), y=Periodic())
_FLAT, _WALL = Neumann(), Dirichlet(0.0)
#: The boundary conditions of each ``bc_type``: the doubly periodic box, and
#: closed walls with no normal flow and free slip.
BCS = {
    "periodic": {"h": _BOX, "u": _BOX, "v": _BOX},
    "wall": {
        "h": BC(x=_FLAT, y=_FLAT),
        "u": BC(x=_WALL, y=_FLAT),
        "v": BC(x=_FLAT, y=_WALL),
    },
}


#: The axes each ``bc_type`` closes.
WALLS = {"periodic": "", "wall": "xy"}

#: The faces of each ``bc_type`` a field wraps across.
WRAPS = {"periodic": _BOX, "wall": None}


def wrap(field: Any, grid: Any, bc: BC | None, at: str) -> Any:
    """``field`` with its ghosts filled on the periodic faces of the grid.

    For a product of a tendency, whose ghost ring an operator leaves at zero and
    the next one reads. Behind a wall no cell of the tendency reads it.

    Args:
        field: An array or a field at ``at``.
        grid: Its grid.
        bc: The periodic faces, or ``None`` for none.
        at: ``"T"``, ``"U"``, ``"V"`` or ``"X"``.

    Returns:
        The field.
    """
    return field if bc is None else fill_ghosts(field, grid, bc, stagger=at)


def velocity_laplacian(
    velocity: Any,
    grid: CartesianCGrid2D | SphericalCGrid2D,
    mask: Mask2D | None,
    at: str,
    *,
    thickness_bc: BC,
    wraps: BC | None,
) -> Any:
    """The Laplacian of a velocity component, with free slip along the coasts.

    Along the component the gradients are at T, and the velocity on a coast is
    zero. Across it they are at X, and a coast has no shear: the gradient is zero
    on the corners without four wet cells.

    Args:
        velocity: An array or a field at ``at``, with its ghosts filled.
        grid: Its grid.
        mask: The wet cells, or ``None`` for all.
        at: ``"U"`` or ``"V"``.
        thickness_bc: The boundary condition of a field at T that is even about a
            wall, as a thickness and the divergence of a velocity are.
        wraps: The periodic faces, or ``None`` for none.

    Returns:
        The Laplacian at the same points, zero on the dry ones.
    """
    if mask is None:
        return laplacian(velocity, grid, at=at)
    along, across = ("x", "y") if at == "U" else ("y", "x")
    # Land holds no flow. The ghost ring is the boundary condition's.
    wet = jnp.asarray(mask.wet(at=at))
    wet = wet.at[[0, -1], :].set(True).at[:, [0, -1]].set(True)
    array = jnp.where(wet, values(velocity), 0.0)

    def flux(axis: str, dst: str) -> Array:
        gradient = difference(array, grid, axis=axis, src=at, dst=dst, mask=mask)
        return grid.area(at=dst) * gradient

    normal = fill_ghosts(flux(along, "T"), grid, thickness_bc, stagger="T")
    shear = wrap(flux(across, "X"), grid, wraps, "X")
    # The areas carry the metric of the sphere: 1/cos d(cos d u).
    total = difference(normal, grid, axis=along, src="T", dst=at, mask=mask)
    total = total + difference(shear, grid, axis=across, src="X", dst=at, mask=mask)
    return placed(total / grid.area(at=at), velocity, at)


def hold_walls(tendency: Any, walls: str, at: str) -> Any:
    """The tendency of a velocity, zero on the wall faces normal to it.

    The integrator applies the boundary conditions to the input of a tendency,
    not to the state it returns, so a tendency on a wall would add up there.

    Args:
        tendency: An array or a field at ``at``.
        walls: The axes the box is closed along.
        at: ``"U"`` or ``"V"``.

    Returns:
        The tendency.
    """
    axis = "x" if at == "U" else "y"
    if axis not in walls:
        return tendency
    faces = jnp.array([0, -2, -1])
    where = (..., faces) if axis == "x" else (..., faces, slice(None))
    return each(lambda array: array.at[where].set(0.0), tendency)


def fill_box(state: Any, grid: CartesianCGrid2D, bc_type: str) -> Any:
    """A state with fields ``h``, ``u`` and ``v`` with its ghost ring filled.

    Args:
        state: A shallow-water state of arrays or fields.
        grid: The grid of the state.
        bc_type: ``"periodic"`` or ``"wall"``.

    Returns:
        A state of the same type.
    """
    filled = fill_ghosts(
        {name: getattr(state, name) for name in POINTS},
        grid,
        BCS[bc_type],
        stagger=POINTS,
    )
    return type(state)(**filled)


def corner_mean(field: Float[Array, "*batch Ny Nx"]) -> Float[Array, "*batch Ny Nx"]:
    """The mean of the four cells around each corner, the walls included.

    gridax's operators leave the ghost ring at zero, and the corners on the
    southern and western walls are in it. The last row and column, which have no
    cell beyond them, are zero.
    """
    batch = [(0, 0)] * (field.ndim - 2)
    interior = avg_xfwd_yfwd_2d(jnp.pad(field, [*batch, (1, 0), (1, 0)]))
    return jnp.pad(interior, [*batch, (0, 1), (0, 1)])


class FreeSlip(eqx.Module):
    """The potential vorticity on the walls and coasts that water slips along.

    The relative vorticity vanishes on a free-slip wall, so the potential
    vorticity there is ``f / h``, with the thickness averaged over the wet cells
    around the corner.

    Args:
        corners: The corners on a wall or a coast that have a wet cell.
        wet: The wet cells, with their ghosts filled as the thickness's are.
        numerator: ``f`` at the corners, times the wet share of the four cells.
    """

    corners: Bool[Array, "Ny Nx"]
    wet: Bool[Array, "Ny Nx"]
    numerator: Float[Array, "Ny Nx"]

    @classmethod
    def build(
        cls,
        grid: CartesianCGrid2D | SphericalCGrid2D,
        mask: Mask2D | None,
        bc: BC,
        f: Float[Array, "Ny Nx"],
        *,
        walls: str,
    ) -> FreeSlip:
        """Find the corners, once per model.

        Args:
            grid: The grid.
            mask: The wet cells, or ``None`` for all.
            bc: The boundary condition of the thickness.
            f: The Coriolis parameter at T-points, with its ghost ring.
            walls: The axes the box is closed along, ``"xy"``, ``"y"`` or ``""``.

        Returns:
            The rule for this domain.
        """
        wet = jnp.ones_like(f) if mask is None else mask.wet(at="T").astype(f.dtype)
        wet = fill_ghosts(wet, grid, bc, stagger="T")
        share = corner_mean(wet)
        # What gridax computes, and what is inside the water: the interior corners
        # off the walls with four wet cells.
        computed = jnp.zeros(share.shape, dtype=bool).at[1:-1, 1:-1].set(True)
        if "y" in walls:
            computed = computed.at[-2, :].set(False)
        if "x" in walls:
            computed = computed.at[:, -2].set(False)
        if mask is not None:
            computed = computed & mask.wet(at="X", kind="strict")
        return cls(
            corners=(share > 0.0) & ~computed,
            wet=wet > 0.5,
            numerator=corner_mean(f) * share,
        )

    def __call__(
        self, q: Float[Array, "*batch Ny Nx"], h: Float[Array, "*batch Ny Nx"]
    ) -> Float[Array, "*batch Ny Nx"]:
        """``q`` with ``f / h`` on the walls and coasts.

        Args:
            q: The potential vorticity at X-points.
            h: The thickness at T-points, with its ghosts filled.

        Returns:
            The potential vorticity.
        """
        thickness = corner_mean(jnp.where(self.wet, h, 0.0))
        slip = self.numerator / jnp.where(self.corners, thickness, 1.0)
        return jnp.where(self.corners, slip, q)
