"""What the shallow-water models share: the ghosts and the default scheme."""

from __future__ import annotations

from typing import Any

from gridax import (
    BC,
    CartesianCGrid2D,
    Dirichlet,
    Neumann,
    Periodic,
    fill_ghosts,
)
from gridax.schemes import Upwind


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
