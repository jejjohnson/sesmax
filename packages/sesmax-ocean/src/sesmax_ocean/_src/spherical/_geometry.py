"""The geometry the spherical models share: what they accept, and the latitudes.

Both models take arbitrary ordered longitude and latitude bounds, but
their boundary conditions only describe one shape of domain: a band
that wraps in longitude and stops short of the poles. These say so at
construction rather than leaving the caller to discover it from wrong
answers.
"""

from __future__ import annotations

import jax.numpy as jnp
from gridax import SphericalCGrid2D
from jaxtyping import Array, Float


def latitudes(grid: SphericalCGrid2D) -> Float[Array, "Ny Nx"]:
    """The latitude of every T-point, in radians."""
    lat = jnp.asarray(grid.coords("y"))
    return jnp.broadcast_to(lat[:, None], (grid.Ny, grid.Nx))


def require_global_longitude(context: str, lon_range: tuple[float, float]) -> None:
    """Reject a longitude span that is not the whole globe.

    ``apply_boundary_conditions`` connects the western and eastern
    ghost columns unconditionally, which is right for a lat-lon grid
    that closes on itself and wrong for a regional sector — there it
    would carry mass and momentum between two unrelated boundaries.
    Both factories accept arbitrary ordered bounds, so the restriction
    has to be stated rather than assumed.

    Args:
        context: Caller name, for the error message.
        lon_range: ``(lon_min, lon_max)`` in degrees.

    Raises:
        ValueError: If the span is not 360 degrees.
    """
    span = float(lon_range[1]) - float(lon_range[0])
    if abs(span - 360.0) > 1.0e-9:
        raise ValueError(
            f"{context}: lon_range={lon_range!r} spans {span:g} degrees, but "
            f"the zonal boundary condition is periodic and only a full 360 "
            f"degree span closes on itself. A regional sector needs "
            f"non-periodic zonal boundaries, which this model does not "
            f"implement yet."
        )


def require_open_poles(context: str, lat_range: tuple[float, float]) -> None:
    """Reject a latitude range that reaches either pole.

    The polar rows carry a Dirichlet condition — ``v = 0`` for the
    shallow-water model, ``psi = 0`` for QG — which represents a solid
    wall. That is a fair description of a latitude band, and a wrong
    one at a true pole, where the sphere is regular and the flow simply
    continues: a zonal solid-body flow has a streamfunction
    proportional to ``sin(phi)``, taking different values at the two
    poles, and cannot be represented at all under ``psi = 0`` on both.

    Args:
        context: Caller name, for the error message.
        lat_range: ``(lat_min, lat_max)`` in degrees.

    Raises:
        ValueError: If either bound is at or beyond a pole.
    """
    lat_min, lat_max = (float(value) for value in lat_range)
    if lat_min <= -90.0 or lat_max >= 90.0:
        raise ValueError(
            f"{context}: lat_range={lat_range!r} reaches a pole. The polar "
            f"rows carry a solid-wall condition, which is right for a "
            f"latitude band and wrong at a true pole, where the sphere is "
            f"regular. Pole coupling with a single gauge constraint is not "
            f"implemented; use a band such as (-80, 80)."
        )
