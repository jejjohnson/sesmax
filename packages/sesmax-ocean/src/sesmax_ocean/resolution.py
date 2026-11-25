"""Public surface for sesmax-ocean's resolution checks.

Does the grid span the features a model must contain: the Munk and Stommel
boundary layers, ``(nu/beta)^(1/3)`` and ``kappa/beta`` wide, the first baroclinic
deformation radius, and on a sphere the equatorial one. Each check raises
:class:`AssertionFailedError` when it does not, and warns when it barely does.
"""

from __future__ import annotations

from sesmax_ocean._src.resolution import (
    AssertionFailedError,
    check_deformation_radius,
    check_equatorial_deformation_radius,
    check_munk_width,
    check_stommel_width,
)


__all__ = [
    "AssertionFailedError",
    "check_deformation_radius",
    "check_equatorial_deformation_radius",
    "check_munk_width",
    "check_stommel_width",
]
