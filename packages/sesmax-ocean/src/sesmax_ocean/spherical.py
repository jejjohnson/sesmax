"""Public surface for sesmax-ocean's models on the sphere.

Both run on a latitude band that wraps in longitude and stops short of the
poles: :class:`SphericalSWM` steps a thickness and a velocity, and
:class:`SphericalQG` a vorticity it inverts for the streamfunction, ∇²ψ = q.
"""

from __future__ import annotations

from sesmax_ocean._src.spherical.qg import (
    SphericalQG,
    SphericalQGDiagnostics,
    SphericalQGParams,
    SphericalQGPhysConsts,
    SphericalQGState,
)
from sesmax_ocean._src.spherical.swm import (
    SphericalSWM,
    SphericalSWMDiagnostics,
    SphericalSWMParams,
    SphericalSWMPhysConsts,
    SphericalSWMState,
)


__all__ = [
    "SphericalQG",
    "SphericalQGDiagnostics",
    "SphericalQGParams",
    "SphericalQGPhysConsts",
    "SphericalQGState",
    "SphericalSWM",
    "SphericalSWMDiagnostics",
    "SphericalSWMParams",
    "SphericalSWMPhysConsts",
    "SphericalSWMState",
]
