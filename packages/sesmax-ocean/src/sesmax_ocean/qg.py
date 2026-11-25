"""Public surface for sesmax-ocean's quasi-geostrophic models.

:class:`BarotropicQG` and :class:`BaroclinicQG` step a potential vorticity on
T-points and invert it for the streamfunction, ∇²ψ = q for one layer and mode by
mode for several. :class:`ReparameterizedQG` steps the thickness and the velocity
of :class:`sesmax_ocean.swm.MultilayerShallowWater2D` and projects them onto
geostrophic balance.
"""

from __future__ import annotations

from sesmax_ocean._src.qg.baroclinic import (
    BaroclinicQG,
    BaroclinicQGDiagnostics,
    BaroclinicQGParams,
    BaroclinicQGPhysConsts,
    BaroclinicQGState,
)
from sesmax_ocean._src.qg.barotropic import (
    BarotropicQG,
    BarotropicQGDiagnostics,
    BarotropicQGParams,
    BarotropicQGPhysConsts,
    BarotropicQGState,
)
from sesmax_ocean._src.qg.reparameterized import (
    ReparameterizedQG,
    ReparamQGDiagnostics,
)


__all__ = [
    "BaroclinicQG",
    "BaroclinicQGDiagnostics",
    "BaroclinicQGParams",
    "BaroclinicQGPhysConsts",
    "BaroclinicQGState",
    "BarotropicQG",
    "BarotropicQGDiagnostics",
    "BarotropicQGParams",
    "BarotropicQGPhysConsts",
    "BarotropicQGState",
    "ReparamQGDiagnostics",
    "ReparameterizedQG",
]
