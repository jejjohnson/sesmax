"""Public surface for sesmax-ocean's quasi-geostrophic models.

Both step a potential vorticity on T-points and invert it for the
streamfunction: :class:`BarotropicQG` one layer, ∇²ψ = q, and
:class:`BaroclinicQG` several, mode by mode.
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
]
