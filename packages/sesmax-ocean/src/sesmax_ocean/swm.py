"""Public surface for sesmax-ocean's shallow-water models.

Both step a layer thickness on T-points and a velocity on U- and V-points, in
vector-invariant form: :class:`NonlinearShallowWater2D` one layer, and
:class:`MultilayerShallowWater2D` several, coupled by their pressure.
"""

from __future__ import annotations

from sesmax_ocean._src.swm.multilayer import (
    MultilayerShallowWater2D,
    MultilayerSW2DDiagnostics,
    MultilayerSW2DParams,
    MultilayerSW2DPhysConsts,
    MultilayerSW2DState,
)
from sesmax_ocean._src.swm.nonlinear_2d import (
    NonlinearShallowWater2D,
    NonlinearSW2DDiagnostics,
    NonlinearSW2DParams,
    NonlinearSW2DPhysConsts,
    NonlinearSW2DState,
)


__all__ = [
    "MultilayerSW2DDiagnostics",
    "MultilayerSW2DParams",
    "MultilayerSW2DPhysConsts",
    "MultilayerSW2DState",
    "MultilayerShallowWater2D",
    "NonlinearSW2DDiagnostics",
    "NonlinearSW2DParams",
    "NonlinearSW2DPhysConsts",
    "NonlinearSW2DState",
    "NonlinearShallowWater2D",
]
