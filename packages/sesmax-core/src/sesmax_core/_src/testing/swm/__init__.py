"""The linear shallow-water test problems and the 1D nonlinear one.

Moved from somax 9cb19b3 (``somax/_src/models/swm``), without the nonlinear
2D and multilayer models, which move with the ocean component.
"""

from __future__ import annotations

from sesmax_core._src.testing.swm.linear_1d import (
    LinearShallowWater1D,
    LinearSW1DDiagnostics,
    LinearSW1DParams,
    LinearSW1DPhysConsts,
    LinearSW1DState,
)
from sesmax_core._src.testing.swm.linear_2d import (
    LinearShallowWater2D,
    LinearSW2DDiagnostics,
    LinearSW2DParams,
    LinearSW2DPhysConsts,
    LinearSW2DState,
)
from sesmax_core._src.testing.swm.linear_swm_terms import (
    LinearSWM2DTermModel,
    LinearSWMCoriolis,
    LinearSWMDiffusion,
    LinearSWMDrag,
    LinearSWMGravityWave,
)
from sesmax_core._src.testing.swm.nonlinear_1d import (
    NonlinearShallowWater1D,
    NonlinearSW1DDiagnostics,
    NonlinearSW1DParams,
    NonlinearSW1DPhysConsts,
    NonlinearSW1DState,
)


__all__ = [
    "LinearSW1DDiagnostics",
    "LinearSW1DParams",
    "LinearSW1DPhysConsts",
    "LinearSW1DState",
    "LinearSW2DDiagnostics",
    "LinearSW2DParams",
    "LinearSW2DPhysConsts",
    "LinearSW2DState",
    "LinearSWM2DTermModel",
    "LinearSWMCoriolis",
    "LinearSWMDiffusion",
    "LinearSWMDrag",
    "LinearSWMGravityWave",
    "LinearShallowWater1D",
    "LinearShallowWater2D",
    "NonlinearSW1DDiagnostics",
    "NonlinearSW1DParams",
    "NonlinearSW1DPhysConsts",
    "NonlinearSW1DState",
    "NonlinearShallowWater1D",
]
