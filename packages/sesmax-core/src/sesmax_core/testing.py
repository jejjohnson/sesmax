"""Public surface for sesmax-core's test problems.

Standard test problems, not Earth-system components: the chaotic Lorenz
systems used as data-assimilation testbeds, and the PDE test problems of
the "12 steps to Navier-Stokes" on finitevolX stencils (1D and 2D
convection, diffusion and Burgers, the Laplace, Poisson and Helmholtz
solvers, 2D incompressible Navier-Stokes). The core's own tests run on
them. The linear shallow-water models (1D and 2D, monolithic and as terms)
and the 1D nonlinear one, with the gravity-wave, inertial-oscillation and
geostrophic-adjustment cases, are the linear geophysical test problems.

- :class:`Lorenz63`: three variables, ``dx/dt = sigma (y - x)``,
  ``dy/dt = x (rho - z) - y``, ``dz/dt = x y - beta z``.
- :class:`Lorenz96`: ``Dx`` variables on a ring,
  ``dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F``.
- :class:`Lorenz96t`: the two-tier version, slow ``X`` coupled to fast ``Y``.
- :class:`LinearConvection1D`, :class:`NonlinearConvection1D`,
  :class:`Diffusion1D`, :class:`Burgers1D`: ``u_t + c u_x = 0``,
  ``u_t + u u_x = 0``, ``u_t = nu u_xx`` and ``u_t + u u_x = nu u_xx`` on a
  periodic interval, and their 2D counterparts.
- :class:`Burgers2DTermModel`: 2D Burgers as advection plus diffusion
  terms (diffusion implicit with ``imex=True``), and :class:`Burgers2DOp`,
  its serialisable pipekit operator.
- :class:`IncompressibleNS2D`: vorticity form,
  ``omega_t + J(psi, omega) = nu laplacian(omega)``, ``laplacian(psi) = omega``.
"""

from __future__ import annotations

from sesmax_core._src.testing.gfd_testcases import (
    geostrophic_adjustment_2d,
    gravity_wave_1d,
    inertial_oscillation_1d,
)
from sesmax_core._src.testing.lorenz63 import (
    L63Diagnostics,
    L63Params,
    L63State,
    Lorenz63,
)
from sesmax_core._src.testing.lorenz96 import (
    L96Diagnostics,
    L96Params,
    L96State,
    Lorenz96,
)
from sesmax_core._src.testing.lorenz96t import L96TParams, L96TState, Lorenz96t
from sesmax_core._src.testing.operators import Burgers2DOp
from sesmax_core._src.testing.pde1d import (
    Burgers1D,
    Burgers1DDiagnostics,
    Burgers1DParams,
    Burgers1DState,
    Diffusion1D,
    Diffusion1DDiagnostics,
    Diffusion1DParams,
    Diffusion1DState,
    LinearConvection1D,
    LinearConvection1DDiagnostics,
    LinearConvection1DParams,
    LinearConvection1DState,
    NonlinearConvection1D,
    NonlinearConvection1DDiagnostics,
    NonlinearConvection1DState,
)
from sesmax_core._src.testing.pde2d import (
    Burgers2D,
    Burgers2DAdvection,
    Burgers2DDiagnostics,
    Burgers2DDiffusion,
    Burgers2DParams,
    Burgers2DState,
    Burgers2DTermModel,
    Diffusion2D,
    Diffusion2DDiagnostics,
    Diffusion2DParams,
    Diffusion2DState,
    HelmholtzSolver2D,
    IncompressibleNS2D,
    LaplaceSolver2D,
    LinearConvection2D,
    LinearConvection2DDiagnostics,
    LinearConvection2DParams,
    LinearConvection2DState,
    NonlinearConvection2D,
    NonlinearConvection2DDiagnostics,
    NonlinearConvection2DState,
    NSDiagnostics,
    NSParams,
    NSVorticityState,
    PoissonSolver2D,
)
from sesmax_core._src.testing.swm import (
    LinearShallowWater1D,
    LinearShallowWater2D,
    LinearSW1DDiagnostics,
    LinearSW1DParams,
    LinearSW1DPhysConsts,
    LinearSW1DState,
    LinearSW2DDiagnostics,
    LinearSW2DParams,
    LinearSW2DPhysConsts,
    LinearSW2DState,
    LinearSWM2DTermModel,
    LinearSWMCoriolis,
    LinearSWMDiffusion,
    LinearSWMDrag,
    LinearSWMGravityWave,
    NonlinearShallowWater1D,
    NonlinearSW1DDiagnostics,
    NonlinearSW1DParams,
    NonlinearSW1DPhysConsts,
    NonlinearSW1DState,
)


__all__ = [
    "Burgers1D",
    "Burgers1DDiagnostics",
    "Burgers1DParams",
    "Burgers1DState",
    "Burgers2D",
    "Burgers2DAdvection",
    "Burgers2DDiagnostics",
    "Burgers2DDiffusion",
    "Burgers2DOp",
    "Burgers2DParams",
    "Burgers2DState",
    "Burgers2DTermModel",
    "Diffusion1D",
    "Diffusion1DDiagnostics",
    "Diffusion1DParams",
    "Diffusion1DState",
    "Diffusion2D",
    "Diffusion2DDiagnostics",
    "Diffusion2DParams",
    "Diffusion2DState",
    "HelmholtzSolver2D",
    "IncompressibleNS2D",
    "L63Diagnostics",
    "L63Params",
    "L63State",
    "L96Diagnostics",
    "L96Params",
    "L96State",
    "L96TParams",
    "L96TState",
    "LaplaceSolver2D",
    "LinearConvection1D",
    "LinearConvection1DDiagnostics",
    "LinearConvection1DParams",
    "LinearConvection1DState",
    "LinearConvection2D",
    "LinearConvection2DDiagnostics",
    "LinearConvection2DParams",
    "LinearConvection2DState",
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
    "Lorenz63",
    "Lorenz96",
    "Lorenz96t",
    "NSDiagnostics",
    "NSParams",
    "NSVorticityState",
    "NonlinearConvection1D",
    "NonlinearConvection1DDiagnostics",
    "NonlinearConvection1DState",
    "NonlinearConvection2D",
    "NonlinearConvection2DDiagnostics",
    "NonlinearConvection2DState",
    "NonlinearSW1DDiagnostics",
    "NonlinearSW1DParams",
    "NonlinearSW1DPhysConsts",
    "NonlinearSW1DState",
    "NonlinearShallowWater1D",
    "PoissonSolver2D",
    "geostrophic_adjustment_2d",
    "gravity_wave_1d",
    "inertial_oscillation_1d",
]
