"""Public surface for sesmax-core's test problems.

Standard test problems, not Earth-system components: the chaotic Lorenz
systems used as data-assimilation testbeds, and the 1D PDE test problems
of the "12 steps to Navier-Stokes" (linear and nonlinear convection,
diffusion, Burgers) on finitevolX stencils. The core's own tests run on
them; the 2D PDE problems and the linear shallow-water models join as they
move from somax.

- :class:`Lorenz63`: three variables, ``dx/dt = sigma (y - x)``,
  ``dy/dt = x (rho - z) - y``, ``dz/dt = x y - beta z``.
- :class:`Lorenz96`: ``Dx`` variables on a ring,
  ``dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F``.
- :class:`Lorenz96t`: the two-tier version, slow ``X`` coupled to fast ``Y``.
- :class:`LinearConvection1D`, :class:`NonlinearConvection1D`,
  :class:`Diffusion1D`, :class:`Burgers1D`: ``u_t + c u_x = 0``,
  ``u_t + u u_x = 0``, ``u_t = nu u_xx`` and ``u_t + u u_x = nu u_xx`` on a
  periodic interval.
"""

from __future__ import annotations

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


__all__ = [
    "Burgers1D",
    "Burgers1DDiagnostics",
    "Burgers1DParams",
    "Burgers1DState",
    "Diffusion1D",
    "Diffusion1DDiagnostics",
    "Diffusion1DParams",
    "Diffusion1DState",
    "L63Diagnostics",
    "L63Params",
    "L63State",
    "L96Diagnostics",
    "L96Params",
    "L96State",
    "L96TParams",
    "L96TState",
    "LinearConvection1D",
    "LinearConvection1DDiagnostics",
    "LinearConvection1DParams",
    "LinearConvection1DState",
    "Lorenz63",
    "Lorenz96",
    "Lorenz96t",
    "NonlinearConvection1D",
    "NonlinearConvection1DDiagnostics",
    "NonlinearConvection1DState",
]
