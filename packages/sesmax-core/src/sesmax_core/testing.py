"""Public surface for sesmax-core's test problems.

Standard test problems, not Earth-system components: the chaotic Lorenz
systems used as data-assimilation testbeds. The core's own tests run on
them, and the 1D and 2D PDE test problems and the linear shallow-water
models join them as they move from somax.

- :class:`Lorenz63`: three variables, ``dx/dt = sigma (y - x)``,
  ``dy/dt = x (rho - z) - y``, ``dz/dt = x y - beta z``.
- :class:`Lorenz96`: ``Dx`` variables on a ring,
  ``dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F``.
- :class:`Lorenz96t`: the two-tier version, slow ``X`` coupled to fast ``Y``.
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


__all__ = [
    "L63Diagnostics",
    "L63Params",
    "L63State",
    "L96Diagnostics",
    "L96Params",
    "L96State",
    "L96TParams",
    "L96TState",
    "Lorenz63",
    "Lorenz96",
    "Lorenz96t",
]
