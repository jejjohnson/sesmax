"""Public surface for sesmax-core's layered-fluid pieces.

:class:`StratificationProfile` holds a layered fluid's thicknesses and
reduced gravities; :class:`ModalTransform` projects between layers and
vertical modes, with the eigendecomposition delegated to
:mod:`sesmax_core.vertical`. The three elliptic inversions recover the
streamfunction from vorticity, (∇² - λ) ψ = ζ, the pressure from a
divergence, ∇² p = ∇ · u, and the streamfunction of each layer from its QG
potential vorticity, (∇² - λ_k) ψ_k = q_k, composed from gridax's grid and
mask pieces and gaussx's solves. The Burger-number helpers turn the
dimensionless numbers of a ``from_nondimensional`` factory into reduced
gravities. An ocean and an atmosphere share them, so they live in the core.

All but the Burger-number helpers are also importable from the top-level
``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.gfd.burger import (
    burger_to_g_prime,
    resolve_burger,
    resolve_burger_spherical,
)
from sesmax_core._src.gfd.inversion import (
    pressure_from_divergence,
    pv_inversion,
    streamfunction_from_vorticity,
)
from sesmax_core._src.gfd.modal import ModalTransform, StratificationProfile


__all__ = [
    "ModalTransform",
    "StratificationProfile",
    "burger_to_g_prime",
    "pressure_from_divergence",
    "pv_inversion",
    "resolve_burger",
    "resolve_burger_spherical",
    "streamfunction_from_vorticity",
]
