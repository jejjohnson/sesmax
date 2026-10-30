"""The core of sesmax: what every simple Earth-system model shares.

``sesmax_core`` holds the model-agnostic machinery moved from somax (at
commit ``9cb19b3``): the state and parameter types, the parameter
constraints, and the model base classes. The term algebra, steppers,
forcing, data-assimilation adapters, monitors and I/O join as the rest of
phase C1 of ``docs/design/workspace_and_core.md`` lands.

A model is an :class:`equinox.Module` whose ``vector_field`` is a pure
function of time, state and parameters, so ``jax.jit``, ``jax.grad`` and
``jax.vmap`` go through a whole integration window.
"""

from __future__ import annotations

from sesmax_core._src.model import Model, TermModel
from sesmax_core._src.types import (
    Diagnostics,
    Params,
    PhysConsts,
    State,
    as_parameter,
    frozen,
    interval,
    positive,
    trainable_mask,
)


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "Diagnostics",
    "Model",
    "Params",
    "PhysConsts",
    "State",
    "TermModel",
    "__version__",
    "as_parameter",
    "frozen",
    "interval",
    "positive",
    "trainable_mask",
]
