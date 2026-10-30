"""The core of sesmax: what every simple Earth-system model shares.

``sesmax_core`` holds the model-agnostic machinery moved from somax (at
commit ``9cb19b3``): the state and parameter types, the parameter
constraints, the model base classes and the term algebra. The steppers,
forcing, data-assimilation adapters, monitors and I/O join as the rest of
phase C1 of ``docs/design/workspace_and_core.md`` lands.

A model is an :class:`equinox.Module` whose ``vector_field`` is a pure
function of time, state and parameters, so ``jax.jit``, ``jax.grad`` and
``jax.vmap`` go through a whole integration window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sesmax_core._src.model import Model, TermModel
from sesmax_core._src.terms import (
    Compose,
    Scaled,
    Sum,
    Term,
    TermFn,
    build_diffrax_terms,
    explicit,
    implicit,
    partition,
)
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


if TYPE_CHECKING:
    from sesmax_core._src.operator_term import OperatorTerm


__version__ = "0.0.0"  # x-release-please-version

__all__ = [
    "Compose",
    "Diagnostics",
    "Model",
    "OperatorTerm",
    "Params",
    "PhysConsts",
    "Scaled",
    "State",
    "Sum",
    "Term",
    "TermFn",
    "TermModel",
    "__version__",
    "as_parameter",
    "build_diffrax_terms",
    "explicit",
    "frozen",
    "implicit",
    "interval",
    "partition",
    "positive",
    "trainable_mask",
]


def __getattr__(name: str) -> object:
    """Import ``OperatorTerm`` on first use.

    It needs pipekit, so a plain ``import sesmax_core`` stays pipekit-free.
    """
    if name == "OperatorTerm":
        from sesmax_core._src.operator_term import OperatorTerm

        return OperatorTerm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
