"""Public surface for sesmax-core's term algebra.

A model's tendency is a sum of terms, each tagged explicit or implicit.
:func:`partition` splits a term tree into its explicit and implicit sums,
and :func:`build_diffrax_terms` turns them into the diffrax terms an
explicit or IMEX solver takes. :class:`OperatorTerm` lifts any pipekit
operator or graph into a term, so a physics term can be swapped for a
learned one.

The names other than :class:`OperatorTerm` are also importable from the
top-level ``sesmax_core``, as they were from ``somax.core``.
"""

from __future__ import annotations

from sesmax_core._src.operator_term import OperatorTerm
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


__all__ = [
    "Compose",
    "OperatorTerm",
    "Scaled",
    "Sum",
    "Term",
    "TermFn",
    "build_diffrax_terms",
    "explicit",
    "implicit",
    "partition",
]
