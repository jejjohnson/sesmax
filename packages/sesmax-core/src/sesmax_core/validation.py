"""Public surface for sesmax-core's input validators.

The checks a model factory runs on its arguments before it builds anything;
each raises a ``ValueError`` naming the caller and the argument. For a check
inside a jitted tendency, use :mod:`sesmax_core.guards`.

Not re-exported from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src._validation import (
    reject_derived_kwargs,
    require_non_negative,
    require_positive,
)


__all__ = ["reject_derived_kwargs", "require_non_negative", "require_positive"]
