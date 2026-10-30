"""Input validators shared by the model factories.

Moved from somax 9cb19b3: the three validators of
``somax/_src/models/_nondim.py`` (``require_positive``,
``require_non_negative``, ``reject_derived_kwargs``). The Burger-number
helpers of that module live in ``sesmax_core._src.gfd``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def require_positive(context: str, **values: float) -> None:
    """Raise if any named value is not finite and strictly positive."""
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{context}: {name} must be a finite positive number; got {value!r}."
            )


def require_non_negative(context: str, **values: float) -> None:
    """Raise if any named value is negative or not finite."""
    for name, value in values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"{context}: {name} must be a finite non-negative number; "
                f"got {value!r}."
            )


def reject_derived_kwargs(
    context: str, derived: Sequence[str], **create_kw: object
) -> None:
    """Refuse forwarded arguments the factory derives for itself.

    ``**create_kw`` is a convenience for the knobs a dimensionless
    description says nothing about — ``bc``, ``method``, ``mask``,
    ``wind_profile``. Letting a stratification or a Coriolis parameter
    through it would silently win over the values derived from the
    dimensionless inputs, while the returned ``Scales`` still described
    the derived ones: the model and its scales would then be different
    systems.

    Args:
        context: Caller name, for error messages.
        derived: Argument names the factory sets itself.
        **create_kw: The forwarded arguments to check.

    Raises:
        ValueError: If any forwarded name is one the factory derives.
    """
    clash = sorted(set(create_kw) & set(derived))
    if clash:
        raise ValueError(
            f"{context}: {clash} is derived from the dimensionless inputs and "
            f"cannot be passed through to create(). Passing it would leave the "
            f"model and the returned Scales describing different systems; drop "
            f"it, or use create() directly for a dimensional build."
        )
