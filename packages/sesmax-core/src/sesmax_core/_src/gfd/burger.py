"""Burger-number helpers for the layered model factories.

Moved from somax 9cb19b3: ``resolve_burger``, ``burger_to_g_prime`` and
``resolve_burger_spherical`` of ``somax/_src/models/_nondim.py``, whose
module docstring explains the two Burger conventions (per interface and
per vertical mode). Its validators live in ``sesmax_core._src._validation``.
"""

from __future__ import annotations

from collections.abc import Sequence

from sesmax_core._src._validation import require_positive


def resolve_burger(
    context: str,
    burger: float | None,
    froude: float | None,
    rossby: float,
) -> float:
    """Settle the Burger number from ``burger`` or ``froude``.

    The two are not independent once the Rossby number is fixed:
    ``Bu = (Ro / Fr)**2``. Accepting both would let a caller state an
    inconsistent pair, so exactly one must be given.

    Args:
        context: Caller name, for error messages.
        burger: Burger number, or ``None``.
        froude: Froude number ``U / sqrt(gH)``, or ``None``.
        rossby: Rossby number, already validated.

    Returns:
        The Burger number.

    Raises:
        ValueError: If both or neither is supplied, or the value is not
            positive.
    """
    if (burger is None) == (froude is None):
        raise ValueError(
            f"{context}: give exactly one of burger or froude. They are "
            f"related by Bu = (Ro/Fr)**2 at fixed Ro, so supplying both "
            f"allows an inconsistent pair."
        )
    if froude is not None:
        require_positive(context, froude=froude)
        return float((rossby / froude) ** 2)
    require_positive(context, burger=burger)
    return float(burger)


def burger_to_g_prime(
    context: str,
    burger: Sequence[float],
    thickness: Sequence[float],
    f0: float = 1.0,
    length: float = 1.0,
) -> tuple[float, ...]:
    """Invert the per-interface Burger numbers into reduced gravities.

    ``Bu_k = g'_k H_k / (f0 L)**2`` gives
    ``g'_k = Bu_k * (f0 L)**2 / H_k``.

    Args:
        context: Caller name, for error messages.
        burger: Per-interface Burger numbers, top to bottom.
        thickness: Per-layer thicknesses in the same units the model
            will use, same length as ``burger``.
        f0: Reference Coriolis parameter of the target model.
        length: Horizontal length scale of the target model.

    Returns:
        Reduced gravities, one per interface.

    Raises:
        ValueError: If the sequences differ in length, are empty, or
            contain a non-positive entry.
    """
    if len(burger) != len(thickness):
        raise ValueError(
            f"{context}: burger and thickness_ratio must have the same "
            f"length; got {len(burger)} and {len(thickness)}."
        )
    # ``len(...) == 0`` rather than ``not burger``: a multi-element
    # NumPy or JAX array is a perfectly good sequence here, but its
    # truth value is ambiguous and raises before the conversion below.
    if len(burger) == 0:
        raise ValueError(f"{context}: burger must not be empty.")
    for index, (bu, h) in enumerate(zip(burger, thickness, strict=True)):
        require_positive(context, **{f"burger[{index}]": bu, f"thickness[{index}]": h})
    factor = (f0 * length) ** 2
    return tuple(
        float(bu) * factor / float(h) for bu, h in zip(burger, thickness, strict=True)
    )


def resolve_burger_spherical(
    context: str,
    burger: float | None,
    froude: float | None,
    lamb: float | None,
    rossby: float,
) -> float:
    """Settle the Burger number from ``burger``, ``froude`` or ``lamb``.

    The spherical literature states the stratification as the Lamb
    parameter ``eps = 4 Omega**2 a**2 / (g H)``, which is exactly the
    inverse Burger number. Accepting it alongside the two Cartesian
    spellings keeps spherical configs idiomatic without introducing a
    second, independent quantity — so exactly one of the three may be
    given.

    Args:
        context: Caller name, for error messages.
        burger: Burger number ``Bu = gH/(f0 a)**2``, or ``None``.
        froude: Froude number ``U/sqrt(gH)``, or ``None``.
        lamb: Lamb parameter ``eps = 1/Bu``, or ``None``.
        rossby: Rossby number, already validated.

    Returns:
        The Burger number.

    Raises:
        ValueError: If not exactly one of the three is supplied, or the
            value is not positive.
    """
    given = [
        name
        for name, value in (("burger", burger), ("froude", froude), ("lamb", lamb))
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"{context}: give exactly one of burger, froude or lamb; got "
            f"{given or 'none'}. They are not independent — Bu = (Ro/Fr)**2 "
            f"at fixed Ro, and lamb = 1/Bu."
        )
    if lamb is not None:
        require_positive(context, lamb=lamb)
        return 1.0 / float(lamb)
    return resolve_burger(context, burger, froude, rossby)
