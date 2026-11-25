"""The public validators and Burger-number helpers of the model factories."""

from __future__ import annotations

import math

import pytest

from sesmax_core.gfd import (
    burger_to_g_prime,
    resolve_burger,
    resolve_burger_spherical,
)
from sesmax_core.validation import (
    reject_derived_kwargs,
    require_non_negative,
    require_positive,
)


@pytest.mark.parametrize("value", [-1.0, math.nan, math.inf, -math.inf])
def test_both_reject_what_is_negative_or_not_finite(value: float) -> None:
    for check in (require_positive, require_non_negative):
        with pytest.raises(ValueError, match=r"factory: drag must be a finite"):
            check("factory", drag=value)


def test_only_require_non_negative_accepts_zero() -> None:
    require_non_negative("factory", drag=0.0, viscosity=2.0)
    with pytest.raises(ValueError, match="drag must be a finite positive"):
        require_positive("factory", drag=0.0)


def test_reject_derived_kwargs_names_the_clash() -> None:
    reject_derived_kwargs("factory", ("f0", "beta"), bc="wall")
    with pytest.raises(ValueError, match=r"factory: \['beta'\] is derived"):
        reject_derived_kwargs("factory", ("f0", "beta"), bc="wall", beta=1.0)


def test_burger_from_froude_and_lamb() -> None:
    # Bu = (Ro / Fr)²
    assert resolve_burger("factory", None, 0.5, 0.1) == pytest.approx(0.04)
    # ε = 1 / Bu
    assert resolve_burger_spherical("factory", None, None, 4.0, 0.1) == 0.25
    with pytest.raises(ValueError, match="exactly one of burger or froude"):
        resolve_burger("factory", 1.0, 0.5, 0.1)
    with pytest.raises(ValueError, match="exactly one of burger, froude or lamb"):
        resolve_burger_spherical("factory", None, None, None, 0.1)


def test_reduced_gravities_invert_the_burger_numbers() -> None:
    f0, length = 2.0, 3.0
    burger, thickness = (0.5, 0.25), (4.0, 8.0)
    g_prime = burger_to_g_prime("factory", burger, thickness, f0, length)
    # Bu_k = g'_k H_k / (f₀ L)²
    recovered = [
        g * h / (f0 * length) ** 2 for g, h in zip(g_prime, thickness, strict=True)
    ]
    assert recovered == pytest.approx(list(burger))
    with pytest.raises(ValueError, match="same length"):
        burger_to_g_prime("factory", burger, (4.0,))
