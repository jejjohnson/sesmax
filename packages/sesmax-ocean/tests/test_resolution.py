"""The resolution checks against their closed forms, on stand-in models.

The tests on the models themselves move with the models.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gridax import CartesianCGrid2D, SphericalCGrid2D

from sesmax_ocean.resolution import (
    AssertionFailedError,
    check_deformation_radius,
    check_equatorial_deformation_radius,
    check_munk_width,
    check_stommel_width,
)


# dx = 10, dy = 1: the western boundary layers are normal to x
GRID = CartesianCGrid2D.from_interior(8, 8, 80.0, 8.0)


def _gyre(*, nu: float = 1.0, kappa: float = 1.0, beta: float = 1.0):
    return SimpleNamespace(
        grid=GRID,
        consts=SimpleNamespace(beta=beta),
        params=SimpleNamespace(lateral_viscosity=nu, bottom_drag=kappa),
    )


def test_munk_width_is_the_cube_root_of_viscosity_over_beta() -> None:
    # δ_M = (nu / β)^(1/3) = 50 = 5 dx
    check_munk_width(None, _gyre(nu=50.0**3), n_cells_min=4.9, n_cells_warn=0.0)
    with pytest.raises(AssertionFailedError, match=r"delta_M/dx = 5\.00"):
        check_munk_width(None, _gyre(nu=50.0**3), n_cells_min=5.1)


def test_stommel_width_is_drag_over_beta() -> None:
    # δ_S = κ / β = 30 = 3 dx
    check_stommel_width(
        None, _gyre(kappa=60.0, beta=2.0), n_cells_min=2.9, n_cells_warn=0.0
    )
    with pytest.raises(AssertionFailedError, match=r"delta_S/dx = 3\.00"):
        check_stommel_width(None, _gyre(kappa=60.0, beta=2.0), n_cells_min=3.1)


def test_a_marginal_layer_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sesmax_ocean._src.resolution._logger", None)
    with pytest.warns(UserWarning, match="Munk layer marginally resolved"):
        check_munk_width(None, _gyre(nu=30.0**3))


@pytest.mark.parametrize("nu", [0.0, -1.0, float("nan")])
def test_a_viscosity_without_a_layer_is_rejected(nu: float) -> None:
    with pytest.raises(AssertionFailedError, match="lateral_viscosity"):
        check_munk_width(None, _gyre(nu=nu))


def test_deformation_radius_needs_the_coarser_spacing() -> None:
    # L_d = √(g' H) / f₀ = 40 = 4 dx, though it is 40 dy
    layered = SimpleNamespace(
        grid=GRID,
        consts=SimpleNamespace(f0=1.0),
        strat=SimpleNamespace(g_prime=[1e6, 16.0], H=[100.0, 100.0]),
    )
    check_deformation_radius(None, layered, n_cells_min=3.9, n_cells_warn=0.0)
    with pytest.raises(AssertionFailedError, match=r"L_d/dx = 4\.00"):
        check_deformation_radius(None, layered, n_cells_min=4.1)


def test_non_stratified_model_rejected() -> None:
    with pytest.raises(AssertionFailedError, match="lacks strat"):
        check_deformation_radius(None, SimpleNamespace(grid=GRID))


def test_equatorial_radius_is_measured_at_the_widest_cell() -> None:
    # Bu = gH / (2Ω a)² = 1/16, so L_eq / a = 1/2
    sphere = SimpleNamespace(
        grid=SphericalCGrid2D.from_interior(8, 7, (0.0, 360.0), (-70.0, 70.0), R=1.0),
        consts=SimpleNamespace(gravity=1.0 / 16.0, H0=1.0, omega=0.5, radius=1.0),
    )
    # 7 rows from -70° to 70° put one on the equator, where dx = 2π/8
    ratio = 0.5 / 0.7853981633974483
    check_equatorial_deformation_radius(
        None, sphere, n_cells_min=0.99 * ratio, n_cells_warn=0.0
    )
    with pytest.raises(AssertionFailedError, match=r"L_eq/dx = 0\.64"):
        check_equatorial_deformation_radius(None, sphere, n_cells_min=1.01 * ratio)


def test_equatorial_radius_needs_a_sphere() -> None:
    flat = SimpleNamespace(
        grid=GRID,
        consts=SimpleNamespace(gravity=1.0, H0=1.0, omega=1.0, radius=1.0),
    )
    with pytest.raises(AssertionFailedError, match="rigid-lid"):
        check_equatorial_deformation_radius(None, flat)
    with pytest.raises(AssertionFailedError, match="lacks consts/grid"):
        check_equatorial_deformation_radius(None, object())
