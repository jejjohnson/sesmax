"""The edges of the band of `SphericalQG`: the seam in longitude and the walls."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gridax import area_sum
from gridax.schemes import WENO, Upwind
from somax_reference import float64

from sesmax_ocean.spherical import SphericalQG, SphericalQGState


NY, NX = 12, 16


def _model(**kwargs) -> SphericalQG:
    return SphericalQG.create(
        nx=NX, ny=NY, wind_profile="none", cg_tol=1e-13, cg_max_steps=2000, **kwargs
    )


def _vorticity() -> np.ndarray:
    return 1e-5 * np.random.default_rng(0).normal(size=(NY, NX))


def _state(model: SphericalQG, q: np.ndarray) -> SphericalQGState:
    q = jnp.asarray(np.pad(q, 1, mode="wrap"))
    return model.apply_boundary_conditions(SphericalQGState(q=q))


def _tendency(model: SphericalQG, q: np.ndarray) -> np.ndarray:
    return np.asarray(model.vector_field(0.0, _state(model, q)).q)[1:-1, 1:-1]


@pytest.mark.parametrize("scheme", [Upwind(1), WENO(5)], ids=["upwind1", "weno5"])
def test_a_shifted_state_has_the_shifted_tendency(scheme) -> None:
    with float64():
        model = _model(scheme=scheme, lateral_viscosity=1e6)
        q = _vorticity()
        here, there = _tendency(model, q), _tendency(model, np.roll(q, 7, axis=1))
    difference = np.abs(np.roll(here, 7, axis=1) - there).max()
    assert difference < 1e-9 * np.abs(here).max()


def test_a_mirrored_state_has_the_mirrored_tendency() -> None:
    """The equator is a plane of symmetry: both walls are treated alike."""
    with float64():
        model = _model(lateral_viscosity=1e6)
        q = _vorticity()
        here, there = _tendency(model, q), _tendency(model, -q[::-1])
    assert np.abs(here + there[::-1]).max() < 1e-9 * np.abs(here).max()


def test_the_flow_stays_in_the_band_and_is_non_divergent() -> None:
    with float64():
        model = _model()
        psi = model.invert_pv(_state(model, _vorticity()).q)
        u, v = model.velocities(psi)
        speed = float(jnp.abs(v).max())
        for wall in (0, -2):
            assert float(jnp.abs(v[wall]).max()) < 1e-12 * speed
        np.testing.assert_array_equal(u[:, 0], u[:, -2])
        divergence = np.asarray(model.ops.divergence(u, v))[1:-1, 1:-1]
        scale = speed / float(model.grid.R * model.grid.dlat)
    assert np.abs(divergence).max() < 1e-12 * scale


def test_the_circulation_is_conserved() -> None:
    """Advection and viscosity are fluxes, and none crosses the walls."""
    with float64():
        model = _model(lateral_viscosity=1e6)
        dq_dt = model.vector_field(0.0, _state(model, _vorticity())).q
        net = float(area_sum(dq_dt, model.grid))
        total = float(area_sum(jnp.abs(dq_dt), model.grid))
    assert abs(net) < 1e-13 * total
