"""Tests for integrating the shallow-water model in transformed coordinates.

Moved from somax 9cb19b3: the cases on this model of
``tests/test_scaled_model.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core import ScaledModel, Scales
from sesmax_ocean.swm import NonlinearShallowWater2D, NonlinearSW2DState


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


class TestEquivalenceShallowWater:
    """A multi-field state with a non-zero ``loc`` on one field."""

    @pytest.fixture
    def inner(self):
        return NonlinearShallowWater2D.create(
            nx=24, ny=24, Lx=1e6, Ly=1e6, f0=1e-4, H0=500.0, lateral_viscosity=50.0
        )

    @pytest.fixture
    def scales(self):
        return Scales.inertial(L=1e6, f0=1e-4, H=500.0, rossby=0.02)

    @pytest.fixture
    def state(self, inner, scales):
        rng = np.random.RandomState(2)
        shape = (inner.grid.Ny, inner.grid.Nx)
        return NonlinearSW2DState(
            h=jnp.asarray(scales.H + 0.05 * scales.eta * rng.randn(*shape)),
            u=jnp.asarray(0.05 * scales.U * rng.randn(*shape)),
            v=jnp.asarray(0.05 * scales.U * rng.randn(*shape)),
        )

    def test_trajectories_agree(self, inner, scales, state):
        model = ScaledModel.from_scales(inner, scales, NonlinearSW2DState)
        y0 = model.transform.forward(state)
        t_nd, n = 0.5, 20
        scaled = model.integrate(y0, t0=0.0, t1=t_nd, dt=t_nd / n).ys
        direct = inner.integrate(
            state, t0=0.0, t1=t_nd * scales.T, dt=t_nd * scales.T / n
        ).ys
        recovered = model.transform.inverse(scaled)
        for field in ("h", "u", "v"):
            assert (
                relative_error(getattr(recovered, field), getattr(direct, field)) < 1e-3
            ), field

    def test_nonzero_loc_is_handled(self, inner, scales, state):
        """``h`` has loc = H, so a sign error there would show up here."""
        model = ScaledModel.from_scales(inner, scales, NonlinearSW2DState)
        assert model.transform.loc.h == pytest.approx(scales.H)
        y0 = model.transform.forward(state)
        assert np.abs(np.asarray(y0.h)).max() < 10.0  # genuinely O(1)


def test_the_nonlinear_model_still_gets_its_offset():
    """The default is unchanged for states that mean total thickness."""
    inner = NonlinearShallowWater2D.create(nx=16, ny=16, H0=500.0)
    scales = Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=0.01)
    wrapper = ScaledModel.from_scales(inner, scales, NonlinearSW2DState)
    assert float(wrapper.transform.loc.h) == pytest.approx(500.0)
