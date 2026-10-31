"""Tests for GFD test case factories.

These cover the pedagogical 1D/2D linear cases that live in
``sesmax_core._src.testing.gfd_testcases``. The Phase-3 removal of the
double-gyre per-pair factories (see #77) shifted their coverage to the
``scenario x model`` registries; see ``tests/test_scenarios_registry.py``,
``tests/test_models_registry.py``, and ``tests/test_cli_run.py``.
"""

from __future__ import annotations

import jax.numpy as jnp

from sesmax_core import Model
from sesmax_core.testing import (
    LinearShallowWater1D,
    LinearShallowWater2D,
    geostrophic_adjustment_2d,
    gravity_wave_1d,
    inertial_oscillation_1d,
)


class TestGravityWave1D:
    def test_creates_valid_model_and_state(self):
        model, state0 = gravity_wave_1d(nx=50)
        assert isinstance(model, LinearShallowWater1D)
        assert isinstance(model, Model)
        assert state0.h.shape == (model.grid.Nx,)

    def test_integrates_finite(self):
        model, state0 = gravity_wave_1d(nx=50)
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))


class TestInertialOscillation1D:
    def test_creates_valid_model_and_state(self):
        model, state0 = inertial_oscillation_1d(nx=10)
        assert isinstance(model, LinearShallowWater1D)
        assert float(jnp.max(jnp.abs(state0.u))) > 0.0

    def test_integrates_finite(self):
        model, state0 = inertial_oscillation_1d(nx=10)
        period = 2.0 * jnp.pi / model.consts.f0
        sol = model.integrate(state0, t0=0.0, t1=float(period), dt=float(period / 100))
        assert jnp.all(jnp.isfinite(sol.ys.u))


class TestGeostrophicAdjustment2D:
    def test_creates_valid_model_and_state(self):
        model, state0 = geostrophic_adjustment_2d(nx=32, ny=32)
        assert isinstance(model, LinearShallowWater2D)
        assert state0.h.shape == (model.grid.Ny, model.grid.Nx)

    def test_integrates_finite(self):
        model, state0 = geostrophic_adjustment_2d(nx=32, ny=32)
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))
