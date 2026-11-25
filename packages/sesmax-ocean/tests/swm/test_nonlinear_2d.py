"""Tests for the 2D nonlinear shallow water model.

Moved from somax 9cb19b3: ``tests/models/test_swm_2d.py``, and the cases on this
model of ``tests/models/test_invariants.py`` and ``tests/models/test_masks.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from gridax import Mask2D

from sesmax_core import Diagnostics, Model, Params, PhysConsts, State
from sesmax_ocean.swm import (
    NonlinearShallowWater2D,
    NonlinearSW2DDiagnostics,
    NonlinearSW2DParams,
    NonlinearSW2DPhysConsts,
    NonlinearSW2DState,
)


def _gaussian_2d(X, Y, mu_x, mu_y, sigma):
    return jnp.exp(-0.5 * (((X - mu_x) / sigma) ** 2 + ((Y - mu_y) / sigma) ** 2))


class TestNonlinearShallowWater2D:
    def test_is_a_model(self):
        model = NonlinearShallowWater2D.create(nx=16, ny=16)
        assert isinstance(model, Model)

    def test_state_types(self):
        assert issubclass(NonlinearSW2DState, State)
        assert issubclass(NonlinearSW2DParams, Params)
        assert issubclass(NonlinearSW2DPhysConsts, PhysConsts)
        assert issubclass(NonlinearSW2DDiagnostics, Diagnostics)

    def test_integrate_finite(self):
        model = NonlinearShallowWater2D.create(
            nx=16, ny=16, H0=100.0, lateral_viscosity=100.0
        )
        h0 = jnp.full((model.grid.Ny, model.grid.Nx), 100.0)
        x = jnp.arange(model.grid.Nx) * model.grid.spacing("x")
        y = jnp.arange(model.grid.Ny) * model.grid.spacing("y")
        X, Y = jnp.meshgrid(x, y)
        h0 = h0 + 0.1 * _gaussian_2d(
            X, Y, model.grid.Lx / 2, model.grid.Ly / 2, model.grid.Lx / 5
        )
        state0 = NonlinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        sol = model.integrate(state0, t0=0.0, t1=10.0, dt=0.1)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_diagnose(self):
        model = NonlinearShallowWater2D.create(nx=16, ny=16, H0=100.0)
        h0 = jnp.full((model.grid.Ny, model.grid.Nx), 100.0)
        state = NonlinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))
        diag = model.diagnose(state)
        assert isinstance(diag, NonlinearSW2DDiagnostics)
        assert diag.potential_vorticity.shape == h0.shape
        assert diag.relative_vorticity.shape == h0.shape

    def test_grad_through_params(self):
        model = NonlinearShallowWater2D.create(
            nx=16, ny=16, H0=100.0, lateral_viscosity=100.0
        )
        h0 = jnp.full((model.grid.Ny, model.grid.Nx), 100.0)
        state0 = NonlinearSW2DState(h=h0, u=jnp.zeros_like(h0), v=jnp.zeros_like(h0))

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=5.0, dt=0.1)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)


class TestNonlinearSWInvariants:
    def test_keys_present(self) -> None:
        model = NonlinearShallowWater2D.create(nx=16, ny=16)
        sh = (model.grid.Ny, model.grid.Nx)
        H0 = model.consts.H0 if hasattr(model.consts, "H0") else 1000.0
        state = model.apply_boundary_conditions(
            NonlinearSW2DState(h=H0 * jnp.ones(sh), u=jnp.zeros(sh), v=jnp.zeros(sh))
        )
        inv = model.diagnose(state).invariants()
        assert set(inv) == {
            "mass",
            "total_energy",
            "potential_enstrophy",
            "casimir_q3",
        }


def _island_mask(ny: int, nx: int) -> Mask2D:
    """Rectangular basin with a small interior island."""
    h = np.ones((ny, nx), dtype=bool)
    h[0, :] = h[-1, :] = False
    h[:, 0] = h[:, -1] = False
    j0, j1 = ny // 2 - 1, ny // 2 + 2
    i0, i1 = nx // 2 - 1, nx // 2 + 2
    h[j0:j1, i0:i1] = False
    return Mask2D.from_mask(jnp.asarray(h))


def test_all_ocean_mask():
    mask = Mask2D.from_dimensions(ny=16, nx=16)
    model = NonlinearShallowWater2D.create(nx=14, ny=14, mask=mask)
    h0 = jnp.ones((model.grid.Ny, model.grid.Nx))
    u0 = jnp.zeros_like(h0)
    state = NonlinearSW2DState(h=h0, u=u0, v=u0)
    state = model.apply_boundary_conditions(state)
    tend = model.vector_field(0.0, state)
    assert jnp.all(jnp.isfinite(tend.h))
    assert jnp.all(jnp.isfinite(tend.u))
    assert jnp.all(jnp.isfinite(tend.v))


class TestCoastalMaskNonlinearSWM2D:
    @pytest.fixture
    def model_and_state(self):
        ny, nx = 20, 20
        mask = _island_mask(ny, nx)
        model = NonlinearShallowWater2D.create(
            nx=nx - 2, ny=ny - 2, bc="wall", mask=mask
        )
        rng = np.random.default_rng(0)
        # h is layer thickness in PV q = (ζ+f)/h, so keep wet-cell
        # thickness strictly positive (baseline ~1 + small perturbation)
        # to avoid division blow-ups; mask is applied on top to enforce
        # the dry-cell invariant.
        h0 = jnp.ones((ny, nx)) + jnp.asarray(0.01 * rng.standard_normal((ny, nx)))
        u0 = jnp.asarray(0.01 * rng.standard_normal((ny, nx)))
        v0 = jnp.asarray(0.01 * rng.standard_normal((ny, nx)))
        h0 = h0 * mask.wet(at="T")
        u0 = u0 * mask.wet(at="U")
        v0 = v0 * mask.wet(at="V")
        state = NonlinearSW2DState(h=h0, u=u0, v=v0)
        state = model.apply_boundary_conditions(state)
        return model, state, mask

    def test_dry_cells_stay_zero_after_one_rhs(self, model_and_state):
        """One RHS evaluation must not inject non-zero into dry T-cells."""
        model, state, mask = model_and_state
        tend = model.vector_field(0.0, state)
        # dh/dt comes through advection → T-point output → mask.wet(at="T") applies
        dry_T = ~mask.wet(at="T")
        assert float(jnp.max(jnp.abs(tend.h[dry_T]))) == 0.0

    def test_integrate_short_finite(self, model_and_state):
        """A few Euler steps remain finite with the coastal mask threaded."""
        model, state, _mask = model_and_state
        dt = 1.0

        @eqx.filter_jit
        def step(s):
            tend = model.vector_field(0.0, s)
            new = NonlinearSW2DState(
                h=s.h + dt * tend.h,
                u=s.u + dt * tend.u,
                v=s.v + dt * tend.v,
            )
            return model.apply_boundary_conditions(new)

        s = state
        for _ in range(3):
            s = step(s)

        assert jnp.all(jnp.isfinite(s.h))
        assert jnp.all(jnp.isfinite(s.u))
        assert jnp.all(jnp.isfinite(s.v))
