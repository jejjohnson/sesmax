"""Tests for the multilayer 2D nonlinear shallow water model.

Moved from somax 9cb19b3: ``tests/models/test_swm_multilayer.py``,
``tests/models/test_conservation.py`` and the cases on this model of
``tests/models/test_invariants.py``. The two tests that build the model through
a scenario stay until the scenarios move.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import pytest

from sesmax_core import (
    Diagnostics,
    Model,
    Params,
    PhysConsts,
    State,
    StratificationProfile,
)
from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DDiagnostics,
    MultilayerSW2DParams,
    MultilayerSW2DPhysConsts,
    MultilayerSW2DState,
)


def _make_model(**kw):
    defaults = dict(
        nx=16,
        ny=16,
        n_layers=2,
        H=(500.0, 500.0),
        g_prime=(9.81, 0.02),
    )
    defaults.update(kw)
    return MultilayerShallowWater2D.create(**defaults)


def _rest_state(model):
    nl = model.consts.n_layers
    Ny, Nx = model.grid.Ny, model.grid.Nx
    h0 = jnp.ones((nl, Ny, Nx)) * model.strat.H[:, None, None]
    u0 = jnp.zeros((nl, Ny, Nx))
    v0 = jnp.zeros((nl, Ny, Nx))
    return MultilayerSW2DState(h=h0, u=u0, v=v0)


class TestMultilayerSW2DContract:
    def test_is_a_model(self):
        model = _make_model()
        assert isinstance(model, Model)

    def test_state_types(self):
        assert issubclass(MultilayerSW2DState, State)
        assert issubclass(MultilayerSW2DParams, Params)
        assert issubclass(MultilayerSW2DPhysConsts, PhysConsts)
        assert issubclass(MultilayerSW2DDiagnostics, Diagnostics)


class TestMultilayerSW2DCreate:
    def test_create_default(self):
        model = MultilayerShallowWater2D.create()
        assert model.consts.n_layers == 3

    def test_create_two_layer(self):
        model = _make_model()
        assert model.consts.n_layers == 2
        assert model.strat.H.shape == (2,)

    def test_create_from_stratification(self):
        strat = StratificationProfile.from_N2_constant(
            N2=1e-5, depth=4000.0, n_layers=4
        )
        model = MultilayerShallowWater2D.create(nx=16, ny=16, stratification=strat)
        assert model.consts.n_layers == 4

    def test_create_validation(self):
        """Mismatched n_layers, H, g_prime should raise ValueError."""
        try:
            MultilayerShallowWater2D.create(
                nx=16,
                ny=16,
                n_layers=3,
                H=(500.0, 500.0),
                g_prime=(9.81, 0.02),
            )
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass

    def test_state_shape(self):
        model = _make_model()
        state = _rest_state(model)
        assert state.h.shape == (2, model.grid.Ny, model.grid.Nx)
        assert state.u.shape == (2, model.grid.Ny, model.grid.Nx)
        assert state.v.shape == (2, model.grid.Ny, model.grid.Nx)


class TestMultilayerSW2DPhysics:
    def test_rest_state_zero_tendency(self):
        """Uniform h = H, u = v = 0, no forcing -> zero tendency."""
        model = _make_model()
        state0 = _rest_state(model)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        assert float(jnp.max(jnp.abs(tend.h))) < 1e-10
        assert float(jnp.max(jnp.abs(tend.u))) < 1e-10
        assert float(jnp.max(jnp.abs(tend.v))) < 1e-10

    def test_wind_forcing_top_layer_only(self):
        """Wind forcing should only affect the top layer."""
        model = _make_model(wind_amplitude=1e-5)
        state0 = _rest_state(model)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        # Top layer should have non-zero momentum tendency
        assert float(jnp.max(jnp.abs(tend.u[0]))) > 0
        # Bottom layer should be zero (no drag, no wind)
        assert float(jnp.max(jnp.abs(tend.u[-1]))) < 1e-15

    def test_bottom_drag_bottom_layer_only(self):
        """Bottom drag on uniform flow should only decelerate bottom layer."""
        model = _make_model(bottom_drag=1e-4)
        state0 = _rest_state(model)
        nl = model.consts.n_layers
        # Set uniform flow in both layers
        u_init = jnp.ones((nl, model.grid.Ny, model.grid.Nx))
        state0 = MultilayerSW2DState(h=state0.h, u=u_init, v=state0.v)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        # Bottom layer should have drag term: -kappa * u
        # Top layer drag contribution should be zero
        # (other terms may be nonzero due to Coriolis)
        assert jnp.all(jnp.isfinite(tend.u))

    def test_pressure_coupling_between_layers(self):
        """Height perturbation in layer 0 should create tendency in layer 1."""
        model = _make_model()
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        # Perturb top layer
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        # Layer 1 should have nonzero momentum tendency from pressure coupling
        assert float(jnp.max(jnp.abs(tend.u[1]))) > 0

    def test_vector_field_finite(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        state0 = model.apply_boundary_conditions(state0)
        tend = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tend.h))
        assert jnp.all(jnp.isfinite(tend.u))
        assert jnp.all(jnp.isfinite(tend.v))


class TestMultilayerSW2DBCs:
    def test_periodic_bcs(self):
        model = _make_model(bc="periodic")
        state0 = _rest_state(model)
        state_bc = model.apply_boundary_conditions(state0)
        assert state_bc.h.shape == state0.h.shape

    def test_wall_bcs(self):
        model = _make_model(bc="wall")
        state0 = _rest_state(model)
        nl = model.consts.n_layers
        u_init = jnp.ones((nl, model.grid.Ny, model.grid.Nx))
        state0 = MultilayerSW2DState(h=state0.h, u=u_init, v=state0.v)
        state_bc = model.apply_boundary_conditions(state0)
        # u should be zero at x-boundaries. somax's assertion read the ghost
        # column -1; the east wall is the face -2, and gridax fills the ghost
        # beyond it by reflection.
        assert jnp.allclose(state_bc.u[:, :, 0], 0.0)
        assert jnp.allclose(state_bc.u[:, :, -2], 0.0)


class TestMultilayerSW2DIntegration:
    def test_integrate_short(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.h))
        assert jnp.all(jnp.isfinite(sol.ys.u))

    def test_integrate_with_perturbation(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(1.0)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))

    def test_three_layer(self):
        model = MultilayerShallowWater2D.create(
            nx=16,
            ny=16,
            n_layers=3,
            H=(400.0, 1100.0, 2500.0),
            g_prime=(9.81, 0.025, 0.0125),
            lateral_viscosity=100.0,
        )
        state0 = _rest_state(model)
        sol = model.integrate(state0, t0=0.0, t1=1000.0, dt=10.0)
        assert jnp.all(jnp.isfinite(sol.ys.h))


class TestMultilayerSW2DDiagnostics:
    def test_diagnose_shapes(self):
        model = _make_model()
        state = _rest_state(model)
        diag = model.diagnose(state)
        assert isinstance(diag, MultilayerSW2DDiagnostics)
        assert diag.potential_vorticity.shape == (
            2,
            model.grid.Ny,
            model.grid.Nx,
        )
        assert diag.relative_vorticity.shape == (
            2,
            model.grid.Ny,
            model.grid.Nx,
        )
        assert diag.kinetic_energy_field.shape == (
            2,
            model.grid.Ny,
            model.grid.Nx,
        )
        assert jnp.isfinite(diag.total_energy)
        assert jnp.isfinite(diag.total_enstrophy)

    def test_energy_nonnegative(self):
        model = _make_model()
        state = _rest_state(model)
        diag = model.diagnose(state)
        assert jnp.all(diag.energy >= 0)


class TestMultilayerSW2DGrad:
    def test_grad_through_params(self):
        model = _make_model(lateral_viscosity=100.0)
        state0 = _rest_state(model)
        Ny, Nx = model.grid.Ny, model.grid.Nx
        h_pert = state0.h.at[0, Ny // 2, Nx // 2].add(0.1)
        state0 = MultilayerSW2DState(h=h_pert, u=state0.u, v=state0.v)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.h**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)


def _ml_model_state():
    model = MultilayerShallowWater2D.create(
        nx=16, ny=16, n_layers=2, H=(500.0, 1500.0), g_prime=(9.81, 0.02)
    )
    sh = (2, model.grid.Ny, model.grid.Nx)
    h = model.strat.H[:, None, None] * jnp.ones(sh)
    state = model.apply_boundary_conditions(
        MultilayerSW2DState(h=h, u=jnp.zeros(sh), v=jnp.zeros(sh))
    )
    return model, state


class TestMultilayerInvariants:
    def test_keys_present(self) -> None:
        model, state = _ml_model_state()
        inv = model.diagnose(state).invariants()
        assert set(inv) == {
            "mass",
            "total_energy",
            "potential_enstrophy",
            "casimir_q3",
        }

    def test_mass_positive_and_finite(self) -> None:
        model, state = _ml_model_state()
        inv = model.diagnose(state).invariants()
        assert float(inv["mass"]) > 0
        assert all(bool(jnp.isfinite(v)) for v in inv.values())

    def test_mass_matches_thickness_integral(self) -> None:
        model, state = _ml_model_state()
        diag = model.diagnose(state)
        s = (slice(None), slice(1, -1), slice(1, -1))
        cell_area = model.grid.spacing("x") * model.grid.spacing("y")
        expected = float(jnp.sum(state.h[s]) * cell_area)
        # diagnose() integrates with gridax's area_sum, sum(dA * h), which
        # rounds differently from sum(h) * dA in float32 (rel. ~5e-8).
        assert float(jnp.sum(diag.mass)) == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
def test_multilayer_mass_conserved_tightly() -> None:
    """Flux-form mass is conserved to near machine precision (periodic basin).

    No forcing / drag / viscosity, periodic BCs — the continuity equation is
    in flux form, so total mass per layer should be conserved up to the
    time-integration error, far tighter than the energy/enstrophy budget.
    """
    model = MultilayerShallowWater2D.create(
        nx=32,
        ny=32,
        n_layers=2,
        H=(500.0, 1500.0),
        g_prime=(9.81, 0.02),
        lateral_viscosity=0.0,
        bottom_drag=0.0,
        wind_amplitude=0.0,
        bc="periodic",
    )
    sh = (2, model.grid.Ny, model.grid.Nx)
    # A small geostrophic-ish thickness perturbation to make the run nontrivial.
    key = jnp.linspace(0.0, 1.0, model.grid.Nx)
    bump = 1.0 * jnp.sin(2 * jnp.pi * key)[None, None, :]
    h0 = model.strat.H[:, None, None] * jnp.ones(sh) + bump
    state0 = model.apply_boundary_conditions(
        MultilayerSW2DState(h=h0, u=jnp.zeros(sh), v=jnp.zeros(sh))
    )

    mass0 = jnp.sum(model.diagnose(state0).invariants()["mass"])
    sol = model.integrate(state0, t0=0.0, t1=200.0, dt=10.0)
    final = MultilayerSW2DState(h=sol.ys.h[-1], u=sol.ys.u[-1], v=sol.ys.v[-1])
    mass1 = jnp.sum(model.diagnose(final).invariants()["mass"])

    rel_drift = float(jnp.abs(mass1 - mass0) / jnp.abs(mass0))
    assert rel_drift < 1e-6


@pytest.mark.slow
def test_multilayer_energy_bounded_no_growth() -> None:
    """Unforced energy should not grow (implicit dissipation only decays it)."""
    model = MultilayerShallowWater2D.create(
        nx=32,
        ny=32,
        n_layers=2,
        H=(500.0, 1500.0),
        g_prime=(9.81, 0.02),
        lateral_viscosity=0.0,
        bottom_drag=0.0,
        wind_amplitude=0.0,
        bc="periodic",
    )
    sh = (2, model.grid.Ny, model.grid.Nx)
    key = jnp.linspace(0.0, 1.0, model.grid.Nx)
    bump = 1.0 * jnp.sin(2 * jnp.pi * key)[None, None, :]
    h0 = model.strat.H[:, None, None] * jnp.ones(sh) + bump
    state0 = model.apply_boundary_conditions(
        MultilayerSW2DState(h=h0, u=jnp.zeros(sh), v=jnp.zeros(sh))
    )

    e0 = float(model.diagnose(state0).invariants()["total_energy"])
    sol = model.integrate(state0, t0=0.0, t1=200.0, dt=10.0)
    final = MultilayerSW2DState(h=sol.ys.h[-1], u=sol.ys.u[-1], v=sol.ys.v[-1])
    e1 = float(model.diagnose(final).invariants()["total_energy"])

    # Allow a small relative tolerance for time-truncation wiggle; the point
    # is the unforced run does not *grow* energy.
    assert e1 <= e0 * (1.0 + 1e-3)
