"""Tests for barotropic quasi-geostrophic model.

Moved from somax 9cb19b3: ``tests/models/test_qg_barotropic.py``, and the cases
on this model of ``test_invariants.py`` and ``test_masks.py``.
"""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from gridax import Mask2D

from sesmax_core import Diagnostics, Model, Params, PhysConsts, State
from sesmax_ocean.qg import (
    BarotropicQG,
    BarotropicQGDiagnostics,
    BarotropicQGParams,
    BarotropicQGPhysConsts,
    BarotropicQGState,
)


class TestBarotropicQG:
    def test_is_a_model(self):
        model = BarotropicQG.create(nx=16, ny=16)
        assert isinstance(model, Model)

    def test_state_types(self):
        assert issubclass(BarotropicQGState, State)
        assert issubclass(BarotropicQGParams, Params)
        assert issubclass(BarotropicQGPhysConsts, PhysConsts)
        assert issubclass(BarotropicQGDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=100.0)
        q0 = 0.01 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.q))

    def test_integrate(self):
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=100.0, wind_amplitude=1e-5
        )
        q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        sol = model.integrate(state0, t0=0.0, t1=100.0, dt=1.0)
        assert isinstance(sol, dfx.Solution)
        assert jnp.all(jnp.isfinite(sol.ys.q))

    def test_wind_forcing_develops_pv(self):
        """Wind forcing should drive non-zero PV from rest."""
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=100.0, wind_amplitude=1e-5
        )
        q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        sol = model.integrate(
            state0, t0=0.0, t1=1000.0, dt=1.0, saveat=dfx.SaveAt(t1=True)
        )
        q_final = sol.ys.q[0]
        # PV should be non-zero in the interior
        assert float(jnp.max(jnp.abs(q_final[2:-2, 2:-2]))) > 1e-10

    def test_diagnose(self):
        model = BarotropicQG.create(nx=16, ny=16)
        q0 = 0.01 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state = BarotropicQGState(q=q0)
        diag = model.diagnose(state)
        assert isinstance(diag, BarotropicQGDiagnostics)
        assert diag.psi.shape == q0.shape
        assert diag.u.shape == q0.shape
        assert jnp.isfinite(diag.kinetic_energy)
        assert jnp.isfinite(diag.enstrophy)

    def test_grad_through_params(self):
        model = BarotropicQG.create(nx=16, ny=16, lateral_viscosity=100.0)
        q0 = 0.01 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=50.0, dt=1.0)
            return jnp.sum(sol.ys.q**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.lateral_viscosity)

    def test_zero_pv_zero_tendency(self):
        """Zero PV with no forcing should give zero tendency."""
        model = BarotropicQG.create(
            nx=16, ny=16, lateral_viscosity=0.0, wind_amplitude=0.0
        )
        q0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = BarotropicQGState(q=q0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert float(jnp.max(jnp.abs(tendency.q))) < 1e-12


class TestBarotropicQGInvariants:
    def test_energy_enstrophy_keys(self) -> None:
        model = BarotropicQG.create(nx=16, ny=16)
        state = BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))
        inv = model.diagnose(state).invariants()
        assert set(inv) == {"kinetic_energy", "enstrophy"}


def test_accepts_an_all_ocean_mask():
    mask = Mask2D.from_dimensions(ny=16, nx=16)
    model = BarotropicQG.create(nx=14, ny=14, mask=mask)
    assert model.ops.mask is mask


def _asymmetry(psi):
    """‖ψ(y) + ψ(L_y - y)‖ / ‖ψ‖ over the corners from wall to wall."""
    psi = psi[..., :-1, 1:-1]
    mirrored = jnp.flip(psi, axis=-2)
    return jnp.sqrt(jnp.sum((psi + mirrored) ** 2) / jnp.sum(psi**2))


def test_holds_q_at_zero_on_all_four_walls():
    """The walls are the corners 0 and N, where ψ is zero too (#212)."""
    model = BarotropicQG.create(nx=16, ny=16)
    state = model.apply_boundary_conditions(
        BarotropicQGState(q=jnp.ones(model.grid.shape))
    )
    for wall in (state.q[0], state.q[-2], state.q[:, 0], state.q[:, -2]):
        assert jnp.all(wall == 0.0)
    assert jnp.all(state.q[1:-2, 1:-2] == 1.0)


def test_a_double_gyre_from_rest_stays_a_mirror_image():
    """The wind is odd about the middle latitude, and so is ψ (#212).

    With q free on the northern and eastern walls the asymmetry was 5e-4 after
    ten days; with all four walls held it is that of float32 rounding.
    """
    model = BarotropicQG.create(
        nx=32,
        ny=32,
        Lx=1.0e6,
        Ly=1.0e6,
        lateral_viscosity=150.0,
        bottom_drag=1.0e-7,
        wind_amplitude=1.0e-12,
    )
    solution = model.integrate(
        BarotropicQGState(q=jnp.zeros(model.grid.shape)),
        t0=0.0,
        t1=10.0 * 86400.0,
        dt=3600.0,
        max_steps=1000,
    )
    final = jax.tree.map(lambda field: field[-1], solution.ys)
    assert _asymmetry(model.diagnose(final).psi) < 1e-5
