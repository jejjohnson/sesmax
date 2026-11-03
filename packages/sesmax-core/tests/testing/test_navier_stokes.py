"""Tests for 2D incompressible Navier-Stokes."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core import Diagnostics, Model, Params, State
from sesmax_core._src.testing.pde2d.navier_stokes import (
    GHIA_RE100_U,
    GHIA_RE100_V,
    GHIA_RE100_X,
    GHIA_RE100_Y,
)
from sesmax_core.testing import (
    IncompressibleNS2D,
    NSDiagnostics,
    NSParams,
    NSVorticityState,
)


class TestIncompressibleNS2D:
    def test_is_somax_model(self):
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        assert isinstance(model, Model)

    def test_state_types(self):
        assert issubclass(NSVorticityState, State)
        assert issubclass(NSParams, Params)
        assert issubclass(NSDiagnostics, Diagnostics)

    def test_vector_field_finite(self):
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        state0 = model.apply_boundary_conditions(state0)
        tendency = model.vector_field(0.0, state0)
        assert jnp.all(jnp.isfinite(tendency.omega))

    def test_integrate_cavity(self):
        """Lid-driven cavity should integrate without NaN."""
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1, problem="cavity")
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        assert jnp.all(jnp.isfinite(sol.ys.omega))

    def test_integrate_channel(self):
        """Channel flow should integrate without NaN and use periodic-x BCs."""
        model = IncompressibleNS2D.create(
            nx=16,
            ny=16,
            nu=0.1,
            problem="channel",
            body_force=1.0,
            u_lid=0.0,
        )
        # Verify channel configuration
        assert model.problem == "channel"
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        assert jnp.all(jnp.isfinite(sol.ys.omega))
        # Channel BCs enforce periodicity in x: left/right ghost cells should match
        omega_final = sol.ys.omega[0]
        state_bc = model.apply_boundary_conditions(NSVorticityState(omega=omega_final))
        interior_rows = slice(1, -1)
        assert jnp.allclose(
            state_bc.omega[interior_rows, 0], state_bc.omega[interior_rows, -2]
        )

    def test_diagnose(self):
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        diag = model.diagnose(state0)
        assert isinstance(diag, NSDiagnostics)
        assert diag.psi.shape == omega0.shape
        assert diag.u.shape == omega0.shape
        assert jnp.isfinite(diag.kinetic_energy)

    def test_grad_through_nu(self):
        """jax.grad should flow through viscosity."""
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1)
        omega0 = 0.1 * jnp.ones((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)

        @eqx.filter_grad
        def grad_fn(m):
            sol = m.integrate(state0, t0=0.0, t1=0.02, dt=0.001)
            return jnp.sum(sol.ys.omega**2)

        grads = grad_fn(model)
        assert jnp.isfinite(grads.params.nu)

    def test_cavity_develops_vorticity(self):
        """Lid-driven cavity should develop non-zero vorticity."""
        model = IncompressibleNS2D.create(
            nx=16, ny=16, nu=0.1, problem="cavity", u_lid=1.0
        )
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state0 = NSVorticityState(omega=omega0)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.1,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        omega_final = sol.ys.omega[0]
        # Vorticity should be non-zero (lid drives flow)
        assert float(jnp.max(jnp.abs(omega_final[1:-1, 1:-1]))) > 0.01

    def test_lid_vorticity_on_wall_row(self):
        """Thom's formula sets the lid's vorticity on the wall row, not the ghost.

        The DST solve puts psi = 0 on rows and columns 0 and -2 (the walls);
        from rest psi is zero everywhere, so the lid row holds -2 u_lid / dy.
        """
        model = IncompressibleNS2D.create(nx=16, ny=16, nu=0.1, u_lid=1.0)
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        state = model.apply_boundary_conditions(NSVorticityState(omega=omega0))
        lid = -2.0 * model.u_lid / model.grid.dy
        assert jnp.allclose(state.omega[-2, 1:-2], lid)
        assert jnp.allclose(state.omega[-1, :], 0.0)
        assert jnp.allclose(state.omega[1:-2, -1], 0.0)

    @pytest.mark.slow
    def test_cavity_matches_ghia_re100(self):
        """At Re = 100 the steady centreline velocities match Ghia et al. (1982).

        psi, u and v live on the X nodes j * dy, j = 0..ny (walls at 0 and ny);
        the wall values come from the boundary conditions. The 0.03 bound is
        three times the deviation of a 32x32 upwind run at t = 10 (about 0.01).
        """
        n = 32
        model = IncompressibleNS2D.create(nx=n, ny=n, nu=0.01, u_lid=1.0)
        omega0 = jnp.zeros((model.grid.Ny, model.grid.Nx))
        sol = model.integrate(
            NSVorticityState(omega=omega0),
            t0=0.0,
            t1=10.0,
            dt=0.002,
            saveat=dfx.SaveAt(t1=True),
            max_steps=10_000,
        )
        diag = model.diagnose(NSVorticityState(omega=sol.ys.omega[0]))
        nodes = np.arange(n + 1) / n
        u_centre = np.array(diag.u[: n + 1, n // 2])
        u_centre[0], u_centre[-1] = 0.0, model.u_lid
        v_centre = np.array(diag.v[n // 2, : n + 1])
        v_centre[0], v_centre[-1] = 0.0, 0.0
        u_dev = np.interp(np.asarray(GHIA_RE100_Y), nodes, u_centre) - GHIA_RE100_U
        v_dev = np.interp(np.asarray(GHIA_RE100_X), nodes, v_centre) - GHIA_RE100_V
        assert float(np.abs(u_dev).max()) < 0.03
        assert float(np.abs(v_dev).max()) < 0.03

    def test_unknown_problem_is_rejected(self):
        with pytest.raises(ValueError, match="problem must be one of"):
            IncompressibleNS2D.create(nx=8, ny=8, problem="annulus")

    def test_cavity_carries_a_zero_flux(self):
        """The flux leaf exists in every problem but only the channel moves it."""
        model = IncompressibleNS2D.create(nx=8, ny=8, nu=0.1)
        state = model.apply_boundary_conditions(
            NSVorticityState(omega=jnp.ones((10, 10)))
        )
        assert float(model.vector_field(0.0, state).flux) == 0.0

    def test_periodic_box_wraps_the_ghost_ring(self):
        """No walls: the ghost ring copies the opposite interior edge."""
        model = IncompressibleNS2D.create(nx=8, ny=8, nu=0.1, problem="periodic")
        omega = jnp.arange(100.0).reshape(10, 10)
        state = model.apply_boundary_conditions(NSVorticityState(omega=omega))
        assert jnp.allclose(state.omega[1:-1, 0], omega[1:-1, -2])
        assert jnp.allclose(state.omega[1:-1, -1], omega[1:-1, 1])
        assert jnp.allclose(state.omega[0, 1:-1], omega[-2, 1:-1])
        assert jnp.allclose(state.omega[1:-1, 1:-1], omega[1:-1, 1:-1])

    def test_channel_wall_vorticity_uses_the_flux(self):
        """From rest with Q > 0, Thom's formula gives the shear of psi = Q y / Ly."""
        model = IncompressibleNS2D.create(
            nx=8, ny=8, Lx=2.0, Ly=1.0, nu=0.1, problem="channel"
        )
        state = model.apply_boundary_conditions(
            NSVorticityState(omega=jnp.zeros((10, 10)), flux=jnp.asarray(0.5))
        )
        dy = model.grid.dy
        # psi[1] = Q dy / Ly and psi[-3] = Q (Ly - dy) / Ly at zero interior omega
        assert jnp.allclose(state.omega[0, 1:-1], -2.0 * 0.5 * dy / dy**2, rtol=1e-5)
        assert jnp.allclose(state.omega[-2, 1:-1], 2.0 * 0.5 * dy / dy**2, rtol=1e-5)

    def test_channel_stays_uniform_in_x(self):
        """A pressure-driven channel from rest has no x-dependence to grow."""
        model = IncompressibleNS2D.create(
            nx=8, ny=8, Lx=2.0, Ly=1.0, nu=0.1, problem="channel", pressure_gradient=1.0
        )
        state = model.apply_boundary_conditions(
            NSVorticityState(omega=jnp.zeros((10, 10)), flux=jnp.asarray(0.5))
        )
        tendency = model.vector_field(0.0, state)
        interior = tendency.omega[1:-2, 1:-1]
        assert jnp.allclose(interior, interior[:, :1], atol=1e-5)
        assert float(tendency.flux) == pytest.approx(
            1.0 * 1.0
            - 0.1
            * (
                float(jnp.mean(state.omega[-2, 1:-1]))
                - float(jnp.mean(state.omega[0, 1:-1]))
            ),
            rel=1e-5,
        )

    def test_channel_and_box_take_no_mask(self):
        from finitevolx import CartesianGrid2D, Mask2D

        grid = CartesianGrid2D.from_interior(8, 8, 1.0, 1.0)
        mask = Mask2D.from_dimensions(grid.Ny, grid.Nx)
        with pytest.raises(ValueError, match="does not take a mask"):
            IncompressibleNS2D.create(nx=8, ny=8, problem="periodic", mask=mask)

    @pytest.mark.slow
    def test_periodic_taylor_green_decays_at_the_viscous_rate(self):
        """omega = 2k sin(kx) sin(ky) e^(-2 nu k^2 t) is exact on the periodic box.

        The 0.03 bound on the RMS error and 2e-3 on the amplitude ratio are
        about twice what a 32x32 upwind3 run measures (0.013 and 4e-4).
        """
        n, nu, t1 = 32, 0.01, 0.5
        model = IncompressibleNS2D.create(
            nx=n, ny=n, nu=nu, problem="periodic", method="upwind3"
        )
        k = 2.0 * jnp.pi
        x = jnp.arange(n + 2) * model.grid.dx
        grid_x, grid_y = jnp.meshgrid(x, x)
        omega0 = 2.0 * k * jnp.sin(k * grid_x) * jnp.sin(k * grid_y)
        sol = model.integrate(
            NSVorticityState(omega=omega0),
            t0=0.0,
            t1=t1,
            dt=0.001,
            saveat=dfx.SaveAt(t1=True),
        )
        decay = float(jnp.exp(-2.0 * nu * k**2 * t1))
        error = (sol.ys.omega[0] - omega0 * decay)[1:-1, 1:-1]
        ratio = float(jnp.max(jnp.abs(sol.ys.omega[0][1:-1, 1:-1])) / (2.0 * k))
        assert float(jnp.sqrt(jnp.mean(error**2))) < 0.03
        assert abs(ratio - decay) < 2e-3

    @pytest.mark.slow
    def test_channel_reaches_poiseuille(self):
        """From rest, the channel settles to u = (G / 2 nu) y (Ly - y).

        At t = 10 (ten viscous times Ly^2 / nu) a 32x32 run measures a flux
        error of 0.2% and a velocity RMS error of 5e-5 against a peak of
        1.25; the bounds are 1% and 1e-3.
        """
        n, G, nu, Ly = 32, 1.0, 0.1, 1.0
        model = IncompressibleNS2D.create(
            nx=n, ny=n, Lx=2.0, Ly=Ly, nu=nu, problem="channel", pressure_gradient=G
        )
        sol = model.integrate(
            NSVorticityState(omega=jnp.zeros((n + 2, n + 2))),
            t0=0.0,
            t1=10.0,
            dt=0.002,
            saveat=dfx.SaveAt(t1=True),
            max_steps=10_000,
        )
        final = NSVorticityState(omega=sol.ys.omega[0], flux=sol.ys.flux[0])
        psi = model.diagnose(model.apply_boundary_conditions(final)).psi
        exact_flux = G * Ly**3 / (12.0 * nu)
        assert abs(float(final.flux) - exact_flux) / exact_flux < 0.01
        y = jnp.arange(1, n) * model.grid.dy
        u_centred = (psi[2 : n + 1, 1:-1] - psi[0 : n - 1, 1:-1]) / (
            2.0 * model.grid.dy
        )
        u_exact = (G / (2.0 * nu)) * y * (Ly - y)
        rms = jnp.sqrt(jnp.mean((u_centred - u_exact[:, None]) ** 2))
        assert float(rms) < 1e-3
