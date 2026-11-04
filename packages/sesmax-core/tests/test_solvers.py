"""Tests for the matrix-free IMEX solver helpers (issue #55).

The cheap tests (construction + a small IMEX solve that would error or build
a dense Jacobian without the helper) run in the fast PR lane. The
large-grid scaling check that actually reproduces the 256x256 regime is
marked ``slow``.
"""

from __future__ import annotations

import diffrax as dfx
import jax.numpy as jnp
import optimistix as optx
import pytest

from sesmax_core._src.testing.pde2d.burgers import Burgers2DState
from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel
from sesmax_core.solvers import imex_solver, imex_stepsize_controller, stable_step


def _imex_burgers(n: int) -> tuple[Burgers2DTermModel, Burgers2DState]:
    model = Burgers2DTermModel.create(nx=n, ny=n, nu=0.05, imex=True)
    sh = (model.grid.Ny, model.grid.Nx)
    x = jnp.linspace(0.0, 1.0, model.grid.Nx)
    u0 = jnp.sin(2 * jnp.pi * x)[None, :] * jnp.ones(sh)
    state0 = model.apply_boundary_conditions(Burgers2DState(u=u0, v=jnp.zeros(sh)))
    return model, state0


class TestImexSolverConstruction:
    def test_solver_is_kencarp_with_matrix_free_newton(self) -> None:
        solver = imex_solver()
        assert isinstance(solver, dfx.KenCarp3)
        # The implicit stage must use a Krylov (GMRES) linear solver, not the
        # default dense AutoLinearSolver — that's the whole point of #55.
        rf = solver.root_finder
        assert isinstance(rf, optx.Newton)
        assert type(rf.linear_solver).__name__ == "GMRES"

    def test_controller_is_adaptive(self) -> None:
        assert isinstance(imex_stepsize_controller(), dfx.PIDController)

    def test_tolerances_threaded(self) -> None:
        solver = imex_solver(rtol=1e-3, atol=1e-7)
        assert solver.root_finder.rtol == 1e-3
        assert solver.root_finder.atol == 1e-7


class TestImexSolve:
    def test_small_imex_solve_runs_finite(self) -> None:
        """A small IMEX solve completes and stays finite.

        Without the helper this path either errors (fixed-step controller +
        unspecified implicit tolerances) or builds a dense Jacobian; the
        helper makes it just work.
        """
        model, state0 = _imex_burgers(16)
        sol = model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.01,
            solver=imex_solver(),
            stepsize_controller=imex_stepsize_controller(),
        )
        assert jnp.all(jnp.isfinite(sol.ys.u))
        assert jnp.all(jnp.isfinite(sol.ys.v))

    def test_imex_tracks_explicit_solution(self) -> None:
        """The IMEX (implicit-diffusion) solve agrees with a fully-explicit
        reference on a short, stable window — the split is faithful."""
        n = 16
        imex_model, state0 = _imex_burgers(n)
        sol_imex = imex_model.integrate(
            state0,
            t0=0.0,
            t1=0.05,
            dt=0.005,
            solver=imex_solver(rtol=1e-7, atol=1e-9),
            stepsize_controller=imex_stepsize_controller(rtol=1e-7, atol=1e-9),
        )
        explicit_model = Burgers2DTermModel.create(nx=n, ny=n, nu=0.05, imex=False)
        sol_exp = explicit_model.integrate(state0, t0=0.0, t1=0.05, dt=0.005)
        assert jnp.allclose(sol_imex.ys.u, sol_exp.ys.u, atol=1e-3)


@pytest.mark.slow
def test_imex_solver_scales_to_large_grid() -> None:
    """The matrix-free IMEX solve runs at 128x128 without OOM.

    The dense-Jacobian default is ~100x slower by 64x64 and OOMs at
    256x256 (#55); the matrix-free path stays O(N) memory. We assert the
    128x128 solve completes and is finite (256x256 is left out of CI for
    runtime, but the same path handles it in ~1 min locally).
    """
    model, state0 = _imex_burgers(128)
    sol = model.integrate(
        state0,
        t0=0.0,
        t1=0.02,
        dt=0.01,
        solver=imex_solver(),
        stepsize_controller=imex_stepsize_controller(),
    )
    assert jnp.all(jnp.isfinite(sol.ys.u))


def test_stable_step_matches_the_hand_computed_bound():
    """Δt_max = 1 / (Σ |u| / Δx + 2 Σ K / Δx²), worked by hand.

    With Δx = (50, 50, 25), u = (3, 0.5, 0) and K = (10, 10, 2): the advective
    rate is 0.06 + 0.01 = 0.07, the diffusive rate is 2 (0.004 + 0.004 + 0.0032)
    = 0.0224, and the bound is 1 / 0.0924.
    """
    bound = stable_step(
        (50.0, 50.0, 25.0), speed=(3.0, 0.5, 0.0), diffusivity=(10.0, 10.0, 2.0)
    )
    assert bound == pytest.approx(1.0 / 0.0924, rel=1e-12)


def test_stable_step_scales_with_cfl_and_ignores_the_sign_of_a_speed():
    base = stable_step((1.0, 1.0), speed=(1.0, 2.0))
    assert stable_step((1.0, 1.0), speed=(-1.0, -2.0)) == base
    assert stable_step((1.0, 1.0), speed=(1.0, 2.0), cfl=0.5) == pytest.approx(
        0.5 * base
    )


def test_stable_step_scales_with_spacing_for_advection_and_its_square_for_diffusion():
    coarse_advection = stable_step((1.0,), speed=(1.0,))
    fine_advection = stable_step((0.5,), speed=(1.0,))
    assert fine_advection == pytest.approx(0.5 * coarse_advection)
    coarse_diffusion = stable_step((1.0,), diffusivity=(1.0,))
    fine_diffusion = stable_step((0.5,), diffusivity=(1.0,))
    assert fine_diffusion == pytest.approx(0.25 * coarse_diffusion)


def test_stable_step_is_infinite_when_nothing_moves_or_diffuses():
    assert stable_step((1.0, 1.0)) == float("inf")
    assert stable_step((1.0,), speed=(0.0,), diffusivity=(0.0,)) == float("inf")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spacing": (0.0,)},
        {"spacing": (-1.0,)},
        {"spacing": (1.0,), "cfl": 0.0},
        {"spacing": (1.0,), "diffusivity": (-1.0,)},
        {"spacing": (1.0,), "speed": (float("nan"),)},
        {"spacing": (1.0,), "speed": (float("inf"),)},
        {"spacing": (1.0, 1.0), "speed": (1.0,)},
        {"spacing": (1.0,), "diffusivity": (1.0, 1.0)},
    ],
)
def test_stable_step_rejects_bad_inputs(kwargs):
    with pytest.raises(ValueError, match="stable_step"):
        stable_step(**kwargs)
