"""The nonlinear models against the exact solution before a wave breaks.

Without viscosity, ``du/dt + u du/dx = 0`` carries each value along a
characteristic at speed ``u``, so ``u = u_0(x - u t)``. The initial wave here
rides on a background speed, so it crosses the periodic edge during the run,
which also checks the boundary. The comparison with the solution at ``2 t``
guards the rate: these models once used a flux of ``u^2``, and then matched
that solution instead.
"""

from __future__ import annotations

import diffrax as dfx
import jax.numpy as jnp
import pytest

import sesmax_core.testing as sxt


BACKGROUND = 0.5
AMPLITUDE = 0.5
# Halfway to breaking, at t_b = 1 / (2 pi A) = 0.318 in 1D.
END_TIME_1D = 0.15
# u = u_0(x + y - 2 u t) along the diagonal, which breaks twice as fast.
END_TIME_DIAGONAL = 0.075
# First-order upwinding on these grids, measured at 0.0009 (1D) and 0.0068 (2D).
TOLERANCE = 0.01
# The solution at 2 t differs from the solution at t by about 0.2 in RMS.
WRONG_RATE_FLOOR = 0.05


def _characteristic(phase, speed_factor, time):
    """Solve u = U + A sin(2 pi (phase - speed_factor u t)) by Newton."""
    solution = BACKGROUND + AMPLITUDE * jnp.sin(2.0 * jnp.pi * phase)
    for _ in range(30):
        argument = 2.0 * jnp.pi * (phase - speed_factor * solution * time)
        residual = solution - BACKGROUND - AMPLITUDE * jnp.sin(argument)
        slope = 1.0 + 2.0 * jnp.pi * AMPLITUDE * speed_factor * time * jnp.cos(argument)
        solution = solution - residual / slope
    return solution


def _rms(difference):
    return float(jnp.sqrt(jnp.mean(difference**2)))


def _wave(position):
    return BACKGROUND + AMPLITUDE * jnp.sin(2.0 * jnp.pi * position)


MODELS_1D = [
    pytest.param(
        lambda: sxt.NonlinearConvection1D.create(nx=512, Lx=1.0),
        sxt.NonlinearConvection1DState,
        id="nonlinear_convection",
    ),
    pytest.param(
        lambda: sxt.Burgers1D.create(nx=512, Lx=1.0, nu=1e-6),
        sxt.Burgers1DState,
        id="burgers",
    ),
]
MODELS_2D = [
    pytest.param(
        lambda: sxt.NonlinearConvection2D.create(nx=128, ny=128, Lx=1.0, Ly=1.0),
        sxt.NonlinearConvection2DState,
        id="nonlinear_convection",
    ),
    pytest.param(
        lambda: sxt.Burgers2D.create(nx=128, ny=128, Lx=1.0, Ly=1.0, nu=1e-6),
        sxt.Burgers2DState,
        id="burgers",
    ),
    pytest.param(
        lambda: sxt.Burgers2DTermModel.create(nx=128, ny=128, Lx=1.0, Ly=1.0, nu=1e-6),
        sxt.Burgers2DState,
        id="burgers_terms",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(("make_model", "state_type"), MODELS_1D)
def test_1d_matches_characteristics(make_model, state_type):
    model = make_model()
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    state = model.apply_boundary_conditions(state_type(u=_wave(x)))
    solution = model.integrate(
        state, t0=0.0, t1=END_TIME_1D, dt=2.5e-4, saveat=dfx.SaveAt(t1=True)
    )
    u = solution.ys.u[0, 1:-1]
    assert _rms(u - _characteristic(x[1:-1], 1.0, END_TIME_1D)) < TOLERANCE
    assert _rms(u - _characteristic(x[1:-1], 1.0, 2 * END_TIME_1D)) > WRONG_RATE_FLOOR


@pytest.mark.slow
@pytest.mark.parametrize(("make_model", "state_type"), MODELS_2D)
def test_2d_uniform_in_y_matches_characteristics(make_model, state_type):
    model = make_model()
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    x_grid = jnp.tile(x, (model.grid.Ny, 1))
    state = model.apply_boundary_conditions(
        state_type(u=_wave(x_grid), v=jnp.zeros_like(x_grid))
    )
    solution = model.integrate(
        state, t0=0.0, t1=END_TIME_1D, dt=5e-4, saveat=dfx.SaveAt(t1=True)
    )
    u = solution.ys.u[0, 1:-1, 1:-1]
    assert _rms(u[3] - _characteristic(x[1:-1], 1.0, END_TIME_1D)) < TOLERANCE
    assert (
        _rms(u[3] - _characteristic(x[1:-1], 1.0, 2 * END_TIME_1D)) > WRONG_RATE_FLOOR
    )


@pytest.mark.slow
@pytest.mark.parametrize(("make_model", "state_type"), MODELS_2D)
def test_2d_diagonal_flow_matches_characteristics(make_model, state_type):
    """With u = v = f(x + y) the flow has divergence, which the form must handle."""
    model = make_model()
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    y = jnp.arange(model.grid.Ny) * model.grid.dy
    x_grid, y_grid = jnp.meshgrid(x, y)
    phase = x_grid + y_grid
    state = model.apply_boundary_conditions(state_type(u=_wave(phase), v=_wave(phase)))
    solution = model.integrate(
        state, t0=0.0, t1=END_TIME_DIAGONAL, dt=2.5e-4, saveat=dfx.SaveAt(t1=True)
    )
    u = solution.ys.u[0, 1:-1, 1:-1]
    exact = _characteristic(phase[1:-1, 1:-1], 2.0, END_TIME_DIAGONAL)
    assert _rms(u - exact) < TOLERANCE


@pytest.mark.slow
@pytest.mark.parametrize(("make_model", "state_type"), MODELS_1D)
def test_1d_flux_form_conserves_across_the_periodic_edge(make_model, state_type):
    model = make_model()
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    state = model.apply_boundary_conditions(state_type(u=_wave(x)))
    solution = model.integrate(
        state, t0=0.0, t1=END_TIME_1D, dt=2.5e-4, saveat=dfx.SaveAt(t1=True)
    )
    total_change = jnp.sum(solution.ys.u[0, 1:-1] - state.u[1:-1]) * model.grid.dx
    # Single-precision round-off over the run.
    assert jnp.abs(total_change) < 1e-5


@pytest.mark.slow
def test_shallow_water_conserves_mass_across_the_periodic_edge():
    """A wave crossing the edge keeps its mass, instead of hitting a closed wall."""
    model = sxt.NonlinearShallowWater1D.create(
        nx=500, Lx=1e6, g=9.81, f0=0.0, H0=100.0, lateral_viscosity=50.0
    )
    x = jnp.arange(model.grid.Nx) * model.grid.dx
    height = 100.0 + 2.0 * jnp.exp(-0.5 * ((x - 5e5) / (1e6 / 15.0)) ** 2)
    state = sxt.NonlinearSW1DState(
        h=height, u=jnp.zeros_like(height), v=jnp.zeros_like(height)
    )
    solution = model.integrate(
        state,
        t0=0.0,
        t1=4e4,
        dt=0.3 * model.grid.dx / 31.3,
        saveat=dfx.SaveAt(ts=jnp.array([2e4, 4e4])),
    )
    bump_mass = jnp.sum(height[1:-1] - 100.0)
    mass_change = jnp.sum(solution.ys.h[-1, 1:-1] - height[1:-1])
    # Single precision on a 100 m layer leaves about 1e-3 of the bump's mass.
    assert jnp.abs(mass_change / bump_mass) < 1e-2
    # The wave has reached the edge by t = 2e4 s, so the edge is not pinned there.
    assert jnp.abs(solution.ys.h[0, 1] - 100.0) > 0.1
