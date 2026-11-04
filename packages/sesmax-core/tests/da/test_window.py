"""Tests for ``WindowOperator`` (a control through a model run to the observations)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


pytest.importorskip("vardax")

import equinox as eqx
import lineax as lx
import optax
import vardax

from sesmax_core import BasisForcing, HatInTime, Model, SpatialBasis
from sesmax_core.da import WindowOperator
from sesmax_core.testing import L96State, Lorenz96


KNOTS = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
CELLS = 12
PSD = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})


def _uniform_forcing(coeffs, size):
    """The same value on every cell, which is the hat series in time."""
    spatial = SpatialBasis.from_array(jnp.ones((size, 5)), std=jnp.ones(5))
    return BasisForcing(
        coeffs=coeffs,
        spatial=spatial,
        temporal=HatInTime(knots=KNOTS),
        grid_shape=(size,),
    )


class ForcedDecay(Model):
    """dx/dt = -r x + f(t), linear in the forcing."""

    forcing: BasisForcing
    rate: float = eqx.field(static=True)

    def vector_field(self, t, state, args=None):
        return -self.rate * state + self.forcing(t)

    def apply_boundary_conditions(self, state):
        return state


class ForcedLorenz96(Lorenz96):
    """Lorenz-96 with a time-varying addition to its constant forcing."""

    forcing: BasisForcing = eqx.field(kw_only=True)

    def vector_field(self, t, state, args=None):
        free = super().vector_field(t, state, args)
        return L96State(x=free.x + self.forcing(t))


def _decay_window(rate, save_times, *, size=2, dt=0.05):
    def build_model(control):
        return ForcedDecay(forcing=_uniform_forcing(control, size), rate=rate)

    return WindowOperator(
        build_model=build_model,
        initial_state=jnp.ones(size),
        obs_op=lambda state: state,
        save_times=save_times,
        dt=dt,
    )


def _lorenz_window(start=None, *, length=2.0):
    base = Lorenz96.create(F=8.0)
    if start is None:
        start = L96State(x=8.0 + jnp.sin(jnp.arange(CELLS, dtype=float)))
    knots = jnp.linspace(0.0, length, 5)

    def build_model(control):
        spatial = SpatialBasis.from_array(jnp.ones((CELLS, 5)), std=jnp.ones(5))
        forcing = BasisForcing(
            coeffs=control,
            spatial=spatial,
            temporal=HatInTime(knots=knots),
            grid_shape=(CELLS,),
        )
        return ForcedLorenz96(
            params=base.params, advection=base.advection, forcing=forcing
        )

    return WindowOperator(
        build_model=build_model,
        initial_state=start,
        obs_op=lambda state: state.x[::2],
        save_times=jnp.linspace(length / 8, length, 8),
        dt=0.05,
    )


def test_constant_control_matches_the_closed_form():
    """x(t) = f / r + (x(0) - f / r) exp(-r t) for a constant forcing."""
    rate = 0.7
    value = 3.0
    save_times = jnp.array([0.5, 1.0, 2.0])

    predicted = _decay_window(rate, save_times)(value * jnp.ones(5))

    steady = value / rate
    exact = steady + (1.0 - steady) * jnp.exp(-rate * save_times)
    np.testing.assert_allclose(predicted, jnp.repeat(exact, 2), rtol=1e-5)


def test_observations_stack_by_time_and_then_by_observation():
    """Each time's observations stay together in the flat vector."""
    save_times = jnp.array([0.5, 1.0, 2.0])
    window = _decay_window(0.7, save_times, size=2)

    predicted = window(jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]))

    assert predicted.shape == (6,)
    np.testing.assert_allclose(predicted[0::2], predicted[1::2], rtol=1e-6)


def test_the_forcing_sees_the_right_clock():
    """A hat series integrates to its trapezoid, which a fixed clock misses."""
    control = jnp.array([0.0, 1.0, 3.0, 2.0, 0.0])
    window = _decay_window(0.0, KNOTS[1:], size=1, dt=0.05)

    predicted = window(control)

    # x(t_k) = x(0) + the trapezoid rule on the knots up to t_k
    trapezoids = 0.5 * (control[:-1] + control[1:]) * 0.5
    np.testing.assert_allclose(predicted, 1.0 + jnp.cumsum(trapezoids), rtol=1e-5)

    # Each substep from the same start time, as VardaxForwardModel has to do.
    model = window.build_model(control)
    state = jnp.ones(1)
    for _ in range(4):
        state = model.step(state, 0.5, t0=0.0)
    assert abs(float(state[0]) - float(predicted[-1])) > 0.5


@pytest.mark.slow
def test_gradient_matches_central_differences():
    """The adjoint through the nonlinear run agrees with finite differences."""
    with jax.enable_x64(True):
        window = _lorenz_window()
        control = jnp.array([0.0, 0.5, -0.5, 0.25, 0.0])
        direction = jnp.array([1.0, -1.0, 0.5, 0.5, -1.0])

        def total(candidate):
            return jnp.sum(window(candidate))

        gradient = eqx.filter_grad(total)(control)

        step = 1e-5
        above = total(control + step * direction)
        below = total(control - step * direction)
        central = (above - below) / (2.0 * step)
        np.testing.assert_allclose(jnp.dot(gradient, direction), central, rtol=1e-6)


@pytest.mark.slow
def test_obs_op_works_on_a_row_and_forms_the_jacobian():
    """vardax's leading row goes in and out, and ``linearize`` is the Jacobian."""
    save_times = jnp.array([0.5, 1.0])
    window = _decay_window(0.7, save_times)
    control = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])

    operator = window.as_obs_op()

    np.testing.assert_array_equal(operator(control[None])[0], window(control))
    jacobian = operator.linearize(control[None]).as_matrix()
    np.testing.assert_allclose(jacobian, jax.jacrev(window)(control), rtol=1e-6)


@pytest.mark.slow
def test_three_d_var_on_a_linear_model_matches_the_blue():
    """On a linear model, the 3D-Var minimum is the closed-form BLUE."""
    window = _decay_window(0.7, jnp.array([0.5, 1.0, 1.5, 2.0]))
    truth = jnp.array([0.0, 2.0, -1.0, 1.5, 0.5])
    observations = window(truth)
    size = observations.size
    noise = 0.25

    three = vardax.ThreeDVar(
        obs_op=window.as_obs_op(),
        prior_mean=jnp.zeros((1, 5)),
        prior_cov_op=lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct((1, 5), observations.dtype)
        ),
        obs_cov_op=lx.FunctionLinearOperator(
            lambda vector: noise * vector,
            jax.ShapeDtypeStruct((1, size), observations.dtype),
            PSD,
        ),
        cov_solver=lx.AutoLinearSolver(well_posed=True),
    )
    batch = vardax.Batch1D(input=observations[None, None], mask=jnp.ones((1, 1, size)))
    analysis = three(batch)[0, 0]

    # theta* = theta_0 + (I + Jᵀ J / r)⁻¹ Jᵀ (y - G(theta_0)) / r
    jacobian = jax.jacrev(window)(jnp.zeros(5))
    residual = observations - window(jnp.zeros(5))
    precision = jnp.eye(5) + jacobian.T @ jacobian / noise
    blue = jnp.linalg.solve(precision, jacobian.T @ residual / noise)
    np.testing.assert_allclose(analysis, blue, atol=2e-3)


@pytest.mark.slow
def test_twin_experiment_recovers_a_time_varying_forcing():
    """From a start on the attractor, the analysis beats the prior and fits the data."""
    base = Lorenz96.create(F=8.0)
    start = L96State(x=(8.0 * jnp.ones(CELLS)).at[0].add(0.01))
    advance = eqx.filter_jit(lambda state: base.step(state, 0.05))
    for _ in range(200):
        start = advance(start)
    window = _lorenz_window(start, length=0.5)
    truth = jnp.array([0.0, 2.0, -1.0, 1.5, 0.5])
    noise_std = 0.05
    clean = window(truth)
    noise = noise_std * jax.random.normal(jax.random.key(0), clean.shape)
    observations = clean + noise
    size = observations.size

    three = vardax.ThreeDVar(
        obs_op=window.as_obs_op(),
        prior_mean=jnp.zeros((1, 5)),
        prior_cov_op=lx.IdentityLinearOperator(
            jax.ShapeDtypeStruct((1, 5), observations.dtype)
        ),
        obs_cov_op=lx.FunctionLinearOperator(
            lambda vector: noise_std**2 * vector,
            jax.ShapeDtypeStruct((1, size), observations.dtype),
            PSD,
        ),
        # A quasi-Newton first step along this gradient blows the model up.
        minimiser=vardax.OptaxMinimiser(optax.adam(0.05), max_steps=300),
        cov_solver=lx.AutoLinearSolver(well_posed=True),
    )
    batch = vardax.Batch1D(input=observations[None, None], mask=jnp.ones((1, 1, size)))
    analysis = three(batch)[0, 0]

    def cost(control):
        misfit = (observations - window(control)) / noise_std
        return 0.5 * jnp.sum(control**2) + 0.5 * jnp.sum(misfit**2)

    prior_error = jnp.sqrt(jnp.mean(truth**2))
    analysis_error = jnp.sqrt(jnp.mean((analysis - truth) ** 2))
    assert analysis_error < 0.7 * prior_error
    # The solve reached the truth's cost, so the rest of the error isn't the solver's.
    assert cost(analysis) < 1.1 * cost(truth)
