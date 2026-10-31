"""Tests for the training rollouts on a model defined here.

somax's own suite (``tests/test_train.py`` at 9cb19b3) runs on the linear
shallow-water term model, which moves into ``sesmax_core.testing`` later
and brings that suite with it. These tests check the same contracts now on
a linear decay with a closed-form trajectory.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, PyTree

from sesmax_core import Model, Params, State
from sesmax_core.train import ModelStep, rollout, rollout_loss, state_mse


DT = 0.1


class Scalar(State):
    x: Array


class DecayParams(Params):
    rate: Array


class Decay(Model):
    """dx/dt = -r x."""

    params: DecayParams

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        return Scalar(x=-self.params.rate * state.x)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        return state


@pytest.fixture
def model() -> Decay:
    return Decay(params=DecayParams(rate=jnp.asarray(0.5)))


@pytest.fixture
def state0() -> Scalar:
    return Scalar(x=jnp.array([1.0, -2.0, 0.5]))


def test_rollout_matches_a_python_loop(model, state0):
    traj = rollout(model, state0, dt=DT, n_steps=5)
    state = state0
    for k in range(5):
        state = model.step(state, DT)
        np.testing.assert_allclose(traj.x[k], state.x, rtol=1e-6)


def test_rollout_follows_the_closed_form(model, state0):
    traj = rollout(model, state0, dt=DT, n_steps=10)
    expected = np.exp(-0.5 * DT * 10) * np.asarray(state0.x)
    # one Tsit5 step per DT: float32, 5th order in a smooth decay
    np.testing.assert_allclose(traj.x[-1], expected, rtol=1e-5)


def test_rollout_can_include_the_initial_state_and_subsample(model, state0):
    traj = rollout(model, state0, dt=DT, n_steps=6, save_every=2, include_initial=True)
    assert traj.x.shape == (4, 3)
    np.testing.assert_allclose(traj.x[0], state0.x)
    full = rollout(model, state0, dt=DT, n_steps=6)
    np.testing.assert_allclose(traj.x[1:], full.x[1::2], rtol=1e-6)


def test_rollout_loss_is_zero_on_its_own_trajectory_and_positive_otherwise(
    model, state0
):
    ref = rollout(model, state0, dt=DT, n_steps=4)
    assert float(rollout_loss(model, state0, ref, dt=DT)) == pytest.approx(0.0)
    other = Decay(params=DecayParams(rate=jnp.asarray(1.0)))
    assert float(rollout_loss(other, state0, ref, dt=DT)) > 0.0


def test_rollout_loss_is_the_mean_snapshot_loss(model, state0):
    ref = jax.tree_util.tree_map(
        jnp.zeros_like, rollout(model, state0, dt=DT, n_steps=4)
    )
    pred = rollout(model, state0, dt=DT, n_steps=4)
    expected = np.mean(
        [float(state_mse(Scalar(x=pred.x[k]), Scalar(x=ref.x[k]))) for k in range(4)]
    )
    got = float(rollout_loss(model, state0, ref, dt=DT))
    assert got == pytest.approx(expected, rel=1e-6)


def test_gradient_of_the_rollout_loss_matches_finite_differences(state0):
    ref = rollout(
        Decay(params=DecayParams(rate=jnp.asarray(0.8))), state0, dt=DT, n_steps=8
    )

    def loss(rate):
        m = Decay(params=DecayParams(rate=rate))
        return rollout_loss(m, state0, ref, dt=DT)

    rate = jnp.asarray(0.5)
    grad = float(jax.grad(loss)(rate))
    eps = 1e-3
    fd = (float(loss(rate + eps)) - float(loss(rate - eps))) / (2 * eps)
    # central difference in float32: agreement to ~1e-3 relative
    assert grad == pytest.approx(fd, rel=1e-2)
    assert grad < 0.0  # the reference decays faster, so raising the rate helps


def test_model_step_is_a_pipekit_operator_over_time_and_state(model, state0):
    step = ModelStep(model=model, dt=DT)
    t1, s1 = step((jnp.asarray(0.0), state0))
    assert float(t1) == pytest.approx(DT)
    np.testing.assert_allclose(s1.x, model.step(state0, DT).x, rtol=1e-6)
    assert eqx.is_array(s1.x)
