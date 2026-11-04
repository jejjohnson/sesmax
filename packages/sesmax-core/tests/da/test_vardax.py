"""Strong-constraint 4DVar driving a somax model through vardax.

End-to-end check that the ``sesmax_core.da`` vardax bridge works: a somax model,
wrapped as a flat-vector ``ForwardModel`` via :class:`VardaxForwardModel`, can
be assimilated by ``vardax.StrongFourDVar`` with `gaussx`-built background and
observation covariances. The analysis must improve on the background.

The hard assertion uses the *non-advective* (linear) Lorenz-96 system: strong
4DVar over the fully chaotic model is genuinely fragile inside the outer
minimiser (a long rollout from a poor ``x_0`` can blow up), so the chaotic
case is reserved for the docs tutorial under a fixed, vetted configuration.
Requires the ``da`` dependency group; skipped otherwise.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


pytest.importorskip("vardax")

import diffrax as dfx
import gaussx
import lineax as lx
from pipekit import Operator
from pipekit_cycle import DAState
from vardax import (
    Batch1D,
    IncrementalFourDVar,
    MaskedIdentity,
    StrongFourDVar,
    VarDACycle,
)

from sesmax_core._src.testing.lorenz96 import L96State, Lorenz96
from sesmax_core.da import VardaxForwardModel, state_to_vector


def _structured_background(key, n: int) -> lx.AbstractLinearOperator:
    """Build a `gaussx` diag + low-rank background covariance B.

    Returns a dense PSD-tagged `lineax` operator: gaussx constructs the
    structured ``LowRankUpdate``; we materialise it so vardax's internal
    ``lx.CG`` solves (which need ``linearise``) accept it.
    """
    psd = lx.positive_semidefinite_tag
    u = jax.random.normal(key, (n, 2)) * 0.3
    base = lx.DiagonalLinearOperator(0.5 * jnp.ones(n))
    b_struct = gaussx.LowRankUpdate(base, u, jnp.ones(2), tags=psd)
    return lx.MatrixLinearOperator(b_struct.as_matrix(), psd)


def _twin_4dvar(advection: bool, n_steps: int, bg_std: float):
    """Run a strong-4DVar twin experiment; return (background, analysis, truth)."""
    key = jax.random.PRNGKey(0)
    K = 12
    dt = 0.05
    obs_var = 0.04

    model = Lorenz96.create(F=8.0, advection=advection)
    step = jax.jit(lambda s: model.step(s, dt))

    # Spin the truth onto the model's attractor / equilibrium.
    truth = L96State(x=8.0 * jnp.ones(K).at[0].add(0.01))
    for _ in range(200):
        truth = step(truth)
    x0_true, _ = state_to_vector(truth)

    forward = VardaxForwardModel(model=model, template=truth, dt=dt)

    # Truth trajectory over the window (T + 1 timesteps) + noisy observations.
    traj = [x0_true]
    state = truth
    for _ in range(n_steps):
        state = step(state)
        vec, _ = state_to_vector(state)
        traj.append(vec)
    traj = jnp.stack(traj)  # (T+1, K)

    noise = jnp.sqrt(obs_var) * jax.random.normal(key, traj.shape)
    batch = Batch1D(
        input=(traj + noise)[None],
        mask=jnp.ones_like(traj)[None],
        target=traj[None],
    )

    # Background = truth perturbed; structured B from gaussx, diagonal R.
    background = x0_true + bg_std * jax.random.normal(jax.random.PRNGKey(1), (K,))
    psd = lx.positive_semidefinite_tag
    B = _structured_background(jax.random.PRNGKey(2), K)
    R = lx.MatrixLinearOperator(jnp.diag(obs_var * jnp.ones(K)), psd)

    strong = StrongFourDVar(
        forward=forward,
        obs_op=MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=B,
        obs_cov_op=R,
    )
    analysis = strong(batch)[0]
    return background, analysis, x0_true


@pytest.mark.slow
def test_strong_4dvar_improves_on_background():
    """4DVar analysis is closer to truth than the background (linear L96)."""
    background, analysis, truth = _twin_4dvar(advection=False, n_steps=5, bg_std=1.0)
    bg_rmse = jnp.sqrt(jnp.mean((background - truth) ** 2))
    an_rmse = jnp.sqrt(jnp.mean((analysis - truth) ** 2))

    assert jnp.all(jnp.isfinite(analysis))
    assert an_rmse < bg_rmse


def test_somax_forward_model_matches_step():
    """The flat-vector ForwardModel adapter equals a direct pytree ``step``."""
    model = Lorenz96.create(F=8.0)
    state = L96State(x=jnp.linspace(-2.0, 2.0, 12))
    forward = VardaxForwardModel(model=model, template=state, dt=0.05)

    vec, _ = state_to_vector(state)
    got = forward.step(vec, 0.05)
    expected, _ = state_to_vector(model.step(state, 0.05))
    assert jnp.allclose(got, expected, atol=1e-6)


def test_somax_forward_model_is_forward_model():
    """The adapter satisfies the pipekit_cycle ForwardModel surface."""
    from pipekit_cycle import ForwardModel

    model = Lorenz96.create(F=8.0)
    state = L96State(x=jnp.zeros(12))
    forward = VardaxForwardModel(model=model, template=state, dt=0.05)

    assert isinstance(forward, ForwardModel)
    assert forward.dt == 0.05
    assert forward.state_signature is None


def _window(model, *, n_variables, n_steps, background_std, dt=0.05, obs_var=0.04):
    """A truth, a noisy observation window and a background, for one twin run."""
    step = jax.jit(lambda state: model.step(state, dt))
    truth = L96State(x=8.0 * jnp.ones(n_variables).at[0].add(0.01))
    for _ in range(200):
        truth = step(truth)
    initial_truth, _ = state_to_vector(truth)
    trajectory = [initial_truth]
    state = truth
    for _ in range(n_steps):
        state = step(state)
        trajectory.append(state_to_vector(state)[0])
    trajectory = jnp.stack(trajectory)
    noise = jnp.sqrt(obs_var) * jax.random.normal(jax.random.key(0), trajectory.shape)
    background = initial_truth + background_std * jax.random.normal(
        jax.random.key(1), (n_variables,)
    )
    psd = lx.positive_semidefinite_tag
    background_cov = lx.MatrixLinearOperator(0.5 * jnp.eye(n_variables), psd)
    obs_cov = lx.MatrixLinearOperator(obs_var * jnp.eye(n_variables), psd)
    return truth, initial_truth, trajectory + noise, background, background_cov, obs_cov


def _rmse(estimate, truth):
    return float(jnp.sqrt(jnp.mean((estimate - truth) ** 2)))


@pytest.mark.slow
def test_incremental_4dvar_improves_on_background_on_the_chaotic_model():
    """Incremental 4D-Var differentiates forward, so it needs ``adjoint``.

    The chaotic model on a three-step window: this route succeeded on each of
    six truths when it was checked, where ``StrongFourDVar`` succeeded on four.
    """
    model = Lorenz96.create(F=8.0)
    truth, initial_truth, observed, background, background_cov, obs_cov = _window(
        model, n_variables=12, n_steps=3, background_std=0.3
    )
    forward = VardaxForwardModel(
        model=model, template=truth, dt=0.05, adjoint=dfx.DirectAdjoint()
    )
    incremental = IncrementalFourDVar(
        forward=forward,
        obs_op=MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=background_cov,
        obs_cov_op=obs_cov,
    )
    batch = Batch1D(
        input=observed[None], mask=jnp.ones_like(observed)[None], target=None
    )
    analysis = incremental(batch)[0]
    assert _rmse(analysis, initial_truth) < _rmse(background, initial_truth)


def test_default_adjoint_supports_reverse_mode_only():
    """Without ``adjoint`` a forward-mode solver fails, with diffrax's own error."""
    model = Lorenz96.create(F=8.0)
    state = L96State(x=jnp.linspace(-2.0, 2.0, 12))
    forward = VardaxForwardModel(model=model, template=state, dt=0.05)
    vector, _ = state_to_vector(state)
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(lambda x: forward.step(x, 0.05), (vector,), (vector,))

    direct = VardaxForwardModel(
        model=model, template=state, dt=0.05, adjoint=dfx.DirectAdjoint()
    )
    _, tangent = jax.jvp(lambda x: direct.step(x, 0.05), (vector,), (vector,))
    assert jnp.all(jnp.isfinite(tangent))


@pytest.mark.slow
def test_vardax_cycle_runs_a_window_through_the_adapter():
    """One forecast and analysis of ``VarDACycle`` improves on the background."""
    model = Lorenz96.create(F=8.0, advection=False)
    truth, initial_truth, observed, background, background_cov, obs_cov = _window(
        model, n_variables=12, n_steps=3, background_std=1.0
    )
    forward = VardaxForwardModel(model=model, template=truth, dt=0.05)
    strong = StrongFourDVar(
        forward=forward,
        obs_op=MaskedIdentity(),
        prior_mean=background,
        prior_cov_op=background_cov,
        obs_cov_op=obs_cov,
    )

    class WindowSource(Operator):
        def _apply(self, index):
            return observed[None]

        def get_config(self):
            return {}

    cycle = VarDACycle(
        forward=forward,
        obs_op=MaskedIdentity(),
        model=strong,
        obs_source=WindowSource(),
        n_steps=1,
    )
    analysis, state = cycle(background, DAState(obs_err_cov=obs_cov))
    assert state.cycle_count == 1
    assert _rmse(analysis[0], initial_truth) < _rmse(background, initial_truth)
