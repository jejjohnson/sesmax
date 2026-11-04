"""Tests for the Model base class."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from sesmax_core import Model, TermFn, TermModel, implicit, positive


@runtime_checkable
class _ModelStepLike(Protocol):
    """The model-supplied half of ``pipekit_cycle.ForwardModel``.

    Defined here so the structural-conformance test does not pull in a
    dependency on pipekit. A bare ``Model`` provides ``step`` and
    ``state_signature``; the third ``ForwardModel`` member (a default
    ``dt``) is supplied by the ``sesmax_core.operators`` adapter, which is
    tested against the real protocol in ``tests/test_operators.py``.
    """

    def step(self, state: Any, dt: float) -> Any: ...

    @property
    def state_signature(self) -> Any: ...


class ExponentialDecay(Model):
    """Minimal concrete model: dx/dt = -x."""

    rate: float = 1.0

    def vector_field(self, t, state, args=None):
        return -self.rate * state

    def apply_boundary_conditions(self, state):
        return state


class MissingBCs(Model):
    """Model that only implements vector_field."""

    def vector_field(self, t, state, args=None):
        return -state


def test_cannot_instantiate_abstract():
    """Model cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Model()


def test_concrete_model_instantiation():
    model = ExponentialDecay(rate=2.0)
    assert model.rate == 2.0


def test_vector_field():
    model = ExponentialDecay(rate=1.0)
    state = jnp.array([1.0, 2.0, 3.0])
    rhs = model.vector_field(0.0, state)
    assert jnp.allclose(rhs, -state)


def test_build_terms_returns_ode_term():
    model = ExponentialDecay()
    terms = model.build_terms()
    assert isinstance(terms, dfx.ODETerm)


def test_integrate_produces_solution():
    model = ExponentialDecay(rate=1.0)
    state0 = jnp.array([1.0])
    sol = model.integrate(state0, t0=0.0, t1=1.0, dt=0.01)
    assert isinstance(sol, dfx.Solution)
    # After t=1 with rate=1, x ≈ e^{-1} ≈ 0.368
    assert jnp.allclose(sol.ys[-1], jnp.exp(-1.0), atol=1e-3)


def test_integrate_with_saveat():
    model = ExponentialDecay(rate=1.0)
    state0 = jnp.array([1.0])
    sol = model.integrate(
        state0,
        t0=0.0,
        t1=1.0,
        dt=0.01,
        saveat=dfx.SaveAt(ts=jnp.array([0.0, 0.5, 1.0])),
    )
    assert sol.ys.shape == (3, 1)


def test_diagnose_returns_empty_by_default():
    model = ExponentialDecay()
    state = jnp.array([1.0])
    diag = model.diagnose(state)
    assert diag == {}


def test_missing_apply_bcs_raises():
    """Model missing apply_boundary_conditions cannot be instantiated."""
    with pytest.raises(TypeError):
        MissingBCs()


def test_step_returns_bare_state():
    """``step`` returns a bare state matching integrate's final state."""
    model = ExponentialDecay(rate=1.0)
    state0 = jnp.array([1.0, 2.0])

    stepped = model.step(state0, dt=0.01)

    sol = model.integrate(state0, t0=0.0, t1=0.01, dt=0.01)
    final = jtu.tree_map(lambda x: x[-1], sol.ys)

    # same pytree structure / shape as the input state (no leading time axis)
    assert jtu.tree_structure(stepped) == jtu.tree_structure(state0)
    assert stepped.shape == state0.shape
    assert jnp.allclose(stepped, final)


def test_step_composes_to_integration_window():
    """Repeated ``step`` calls match the analytic decay over the window."""
    model = ExponentialDecay(rate=1.0)
    state = jnp.array([1.0, -2.0, 0.5])

    dt = 0.05
    out = state
    for i in range(10):
        out = model.step(out, dt=dt, t0=i * dt)

    # dx/dt = -x integrated over 0.5 -> x * e^{-0.5}
    expected = state * jnp.exp(-0.5)
    assert jnp.allclose(out, expected, rtol=1e-3)


def test_model_provides_forward_model_step_and_signature():
    """A somax model provides the ``step`` + ``state_signature`` members.

    This is the pipekit-cycle / DA integration seam: any ``Model``
    supplies the model half of the ``ForwardModel`` contract purely by
    duck typing, with no subclassing. The ``dt`` member is added by the
    ``sesmax_core.operators`` adapter (see ``tests/test_operators.py``).
    """
    model = ExponentialDecay(rate=1.0)
    assert isinstance(model, _ModelStepLike)
    assert model.state_signature is None


def test_forward_model_step_matches_model_step_exactly():
    """The forward model runs the same solve as ``Model.step``."""
    model = ExponentialDecay(rate=0.7)
    state = jnp.array([1.0, -2.0, 3.0])

    final = model.as_forward_model(dt=0.5, dt0=0.5).step(state, 0.5)

    np.testing.assert_array_equal(final, model.step(state, 0.5))


def test_forward_model_substeps_match_integrate():
    """``dt0`` sets the integrator step, separately from the window."""
    model = ExponentialDecay(rate=0.7)
    state = jnp.array([1.0, -2.0, 3.0])

    final = model.as_forward_model(dt=0.5, dt0=0.05).step(state, 0.5)
    expected = model.integrate(state, 0.0, 0.5, 0.05).ys[-1]

    # Ten steps in single precision round differently by about 1e-7.
    np.testing.assert_allclose(final, expected, rtol=1e-6)


class ClampedDecay(ExponentialDecay):
    """Decay with the first entry held at zero by its boundary condition."""

    def apply_boundary_conditions(self, state):
        return state.at[0].set(0.0)


def test_forward_model_fills_boundaries_in_the_right_hand_side():
    """Each evaluation sees filled boundaries, but the input isn't filled."""
    model = ClampedDecay(rate=0.7)
    state = jnp.array([1.0, -2.0, 3.0])

    final = model.as_forward_model(dt=0.5, dt0=0.5).step(state, 0.5)

    np.testing.assert_array_equal(final[1:], model.step(state, 0.5)[1:])
    assert final[0] == state[0]


def test_forward_model_on_a_filled_state_matches_model_step_exactly():
    """Filling the input first recovers ``Model.step`` everywhere."""
    model = ClampedDecay(rate=0.7)
    state = jnp.array([1.0, -2.0, 3.0])
    filled = model.apply_boundary_conditions(state)

    final = model.as_forward_model(dt=0.5, dt0=0.5).step(filled, 0.5)

    np.testing.assert_array_equal(final, model.step(state, 0.5))


def test_forward_model_unwraps_constrained_parameters():
    """A ``positive`` parameter gives what the plain value gives."""
    state = jnp.array([1.0, -2.0, 3.0])
    plain = ExponentialDecay(rate=jnp.asarray(0.7))
    wrapped = ExponentialDecay(rate=positive(0.7))

    final = wrapped.as_forward_model(dt=0.5, dt0=0.05).step(state, 0.5)
    expected = plain.as_forward_model(dt=0.5, dt0=0.05).step(state, 0.5)

    np.testing.assert_allclose(final, expected, rtol=1e-6)


def test_forward_model_satisfies_the_pipekit_contract():
    """It is a real ``pipekit_cycle.ForwardModel``, with its own ``dt``."""
    from pipekit_cycle import ForwardModel

    forward = ExponentialDecay(rate=0.7).as_forward_model(dt=0.5, dt0=0.05)

    assert isinstance(forward, ForwardModel)
    assert forward.dt == 0.5


def test_forward_model_dynamics_integrates_between_times():
    """``as_dynamics`` carries the absolute time, which ``step`` does not."""

    class Ramp(Model):
        def vector_field(self, t, state, args=None):
            return t * jnp.ones_like(state)

        def apply_boundary_conditions(self, state):
            return state

    model = Ramp()
    state = jnp.array([1.0, -2.0, 3.0])
    forward = model.as_forward_model(dt=0.5, dt0=0.5)

    final = forward.as_dynamics()(state, 1.0, 1.5)

    np.testing.assert_array_equal(final, model.step(state, 0.5, t0=1.0))
    # x(1.5) = x(1) + (1.5² - 1²) / 2
    np.testing.assert_allclose(final, state + 0.625, rtol=1e-6)


def test_forward_model_reverse_gradient_reaches_parameters():
    """The model is a leaf of the forward model, so a filtered grad finds it."""
    state = jnp.array([1.0, -2.0, 3.0])
    model = ExponentialDecay(rate=jnp.asarray(0.7))

    def loss(candidate):
        forward = candidate.as_forward_model(dt=0.5, dt0=0.05)
        return jnp.sum(forward.step(state, 0.5) ** 2)

    gradient = eqx.filter_grad(loss)(model).rate

    # d/dk Σ x² e^(-2 k t) = -2 t Σ x² e^(-2 k t), up to the solver's error
    exact = -2.0 * 0.5 * jnp.sum(state**2) * jnp.exp(-2.0 * 0.7 * 0.5)
    np.testing.assert_allclose(gradient, exact, rtol=1e-3)


def test_forward_model_forward_mode_needs_direct_adjoint():
    """``jax.jvp`` works once the adjoint is ``DirectAdjoint``."""
    model = ExponentialDecay(rate=0.7)
    state = jnp.array([1.0, -2.0, 3.0])
    forward = model.as_forward_model(dt=0.5, dt0=0.05, adjoint=dfx.DirectAdjoint())

    final, tangent = jax.jvp(lambda s: forward.step(s, 0.5), (state,), (state,))

    # The problem is linear, so the tangent along the state is the state's image.
    np.testing.assert_allclose(tangent, final, rtol=1e-6)


def test_term_model_forward_model_rejects_mixed_terms():
    """One vector field can't hold an IMEX split, so it refuses one."""
    decay = TermFn(lambda t, state, args=None: -state)
    model = TermModel(terms=decay + implicit(decay))

    with pytest.raises(ValueError, match="mixes explicit and implicit"):
        model.as_forward_model(dt=0.5, dt0=0.05)


def test_term_model_forward_model_accepts_a_single_kind():
    """An all-explicit tree runs as ``Model.step`` does."""
    decay = TermFn(lambda t, state, args=None: -0.7 * state)
    model = TermModel(terms=decay)
    state = jnp.array([1.0, -2.0, 3.0])

    final = model.as_forward_model(dt=0.5, dt0=0.5).step(state, 0.5)

    np.testing.assert_array_equal(final, model.step(state, 0.5))
