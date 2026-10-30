"""Tests for ``ScaledModel`` on small models defined here.

somax's own suite (``tests/test_scaled_model.py`` at 9cb19b3) runs on the
barotropic QG and nonlinear shallow-water models, which are not in the core.
These tests check the same properties on a linear model whose trajectory is
known in closed form, so ``ScaledModel`` is covered before those models move;
somax's suite follows them.
"""

from __future__ import annotations

import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, PyTree

from sesmax_core import Model, Params, State, TermModel
from sesmax_core.scales import ScaledModel, Scales, StateAffine
from sesmax_core.terms import Term, implicit


L = 1.0e5
U = 0.1
T_END = 2.0e4  # seconds of inner time
DT = 50.0  # seconds of inner time
# JAX runs in float32 by default: 400 Tsit5 steps in two coordinate systems
# agree to ~1e-6 relative, so the equivalence checks allow 1e-5, with an
# absolute floor of 1e-5 of the velocity scale for entries near zero.
RTOL = 1e-5
ATOL = RTOL * U


class FlowState(State):
    """Two velocity components, scaled by ``U`` under ``Scales``."""

    u: Array
    v: Array


class DecayParams(Params):
    rate: Array


class LinearDecay(Model):
    """du/dt = -r u, dv/dt = -r v: x(t) = x(0) exp(-r t)."""

    params: DecayParams

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        return jax.tree_util.tree_map(lambda x: -self.params.rate * x, state)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        return state


class Decay(Term):
    rate: float

    def __call__(self, t, state, args=None):
        return jax.tree_util.tree_map(lambda x: -self.rate * x, state)


@pytest.fixture
def scales() -> Scales:
    return Scales.advective(L=L, U=U, f0=1e-4)


@pytest.fixture
def inner() -> LinearDecay:
    return LinearDecay(params=DecayParams(rate=jnp.asarray(1.0e-4)))


@pytest.fixture
def state0() -> FlowState:
    rng = np.random.RandomState(0)
    return FlowState(
        u=jnp.asarray(U * rng.randn(4, 5)), v=jnp.asarray(U * rng.randn(4, 5))
    )


def _final(model: Model, x0: PyTree, t1: float, dt: float) -> PyTree:
    sol = model.integrate(x0, t0=0.0, t1=t1, dt=dt, solver=dfx.Tsit5())
    return jax.tree_util.tree_map(lambda x: x[-1], sol.ys)


class TestConstruction:
    def test_from_scales_uses_the_time_scale(self, inner, scales):
        model = ScaledModel.from_scales(inner, scales, FlowState)
        assert model.time_scale == pytest.approx(scales.T)

    def test_from_scales_builds_the_matching_transform(self, inner, scales):
        model = ScaledModel.from_scales(inner, scales, FlowState)
        expected = StateAffine.from_scales(FlowState, scales)
        assert model.transform.scale.u == pytest.approx(expected.scale.u)
        assert model.transform.scale.u == pytest.approx(U)

    def test_standardised_leaves_time_alone(self, inner, state0):
        rng = np.random.RandomState(1)
        samples = jax.tree_util.tree_map(
            lambda x: jnp.asarray(rng.randn(8, *x.shape) * U), state0
        )
        model = ScaledModel.standardised(inner, samples, per_gridpoint=True)
        assert model.time_scale == 1.0

    def test_is_a_model(self, inner, scales):
        assert isinstance(ScaledModel.from_scales(inner, scales, FlowState), Model)

    def test_state_signature_delegates(self, inner, scales):
        model = ScaledModel.from_scales(inner, scales, FlowState)
        assert model.state_signature == inner.state_signature


class TestEquivalence:
    def test_wrapped_trajectory_maps_back_to_the_inner_one(self, inner, scales, state0):
        model = ScaledModel.from_scales(inner, scales, FlowState)
        x_inner = _final(inner, state0, T_END, DT)
        y_end = _final(
            model, model.transform.forward(state0), T_END / scales.T, DT / scales.T
        )
        x_wrapped = model.transform.inverse(y_end)
        np.testing.assert_allclose(x_wrapped.u, x_inner.u, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(x_wrapped.v, x_inner.v, rtol=RTOL, atol=ATOL)

    def test_matches_the_closed_form(self, inner, scales, state0):
        model = ScaledModel.from_scales(inner, scales, FlowState)
        y_end = _final(
            model, model.transform.forward(state0), T_END / scales.T, DT / scales.T
        )
        expected = math.exp(-float(inner.params.rate) * T_END)
        x_end = model.transform.inverse(y_end)
        np.testing.assert_allclose(x_end.u, expected * state0.u, rtol=RTOL)

    def test_standardised_wrapper_maps_back_to_the_inner_one(self, inner, state0):
        rng = np.random.RandomState(2)
        samples = jax.tree_util.tree_map(
            lambda x: jnp.asarray(0.05 + rng.randn(16, *x.shape) * U), state0
        )
        model = ScaledModel.standardised(inner, samples, per_gridpoint=True)
        x_inner = _final(inner, state0, T_END, DT)
        x_wrapped = model.transform.inverse(
            _final(model, model.transform.forward(state0), T_END, DT)
        )
        np.testing.assert_allclose(x_wrapped.u, x_inner.u, rtol=RTOL, atol=ATOL)


class TestGradients:
    def test_gradient_through_the_wrapped_window_matches_the_inner_one(
        self, inner, scales, state0
    ):
        def loss_inner(m):
            return jnp.sum(_final(m, state0, T_END, DT).u ** 2)

        def loss_wrapped(m):
            wrapped = ScaledModel.from_scales(m, scales, FlowState)
            y = _final(
                wrapped,
                wrapped.transform.forward(state0),
                T_END / scales.T,
                DT / scales.T,
            )
            return jnp.sum(wrapped.transform.inverse(y).u ** 2)

        g_inner = eqx.filter_grad(loss_inner)(inner).params.rate
        g_wrapped = eqx.filter_grad(loss_wrapped)(inner).params.rate
        assert np.isfinite(g_wrapped)
        assert float(g_wrapped) == pytest.approx(float(g_inner), rel=1e-5)


class TestImexStructureSurvivesWrapping:
    def test_mixed_term_model_keeps_its_multiterm(self, scales):
        inner = TermModel(terms=Decay(rate=1e-5) + implicit(Decay(rate=1e-3)))
        assert isinstance(inner.build_terms(), dfx.MultiTerm)
        model = ScaledModel.from_scales(inner, scales, FlowState)
        assert isinstance(model.build_terms(), dfx.MultiTerm)

    def test_explicit_term_model_stays_a_single_term(self, scales):
        inner = TermModel(terms=Decay(rate=1e-5))
        model = ScaledModel.from_scales(inner, scales, FlowState)
        assert not isinstance(model.build_terms(), dfx.MultiTerm)


class TestTimeScaleMustBeUsable:
    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_an_unusable_time_scale(self, inner, scales, bad):
        transform = StateAffine.from_scales(FlowState, scales)
        with pytest.raises(ValueError, match="time_scale"):
            ScaledModel(inner=inner, transform=transform, time_scale=bad)
