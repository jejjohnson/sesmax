"""Tests for integrating the barotropic QG model in transformed coordinates.

Moved from somax 9cb19b3: the cases on this model of
``tests/test_scaled_model.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core import Model, ScaledModel, Scales, StateAffine
from sesmax_ocean.qg import BarotropicQG, BarotropicQGState


NX = NY = 32
L = 1.0e6
U = 0.05


@pytest.fixture
def qg_inner():
    return BarotropicQG.create(
        nx=NX,
        ny=NY,
        Lx=L,
        Ly=L,
        f0=1e-4,
        beta=1.6e-11,
        lateral_viscosity=1e3,
        bottom_drag=1e-7,
        wind_amplitude=1e-11,
    )


@pytest.fixture
def qg_scales():
    return Scales.advective(L=L, U=U, f0=1e-4)


@pytest.fixture
def qg_state(qg_inner, qg_scales):
    rng = np.random.RandomState(0)
    field = rng.randn(qg_inner.grid.Ny, qg_inner.grid.Nx)
    field -= field.mean()
    return BarotropicQGState(q=jnp.asarray(1e-3 * qg_scales.vorticity * field))


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


class TestConstruction:
    def test_from_scales_uses_the_time_scale(self, qg_inner, qg_scales):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        assert model.time_scale == pytest.approx(qg_scales.T)

    def test_from_scales_builds_the_matching_transform(self, qg_inner, qg_scales):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        assert model.transform.scale.q == pytest.approx(qg_scales.vorticity)

    def test_standardised_leaves_time_alone(self, qg_inner, qg_state):
        """Standardising the state says nothing about the time unit."""
        rng = np.random.RandomState(1)
        samples = jax.tree_util.tree_map(
            lambda x: jnp.asarray(rng.randn(8, *x.shape) * float(jnp.std(x))),
            qg_state,
        )
        model = ScaledModel.standardised(qg_inner, samples, per_gridpoint=True)
        assert model.time_scale == 1.0

    def test_is_a_model(self, qg_inner, qg_scales):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        assert isinstance(model, Model)

    def test_state_signature_delegates(self, qg_inner, qg_scales):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        assert model.state_signature == qg_inner.state_signature


class TestEquivalenceQG:
    """The wrapped trajectory, mapped back, is the inner trajectory."""

    T_ND = 0.05
    N_STEPS = 10

    def test_trajectories_agree(self, qg_inner, qg_scales, qg_state):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)

        scaled = model.integrate(
            y0, t0=0.0, t1=self.T_ND, dt=self.T_ND / self.N_STEPS
        ).ys
        direct = qg_inner.integrate(
            qg_state,
            t0=0.0,
            t1=self.T_ND * qg_scales.T,
            dt=self.T_ND * qg_scales.T / self.N_STEPS,
        ).ys

        assert relative_error(model.transform.inverse(scaled).q, direct.q) < 1e-3

    def test_the_state_actually_evolves(self, qg_inner, qg_scales, qg_state):
        """Otherwise the equivalence test could pass on a frozen state."""
        direct = qg_inner.integrate(
            qg_state,
            t0=0.0,
            t1=self.T_ND * qg_scales.T,
            dt=self.T_ND * qg_scales.T / self.N_STEPS,
        ).ys
        change = np.abs(np.asarray(direct.q) - np.asarray(qg_state.q)).max()
        assert change > 0.01 * np.abs(np.asarray(qg_state.q)).max()

    def test_step_agrees_with_the_inner_step(self, qg_inner, qg_scales, qg_state):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)
        stepped = model.transform.inverse(model.step(y0, dt=0.01))
        direct = qg_inner.step(qg_state, dt=0.01 * qg_scales.T)
        assert relative_error(stepped.q, direct.q) < 1e-3

    def test_tendency_agrees_with_the_chain_rule(self, qg_inner, qg_scales, qg_state):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)
        got = model.vector_field(0.0, y0).q
        expected = (
            qg_scales.T
            * qg_inner.vector_field(0.0, qg_state).q
            / model.transform.scale.q
        )
        assert relative_error(got, expected) < 1e-5

    def test_identity_transform_is_a_pass_through(self, qg_inner, qg_state):
        model = ScaledModel(
            inner=qg_inner,
            transform=StateAffine.identity(qg_state),
            time_scale=1.0,
        )
        np.testing.assert_allclose(
            model.vector_field(0.0, qg_state).q,
            qg_inner.vector_field(0.0, qg_state).q,
            rtol=1e-6,
        )


class TestStandardisedWrapper:
    """A dimensional model integrated on unit-variance state."""

    @pytest.fixture
    def samples(self, qg_inner, qg_scales):
        """Synthetic samples with a spatially varying mean and spread.

        Deliberately not a spun-up trajectory: what this class checks is
        that an arbitrary invertible standardising transform leaves the
        dynamics alone, and a real spin-up would only add a growing
        trend that makes the statistics harder to reason about without
        testing anything extra.
        """
        rng = np.random.RandomState(7)
        shape = (qg_inner.grid.Ny, qg_inner.grid.Nx)
        spread = qg_scales.vorticity * (0.5 + rng.rand(*shape))
        centre = 0.2 * qg_scales.vorticity * rng.randn(*shape)
        draws = centre[None, :, :] + spread[None, :, :] * rng.randn(16, *shape)
        return BarotropicQGState(q=jnp.asarray(draws))

    def test_standardised_state_has_unit_spread(self, qg_inner, samples):
        model = ScaledModel.standardised(qg_inner, samples, per_gridpoint=True)
        standardised = model.transform.forward(samples)
        np.testing.assert_allclose(
            np.std(np.asarray(standardised.q), axis=0), 1.0, rtol=1e-4
        )

    def test_trajectories_agree_with_the_unwrapped_model(
        self, qg_inner, qg_state, samples
    ):
        model = ScaledModel.standardised(qg_inner, samples, per_gridpoint=True)
        y0 = model.transform.forward(qg_state)
        dt = 1e3
        scaled = model.integrate(y0, t0=0.0, t1=10 * dt, dt=dt).ys
        direct = qg_inner.integrate(qg_state, t0=0.0, t1=10 * dt, dt=dt).ys
        assert relative_error(model.transform.inverse(scaled).q, direct.q) < 1e-3

    def test_standardisation_is_not_the_identity(self, qg_inner, qg_state, samples):
        """Otherwise the agreement above would be trivially true."""
        model = ScaledModel.standardised(qg_inner, samples, per_gridpoint=True)
        y0 = model.transform.forward(qg_state)
        assert relative_error(y0.q, qg_state.q) > 1.0

    def test_diagnostics_are_unchanged(self, qg_inner, qg_state, samples):
        model = ScaledModel.standardised(qg_inner, samples, per_gridpoint=True)
        y0 = model.transform.forward(qg_state)
        for key, value in qg_inner.diagnose(qg_state).invariants().items():
            np.testing.assert_allclose(
                model.diagnose(y0).invariants()[key], value, rtol=1e-3
            )


class TestBoundaryConditions:
    def test_conjugated_bcs_match_the_inner_ones(self, qg_inner, qg_scales, qg_state):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)
        got = model.transform.inverse(model.apply_boundary_conditions(y0))
        expected = qg_inner.apply_boundary_conditions(qg_state)
        assert relative_error(got.q, expected.q) < 1e-5

    def test_boundaries_are_actually_zeroed(self, qg_inner, qg_scales, qg_state):
        """The QG model zeroes q at walls; the wrapper must not undo that."""
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)
        inner_view = model.transform.inverse(model.apply_boundary_conditions(y0))
        np.testing.assert_allclose(np.asarray(inner_view.q)[0, :], 0.0, atol=1e-20)
        np.testing.assert_allclose(np.asarray(inner_view.q)[-1, :], 0.0, atol=1e-20)


class TestTransformsAndGradients:
    def test_jit(self, qg_inner, qg_scales, qg_state):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)
        run = eqx.filter_jit(lambda m, y: m.integrate(y, t0=0.0, t1=0.02, dt=0.01).ys)
        np.testing.assert_allclose(
            run(model, y0).q,
            model.integrate(y0, t0=0.0, t1=0.02, dt=0.01).ys.q,
            rtol=1e-5,
        )

    def test_grad_reaches_the_inner_parameters(self, qg_inner, qg_scales, qg_state):
        model = ScaledModel.from_scales(qg_inner, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)

        def loss(m):
            return jnp.sum(m.integrate(y0, t0=0.0, t1=0.02, dt=0.01).ys.q ** 2)

        grads = eqx.filter_grad(loss)(model)
        g = float(grads.inner.params.lateral_viscosity)
        assert np.isfinite(g)
        assert g != 0.0

    def test_grad_flows_through_constrained_inner_parameters(self, qg_scales, qg_state):
        """paramax unwrapping still happens inside the wrapped RHS."""
        from sesmax_core import positive

        inner = BarotropicQG.create(
            nx=NX, ny=NY, Lx=L, Ly=L, f0=1e-4, beta=1.6e-11, lateral_viscosity=1e3
        )
        wrapped_params = eqx.tree_at(
            lambda m: m.params.lateral_viscosity, inner, positive(1e3)
        )
        model = ScaledModel.from_scales(wrapped_params, qg_scales, BarotropicQGState)
        y0 = model.transform.forward(qg_state)

        def loss(m):
            return jnp.sum(m.integrate(y0, t0=0.0, t1=0.02, dt=0.01).ys.q ** 2)

        grads = eqx.filter_grad(loss)(model)
        raw = grads.inner.params.lateral_viscosity.args[0]
        assert np.isfinite(float(raw))


class TestTimeScaleMustBeUsable:
    """A degenerate time scale should say so, not integrate anyway.

    Zero is the quiet one: every tendency is multiplied by it and the
    inner model is evaluated at ``t = 0`` forever, so the solve returns
    a frozen trajectory that looks like a physical result.
    """

    def parts(self):
        model = BarotropicQG.create(nx=16, ny=16)
        transform = StateAffine.from_scales(
            BarotropicQGState, Scales.advective(L=1.0, U=1.0)
        )
        return model, transform

    @pytest.mark.parametrize(
        "bad", [0.0, -1.0, -2.5, float("nan"), float("inf"), float("-inf")]
    )
    def test_a_degenerate_time_scale_is_rejected(self, bad):
        model, transform = self.parts()
        with pytest.raises(ValueError, match="time_scale must be"):
            ScaledModel(inner=model, transform=transform, time_scale=bad)

    def test_a_positive_time_scale_is_accepted(self):
        model, transform = self.parts()
        assert ScaledModel(
            inner=model, transform=transform, time_scale=2.5
        ).time_scale == pytest.approx(2.5)

    def test_the_default_is_unchanged(self):
        model, transform = self.parts()
        wrapper = ScaledModel(inner=model, transform=transform)
        assert wrapper.time_scale == pytest.approx(1.0)

    def test_the_scales_constructor_still_works(self):
        model = BarotropicQG.create(nx=16, ny=16)
        scales = Scales.advective(L=1.0e6, U=0.1)
        wrapper = ScaledModel.from_scales(model, scales, BarotropicQGState)
        assert wrapper.time_scale == pytest.approx(scales.T)

    def test_the_time_scale_actually_scales_the_tendency(self):
        """So rejecting zero matters: the parameter is a real factor.

        A zero can no longer be constructed to demonstrate the frozen
        trajectory directly — which is the point of the check — so this
        pins the linearity that makes zero degenerate.
        """
        from sesmax_ocean.qg import BarotropicQGState

        model, transform = self.parts()
        state = BarotropicQGState(
            q=jnp.asarray(
                1.0e-6 * np.random.RandomState(0).randn(model.grid.Ny, model.grid.Nx)
            )
        )
        single = ScaledModel(inner=model, transform=transform, time_scale=1.0)
        double = ScaledModel(inner=model, transform=transform, time_scale=2.0)
        one = np.asarray(single.vector_field(0.0, state).q)
        two = np.asarray(double.vector_field(0.0, state).q)
        np.testing.assert_allclose(two, 2.0 * one, rtol=1e-4)
