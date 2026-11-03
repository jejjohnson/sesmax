"""Tests for integrating a model in transformed coordinates (somax's cases).

Moved from somax 9cb19b3 (``tests/test_scaled_model.py``): the cases that run
on models in the core, the IMEX structure on Burgers and the linear
shallow-water height offset. The cases on the barotropic QG and nonlinear
shallow-water models move with those models to the ocean component;
``test_scaled_model.py`` checks the same properties on a linear model.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core._src.scaled import ScaledModel
from sesmax_core._src.scales import Scales


NX = NY = 32
L = 1.0e6
U = 0.05


@pytest.fixture
def qg_scales():
    return Scales.advective(L=L, U=U, f0=1e-4)


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


class TestImexStructureSurvivesWrapping:
    """An IMEX model must still be an IMEX model once wrapped.

    The inherited ``build_terms`` would have collapsed the inner
    model's explicit/implicit split into a single ``ODETerm``, which
    makes ``imex_solver()`` incompatible with the term structure and
    leaves the default solver integrating stiff diffusion explicitly.
    """

    def inner(self, imex=True):
        from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel

        return Burgers2DTermModel.create(nx=16, ny=16, nu=0.05, imex=imex)

    def wrapped(self, imex=True):
        from sesmax_core._src.testing.pde2d.burgers import Burgers2DState

        scales = Scales.advective(L=1.0, U=1.0, f0=1.0, H=1.0)
        return ScaledModel.from_scales(self.inner(imex), scales, Burgers2DState)

    def state(self, model):
        from sesmax_core._src.testing.pde2d.burgers import Burgers2DState

        rng = np.random.RandomState(0)
        shape = (model.grid.Ny, model.grid.Nx)
        return Burgers2DState(
            u=jnp.asarray(rng.randn(*shape)), v=jnp.asarray(rng.randn(*shape))
        )

    def test_the_inner_model_really_is_split(self):
        """Otherwise the test below would be vacuous."""
        import diffrax as dfx

        assert isinstance(self.inner(imex=True).build_terms(), dfx.MultiTerm)

    def test_the_wrapper_keeps_the_multiterm(self):
        import diffrax as dfx

        terms = self.wrapped(imex=True).build_terms()
        assert isinstance(terms, dfx.MultiTerm)
        assert len(terms.terms) == 2

    def test_a_non_imex_model_stays_a_single_term(self):
        import diffrax as dfx

        terms = self.wrapped(imex=False).build_terms()
        assert isinstance(terms, dfx.ODETerm)

    def test_the_split_terms_sum_to_the_rescaled_inner_tendency(self):
        """Rescaling each part must not change what they add up to."""
        wrapper = self.wrapped(imex=True)
        state = self.state(wrapper.inner)
        outer_state = wrapper.transform.forward(state)

        parts = [
            term.vector_field(0.0, outer_state, None)
            for term in wrapper.build_terms().terms
        ]
        total = jax.tree_util.tree_map(lambda a, b: a + b, *parts)

        # Against the inner terms, rescaled by hand: both sides then
        # include the boundary conditions the term tree applies, which
        # ``vector_field`` on its own does not.
        inner_parts = [
            term.vector_field(0.0, state, None)
            for term in wrapper.inner.build_terms().terms
        ]
        inner_total = jax.tree_util.tree_map(lambda a, b: a + b, *inner_parts)
        expected = jax.tree_util.tree_map(
            lambda d, scale: wrapper.time_scale * d / scale,
            inner_total,
            wrapper.transform.scale,
        )
        np.testing.assert_allclose(
            np.asarray(total.u), np.asarray(expected.u), rtol=1e-5
        )

    def test_it_integrates_with_the_imex_solver(self):
        from sesmax_core.solvers import imex_solver

        wrapper = self.wrapped(imex=True)
        state = wrapper.transform.forward(self.state(wrapper.inner))
        out = wrapper.integrate(
            state, t0=0.0, t1=0.05, dt=0.01, solver=imex_solver()
        ).ys
        assert np.isfinite(np.asarray(out.u)).all()

    def test_a_stochastic_term_is_refused_rather_than_mangled(self):
        import diffrax as dfx

        from sesmax_core._src.scaled import _rescale_term

        wrapper = self.wrapped(imex=False)
        brownian = dfx.VirtualBrownianTree(
            0.0, 1.0, tol=1e-3, shape=(), key=jax.random.PRNGKey(0)
        )
        term = dfx.ControlTerm(lambda t, y, args: y, brownian)
        with pytest.raises(TypeError, match="cannot rescale"):
            _rescale_term(
                term, transform=wrapper.transform, time_scale=wrapper.time_scale
            )


class TestLinearShallowWaterHeightIsNotOffset:
    """``h`` is already the perturbation in the linear models.

    ``from_scales`` used to apply the total-thickness rule to it, so
    the advertised default call mapped a resting linear state to
    ``-H/eta`` — a large constant coordinate that loosens the solver's
    relative tolerances against the actual anomaly.
    """

    def wrapped(self):
        from sesmax_core._src.testing.swm.linear_2d import (
            LinearShallowWater2D,
            LinearSW2DState,
        )

        inner = LinearShallowWater2D.create(nx=16, ny=16)
        scales = Scales.inertial(L=1.0e6, f0=1.0e-4, H=500.0, rossby=0.01)
        return ScaledModel.from_scales(inner, scales, LinearSW2DState), LinearSW2DState

    def test_the_height_offset_is_zero(self):
        wrapper, _ = self.wrapped()
        assert float(wrapper.transform.loc.h) == 0.0

    def test_a_resting_state_maps_to_zero(self):
        wrapper, state_cls = self.wrapped()
        shape = (wrapper.inner.grid.Ny, wrapper.inner.grid.Nx)
        rest = state_cls(h=jnp.zeros(shape), u=jnp.zeros(shape), v=jnp.zeros(shape))
        got = wrapper.transform.forward(rest)
        np.testing.assert_allclose(np.asarray(got.h), 0.0)
