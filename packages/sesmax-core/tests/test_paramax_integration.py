"""Tests for constrained parameters via paramax.

The contract: a model whose ``Params`` hold wrapped values behaves
exactly like one holding the equivalent plain arrays, gradients land on
the stored unconstrained values, and the constraint cannot be violated
by a gradient step.

Moved from somax 9cb19b3 without the cases that build a barotropic QG model
(somax's ``build_model`` and ``state`` helpers): those are checked on a core
model separately, and move with the QG model to the ocean component.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import paramax
import pytest

from sesmax_core._src.types import (
    as_parameter,
    frozen,
    interval,
    positive,
)


class TestHelpers:
    def test_positive_round_trips(self):
        assert float(paramax.unwrap(positive(100.0))) == pytest.approx(100.0, rel=1e-5)

    def test_positive_round_trips_for_arrays(self):
        values = jnp.asarray([1e-8, 1.0, 1e4])
        np.testing.assert_allclose(paramax.unwrap(positive(values)), values, rtol=1e-4)

    def test_positive_stays_positive_for_any_stored_value(self):
        """The point of the wrapper: no raw value maps to a negative."""
        wrapper = positive(1.0)
        for raw in (-80.0, -1.0, 0.0, 1e3):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            assert float(paramax.unwrap(stepped)) > 0.0

    def test_positive_never_goes_negative_even_on_underflow(self):
        """Far into the tail softplus underflows to 0.0 — never below it."""
        wrapper = positive(1.0)
        for raw in (-1e3, -200.0, -104.0):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            assert float(paramax.unwrap(stepped)) == 0.0

    def test_positive_rejects_non_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            positive(0.0)
        with pytest.raises(ValueError, match="strictly positive"):
            positive(-1.0)

    def test_positive_survives_a_large_value(self):
        """``log(exp(x) - 1)`` would overflow here; the stable form does not."""
        assert np.isfinite(float(paramax.unwrap(positive(1e5))))
        assert float(paramax.unwrap(positive(1e5))) == pytest.approx(1e5, rel=1e-4)

    def test_interval_round_trips(self):
        assert float(paramax.unwrap(interval(0.25, 0.0, 1.0))) == pytest.approx(
            0.25, rel=1e-5
        )

    def test_interval_stays_inside_for_any_stored_value(self):
        wrapper = interval(4.0, 2.0, 6.0)
        # Beyond about +-16 the offset from the bound falls below float32
        # resolution at 2.0, so the value rounds onto the bound itself.
        for raw in (-15.0, 0.0, 15.0):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            got = float(paramax.unwrap(stepped))
            assert 2.0 < got < 6.0

    def test_interval_saturates_onto_a_bound_but_never_past_it(self):
        """Deep in either tail sigmoid reaches exactly 0 or 1."""
        wrapper = interval(4.0, 2.0, 6.0)
        for raw in (-1e3, 1e3):
            stepped = eqx.tree_at(lambda w: w.args[0], wrapper, jnp.asarray(raw))
            got = float(paramax.unwrap(stepped))
            assert 2.0 <= got <= 6.0

    def test_interval_rejects_a_value_outside_the_bounds(self):
        with pytest.raises(ValueError, match="strictly inside"):
            interval(1.5, 0.0, 1.0)

    def test_interval_rejects_unordered_bounds(self):
        with pytest.raises(ValueError, match="upper must exceed lower"):
            interval(0.5, 1.0, 0.0)

    def test_frozen_round_trips(self):
        assert float(paramax.unwrap(frozen(3.0))) == pytest.approx(3.0)

    def test_frozen_has_zero_gradient(self):
        """NonTrainable cuts the backward pass, returning exact zero."""

        def loss(x):
            return jnp.sum(paramax.unwrap(x) ** 2)

        g = eqx.filter_grad(loss)(frozen(3.0))
        leaves = jax.tree_util.tree_leaves(eqx.filter(g, eqx.is_inexact_array))
        assert leaves
        assert all(float(leaf) == 0.0 for leaf in leaves)


class TestNonFiniteInitialValues:
    """NaN must not slip past the range checks.

    Every comparison with NaN is false, so a check written as
    ``any(value <= 0)`` waves it through and builds a wrapper that
    unwraps to NaN — which then contaminates the integration and the
    gradients far from the call that caused it.
    """

    def test_positive_rejects_nan(self):
        with pytest.raises(ValueError, match="strictly positive"):
            positive(jnp.nan)

    def test_positive_rejects_nan_in_an_array(self):
        with pytest.raises(ValueError, match="strictly positive"):
            positive(jnp.asarray([1.0, jnp.nan, 3.0]))

    def test_positive_still_rejects_zero_and_negatives(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="strictly positive"):
                positive(bad)

    def test_infinity_is_rejected(self):
        """+inf passes ``> 0``, but its softplus pre-image is not finite.

        Storing it would unwrap back to ``inf`` — an overflowed
        calibration value reaching the integration rather than the
        error the docstring promises.
        """
        with pytest.raises(ValueError, match="finite"):
            positive(jnp.inf)

    def test_infinity_in_an_array_is_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            positive(jnp.asarray([1.0, jnp.inf]))

    def test_interval_rejects_nan(self):
        with pytest.raises(ValueError, match="strictly inside"):
            interval(jnp.nan, 0.0, 1.0)

    def test_interval_rejects_nan_in_an_array(self):
        with pytest.raises(ValueError, match="strictly inside"):
            interval(jnp.asarray([0.5, jnp.nan]), 0.0, 1.0)

    def test_interval_still_rejects_out_of_range(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError, match="strictly inside"):
                interval(bad, 0.0, 1.0)


class TestPublicFactoriesAcceptConstrainedValues:
    """``create(lateral_viscosity=positive(...))`` must simply work.

    The factories used to call ``jnp.array`` on every parameter, which
    cannot convert a paramax wrapper, so a constrained model could only
    be built by surgery with ``eqx.tree_at``.
    """

    def test_as_parameter_passes_wrappers_through(self):
        wrapper = positive(1.0)
        assert as_parameter(wrapper) is wrapper

    def test_as_parameter_converts_everything_else(self):
        assert isinstance(as_parameter(2.0), jnp.ndarray)


class TestIntervalBoundsMustBeFinite:
    """A semi-infinite interval has no logit.

    ``interval(0.5, 0.0, inf)`` passes the ordering test, but ``scaled``
    collapses to zero, the stored raw value becomes ``-inf``, and
    unwrapping evaluates ``inf * sigmoid(-inf)`` — NaN.
    """

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [(0.0, float("inf")), (float("-inf"), 1.0), (float("-inf"), float("inf"))],
    )
    def test_an_infinite_bound_is_rejected(self, lower, upper):
        with pytest.raises(ValueError, match="bounds must be finite"):
            interval(0.5, lower, upper)

    def test_a_nan_bound_is_rejected(self):
        with pytest.raises(ValueError, match="bounds must be finite"):
            interval(0.5, 0.0, float("nan"))

    def test_finite_bounds_still_work(self):
        assert float(paramax.unwrap(interval(0.5, 0.0, 1.0))) == pytest.approx(0.5)

    def test_the_error_points_at_positive(self):
        """A one-sided lower bound is what ``positive`` is for."""
        with pytest.raises(ValueError, match="positive"):
            interval(0.5, 0.0, float("inf"))


class TestTermModelFactoriesAcceptConstrainedValues:
    """The term-model factories were missed by the earlier fix.

    They convert with ``jnp.asarray`` outside a ``Params``
    construction, so a repo-wide search for ``*Params(`` did not reach
    them and they still raised on a wrapper.
    """

    def test_burgers_term_model_create(self):
        from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel

        model = Burgers2DTermModel.create(nx=16, ny=16, nu=positive(0.05))
        assert model is not None

    def test_burgers_term_model_from_model(self):
        from sesmax_core._src.testing.pde2d.burgers import Burgers2D
        from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel

        plain = Burgers2D.create(nx=16, ny=16, nu=positive(0.05))
        assert Burgers2DTermModel.from_model(plain) is not None

    def test_the_imex_split_survives_a_wrapper(self):
        import diffrax as dfx

        from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel

        model = Burgers2DTermModel.create(nx=16, ny=16, nu=positive(0.05), imex=True)
        assert isinstance(model.build_terms(), dfx.MultiTerm)

    def test_linear_swm_term_model_from_model(self):
        from sesmax_core._src.testing.swm.linear_2d import LinearShallowWater2D
        from sesmax_core._src.testing.swm.linear_swm_terms import LinearSWM2DTermModel

        plain = LinearShallowWater2D.create(
            nx=16, ny=16, lateral_viscosity=positive(10.0)
        )
        assert LinearSWM2DTermModel.from_model(plain) is not None


class TestIntervalSpansMustBeRepresentable:
    """Finite endpoints do not make their difference finite."""

    def test_a_span_that_overflows_float32_is_rejected(self):
        with pytest.raises(ValueError, match="overflows float32"):
            interval(0.0, -3e38, 3e38)

    def test_the_span_really_would_overflow(self):
        """Otherwise the test above would pass for the wrong reason."""
        assert math.isfinite(3e38 - -3e38)
        assert not bool(
            jnp.isfinite(
                jnp.asarray(3e38, jnp.float32) - jnp.asarray(-3e38, jnp.float32)
            )
        )

    def test_the_same_span_is_fine_in_float64(self):
        jax.config.update("jax_enable_x64", True)
        try:
            wrapped = interval(jnp.asarray(0.0, jnp.float64), -3e38, 3e38)
            assert bool(jnp.isfinite(paramax.unwrap(wrapped)))
        finally:
            jax.config.update("jax_enable_x64", False)

    def test_an_ordinary_interval_is_unaffected(self):
        assert float(paramax.unwrap(interval(0.5, 0.0, 1.0))) == pytest.approx(0.5)


class TestBoundaryConditionsSeeUnwrappedParameters:
    """``integrate()`` applies them once before the solve begins.

    That first call bypassed the RHS's unwrapping, so a boundary
    condition that reads a constrained parameter raised on arithmetic
    against the wrapper.
    """

    def test_the_method_is_in_the_unwrapped_list(self):
        from sesmax_core._src.model import _UNWRAPPED_METHODS

        assert "apply_boundary_conditions" in _UNWRAPPED_METHODS
