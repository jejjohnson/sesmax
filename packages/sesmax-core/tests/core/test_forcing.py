"""Tests for the forcing protocol."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from functax.interp import Cubic, CubicSpline, Linear, Pchip

from sesmax_core import (
    ConstantForcing,
    ForcingProtocol,
    InterpolatedForcing,
    Model,
    NoForcing,
    SeasonalWindForcing,
)
from sesmax_core.forcing import TimeVaryingForcing
from sesmax_core.solvers import imex_stepsize_controller


class DummyGrid(eqx.Module):
    """Minimal grid for testing."""

    nx: int = 10


# ---------------------------------------------------------------------------
# Existing forcing tests
# ---------------------------------------------------------------------------


def test_constant_forcing():
    field = jnp.ones(5) * 3.0
    forcing = ConstantForcing(field=field)
    result = forcing(t=0.0, grid=DummyGrid())
    assert jnp.array_equal(result, field)


def test_constant_forcing_ignores_time():
    field = jnp.array([1.0, 2.0])
    forcing = ConstantForcing(field=field)
    r1 = forcing(t=0.0, grid=DummyGrid())
    r2 = forcing(t=100.0, grid=DummyGrid())
    assert jnp.array_equal(r1, r2)


def test_no_forcing():
    forcing = NoForcing()
    result = forcing(t=0.0, grid=DummyGrid())
    assert result == 0.0


def test_forcing_protocol_is_abstract():
    with pytest.raises(TypeError):
        ForcingProtocol()


# ---------------------------------------------------------------------------
# SeasonalWindForcing
# ---------------------------------------------------------------------------


class TestSeasonalWindForcing:
    def test_amplitude_at_t0(self):
        """At t=0 with phase=0, cos(0)=1, so result = tau0."""
        tau0 = jnp.array([1.0, 2.0, 3.0])
        forcing = SeasonalWindForcing(tau0=tau0, omega=2.0 * jnp.pi)
        result = forcing(t=0.0, grid=DummyGrid())
        assert jnp.allclose(result, tau0)

    def test_varies_with_time(self):
        """Result changes between t=0 and t=T/4."""
        tau0 = jnp.array(5.0)
        omega = 2.0 * jnp.pi  # period = 1
        forcing = SeasonalWindForcing(tau0=tau0, omega=omega)
        r0 = forcing(t=0.0, grid=DummyGrid())
        r_quarter = forcing(t=0.25, grid=DummyGrid())
        # cos(0) = 1, cos(pi/2) = 0
        assert jnp.isclose(r0, 5.0)
        assert jnp.isclose(r_quarter, 0.0, atol=1e-6)

    def test_phase_offset(self):
        """Phase shifts the cosine."""
        tau0 = jnp.array(1.0)
        forcing = SeasonalWindForcing(tau0=tau0, omega=2.0 * jnp.pi, phase=jnp.pi / 2)
        # cos(0 + pi/2) = 0
        result = forcing(t=0.0, grid=DummyGrid())
        assert jnp.isclose(result, 0.0, atol=1e-6)

    def test_grad_through_tau0(self):
        """jax.grad flows through the learnable tau0 amplitude."""

        def loss(forcing):
            return jnp.sum(forcing(t=0.0, grid=DummyGrid()) ** 2)

        tau0 = jnp.array([1.0, 2.0])
        forcing = SeasonalWindForcing(tau0=tau0, omega=1.0)
        grads = jax.grad(loss)(forcing)
        # d/d(tau0) sum((tau0 * cos(0))^2) = d/d(tau0) sum(tau0^2) = 2*tau0
        assert jnp.allclose(grads.tau0, 2.0 * tau0)

    def test_is_forcing_protocol(self):
        forcing = SeasonalWindForcing(tau0=jnp.array(1.0), omega=1.0)
        assert isinstance(forcing, ForcingProtocol)


# ---------------------------------------------------------------------------
# InterpolatedForcing
# ---------------------------------------------------------------------------


class TestInterpolatedForcing:
    def test_evaluates_at_known_points(self):
        """Interpolation matches exact values at data points."""
        ts = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([0.0, 10.0, 20.0])
        forcing = InterpolatedForcing.from_data(ts, values)
        assert jnp.isclose(forcing(t=0.0, grid=DummyGrid()), 0.0)
        assert jnp.isclose(forcing(t=1.0, grid=DummyGrid()), 10.0)
        assert jnp.isclose(forcing(t=2.0, grid=DummyGrid()), 20.0)

    def test_interpolates_between_points(self):
        """Linear interpolation at midpoint."""
        ts = jnp.array([0.0, 2.0])
        values = jnp.array([0.0, 10.0])
        forcing = InterpolatedForcing.from_data(ts, values)
        result = forcing(t=1.0, grid=DummyGrid())
        assert jnp.isclose(result, 5.0)

    def test_multidimensional_values(self):
        """Interpolation works with array-valued data."""
        ts = jnp.array([0.0, 1.0])
        values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        forcing = InterpolatedForcing.from_data(ts, values)
        result = forcing(t=0.5, grid=DummyGrid())
        assert jnp.allclose(result, jnp.array([2.0, 3.0]))

    def test_is_forcing_protocol(self):
        ts = jnp.array([0.0, 1.0])
        values = jnp.array([0.0, 1.0])
        forcing = InterpolatedForcing.from_data(ts, values)
        assert isinstance(forcing, ForcingProtocol)

    def test_cubic_at_known_points(self):
        """Cubic interpolation matches exact values at knot points."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        values = jnp.array([0.0, 1.0, 4.0, 9.0])
        forcing = InterpolatedForcing.from_data(ts, values, method="cubic")
        for i, t in enumerate(ts):
            result = forcing(t=float(t), grid=DummyGrid())
            assert jnp.isclose(result, values[i], atol=1e-5), (
                f"Cubic mismatch at t={t}: got {result}, expected {values[i]}"
            )

    def test_cubic_midpoint_smoother_than_linear(self):
        """Cubic interpolation of x^2 at midpoint is closer to true value."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        values = ts**2  # [0, 1, 4, 9]
        cubic = InterpolatedForcing.from_data(ts, values, method="cubic")
        linear = InterpolatedForcing.from_data(ts, values, method="linear")
        # True value at t=1.5 is 2.25
        cubic_val = cubic(t=1.5, grid=DummyGrid())
        linear_val = linear(t=1.5, grid=DummyGrid())
        true_val = 1.5**2
        assert abs(float(cubic_val) - true_val) <= abs(float(linear_val) - true_val)

    def test_cubic_multidimensional(self):
        """Cubic interpolation works with array-valued data."""
        ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        values = jnp.stack([ts, ts**2], axis=-1)  # (4, 2)
        forcing = InterpolatedForcing.from_data(ts, values, method="cubic")
        result = forcing(t=1.0, grid=DummyGrid())
        assert jnp.allclose(result, jnp.array([1.0, 1.0]), atol=1e-5)

    def test_invalid_method_raises(self):
        ts = jnp.array([0.0, 1.0])
        values = jnp.array([0.0, 1.0])
        with pytest.raises(ValueError, match="Unknown method"):
            InterpolatedForcing.from_data(ts, values, method="spline")


class _Forced(Model):
    """dx/dt = -x + u(t)."""

    forcing: InterpolatedForcing

    def vector_field(self, t, state, args=None):
        return -state + self.forcing(t, None)

    def apply_boundary_conditions(self, state):
        return state


class TestInterpolatedForcingSchemes:
    """Any functax 1-D scheme, through integrix's SampledPath (#130)."""

    ts = np.array([0.0, 0.5, 2.0, 3.0, 4.0])

    def test_linear_scheme_equals_the_linear_method(self):
        with jax.enable_x64(True):
            ts = jnp.asarray(self.ts)
            values = jnp.stack([jnp.sin(ts), jnp.cos(ts)], axis=-1)
            path = InterpolatedForcing.from_data(ts, values, method="linear")
            scheme = InterpolatedForcing.from_data(ts, values, method=Linear())
            # inside the samples and past both ends
            for t in np.linspace(-1.0, 5.0, 25):
                expected = path(t, DummyGrid())
                np.testing.assert_allclose(
                    scheme(t, DummyGrid()), expected, rtol=0.0, atol=1e-12
                )

    @pytest.mark.parametrize("method", ["linear", "cubic", Linear(), Pchip()], ids=str)
    def test_step_ts_are_the_sample_times(self, method):
        ts = jnp.asarray(self.ts, dtype=jnp.float32)
        forcing = InterpolatedForcing.from_data(ts, jnp.sin(ts), method=method)
        assert jnp.array_equal(forcing.step_ts, ts)

    def test_pchip_keeps_non_negative_samples_non_negative(self):
        ts = jnp.asarray(self.ts, dtype=jnp.float32)
        discharge = jnp.array([0.0, 0.0, 4.0, 0.0, 1.0])
        forcing = InterpolatedForcing.from_data(ts, discharge, method=Pchip())
        read = jax.vmap(lambda t: forcing(t, None))(jnp.linspace(0.0, 4.0, 1000))
        assert float(read.min()) >= 0.0

    def test_cubic_spline_reproduces_a_cubic(self):
        with jax.enable_x64(True):
            ts = jnp.asarray(self.ts)
            forcing = InterpolatedForcing.from_data(ts, ts**3, method=CubicSpline())
            np.testing.assert_allclose(forcing(1.3, DummyGrid()), 1.3**3, rtol=1e-12)

    def test_outside_is_the_schemes_to_choose(self):
        ts = jnp.asarray(self.ts, dtype=jnp.float32)
        held = InterpolatedForcing.from_data(ts, ts, method=Linear(), outside="clamp")
        assert held(6.0, DummyGrid()) == 4.0
        assert InterpolatedForcing.from_data(ts, ts, method=Linear())(6.0, None) == 6.0
        with pytest.raises(ValueError, match="always extrapolates"):
            InterpolatedForcing.from_data(ts, ts, outside="clamp")

    def test_rejects_what_is_not_a_scheme_and_uneven_times_for_cubic(self):
        ts = jnp.asarray(self.ts, dtype=jnp.float32)
        with pytest.raises(ValueError, match="'scheme' must be"):
            InterpolatedForcing.from_data(ts, ts, method=jnp.interp)
        with pytest.raises(ValueError, match="CubicSpline or Pchip"):
            InterpolatedForcing.from_data(ts, ts, method=Cubic())

    def test_feeds_a_time_varying_forcing(self):
        ts = jnp.asarray(self.ts, dtype=jnp.float32)
        forcing = InterpolatedForcing.from_data(ts, ts**2, method=Pchip())
        operator = TimeVaryingForcing(lambda u, q: u * q, forcing.path.evaluate)
        assert jnp.isclose(operator(2.0, jnp.array(3.0)), 12.0)

    @pytest.mark.parametrize(
        "method", [CubicSpline(), Pchip()], ids=["spline", "pchip"]
    )
    def test_gradient_in_the_samples_through_integrate(self, method):
        def final(values, forcing_ts):
            forcing = InterpolatedForcing.from_data(forcing_ts, values, method=method)
            # 0.25 divides every sample spacing
            solution = _Forced(forcing).integrate(jnp.zeros(()), 0.0, 4.0, 0.25)
            return solution.ys[-1]

        with jax.enable_x64(True):
            ts = jnp.asarray(self.ts)
            values = jnp.sin(ts) + 0.1 * ts
            gradient = jax.jit(jax.grad(final))(values, ts)
            step = 1e-6
            for k in range(ts.size):
                shift = step * jnp.eye(ts.size)[k]
                difference = final(values + shift, ts) - final(values - shift, ts)
                np.testing.assert_allclose(
                    gradient[k], difference / (2 * step), rtol=1e-6, atol=1e-8
                )

    def test_adaptive_steps_land_on_the_samples(self):
        ts = jnp.asarray(self.ts, dtype=jnp.float32)
        forcing = InterpolatedForcing.from_data(ts, jnp.sin(ts), method=Pchip())
        controller = imex_stepsize_controller(step_ts=forcing.step_ts)
        solution = _Forced(forcing).integrate(
            jnp.zeros(()),
            0.0,
            4.0,
            0.3,
            stepsize_controller=controller,
            saveat=dfx.SaveAt(steps=True),
        )
        assert set(self.ts[1:]) <= {float(t) for t in solution.ts}
