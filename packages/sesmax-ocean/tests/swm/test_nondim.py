"""Tests for the shallow-water models' nondimensional factories.

Both use the inertial scale set. The multilayer one takes Burger numbers **per
interface**, which invert to the stratification one-to-one.

Moved from somax 9cb19b3: the cases on these models of
``tests/test_layered_nondim.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core import Scales, StateAffine
from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


class TestNonlinearShallowWaterMapping:
    @pytest.fixture
    def built(self):
        return NonlinearShallowWater2D.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.05,
            burger=2.0,
            beta_hat=0.5,
            ekman=1e-3,
            ekman_lateral=2e-3,
            wind_hat=0.25,
        )

    def test_unit_coriolis_and_domain(self, built):
        model, _ = built
        assert float(model.consts.f0) == pytest.approx(1.0)
        assert model.grid.Lx == pytest.approx(1.0)

    def test_burger_becomes_gravity(self, built):
        model, _ = built
        assert float(model.consts.gravity) == pytest.approx(2.0)

    def test_ekman_numbers_map_to_drag_and_viscosity(self, built):
        model, _ = built
        assert float(model.params.bottom_drag) == pytest.approx(1e-3)
        assert float(model.params.lateral_viscosity) == pytest.approx(2e-3)

    def test_beta_hat_maps_to_beta(self, built):
        model, _ = built
        assert float(model.consts.beta) == pytest.approx(0.5)

    def test_wind_carries_the_velocity_scale(self, built):
        """tau_hat = tau0/(f0 U), so the model kwarg is tau_hat * Ro."""
        model, _ = built
        assert float(model.params.wind_amplitude) == pytest.approx(0.25 * 0.05)

    def test_velocity_scale_is_rossby(self, built):
        _, scales = built
        assert scales.kind == "inertial"
        assert pytest.approx(0.05) == scales.U
        assert pytest.approx(1.0) == scales.T

    def test_burger_round_trips_through_the_scales(self, built):
        _, scales = built
        assert scales.burger == pytest.approx(2.0, rel=1e-9)

    def test_froude_is_equivalent_to_burger(self):
        """Bu = (Ro/Fr)**2, so the two routes must agree."""
        rossby, burger = 0.05, 2.0
        froude = rossby / np.sqrt(burger)
        by_burger, _ = NonlinearShallowWater2D.from_nondimensional(
            nx=16, ny=16, rossby=rossby, burger=burger
        )
        by_froude, _ = NonlinearShallowWater2D.from_nondimensional(
            nx=16, ny=16, rossby=rossby, froude=float(froude)
        )
        assert float(by_froude.consts.gravity) == pytest.approx(
            float(by_burger.consts.gravity), rel=1e-6
        )

    def test_supplying_both_burger_and_froude_is_rejected(self):
        with pytest.raises(ValueError, match="exactly one of burger or froude"):
            NonlinearShallowWater2D.from_nondimensional(
                nx=16, ny=16, rossby=0.05, burger=1.0, froude=0.05
            )

    def test_supplying_neither_is_rejected(self):
        with pytest.raises(ValueError, match="exactly one of burger or froude"):
            NonlinearShallowWater2D.from_nondimensional(nx=16, ny=16, rossby=0.05)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"rossby": 0.0}, "rossby"),
            ({"ekman": -1.0}, "ekman"),
            ({"aspect": -1.0}, "aspect"),
            ({"burger": 0.0}, "burger"),
        ],
    )
    def test_rejects_out_of_range_inputs(self, kwargs, message):
        base = {"nx": 16, "ny": 16, "rossby": 0.05, "burger": 1.0}
        with pytest.raises(ValueError, match=message):
            NonlinearShallowWater2D.from_nondimensional(**{**base, **kwargs})


class TestMultilayerMapping:
    @pytest.fixture
    def built(self):
        return MultilayerShallowWater2D.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.05,
            burger=[1.0, 0.1],
            thickness_ratio=[1.0, 9.0],
            ekman=1e-3,
        )

    def test_thickness_ratio_becomes_the_layer_depths(self, built):
        model, _ = built
        np.testing.assert_allclose(np.asarray(model.strat.H), [1.0, 9.0])

    def test_burger_inverts_to_reduced_gravity(self, built):
        """g'_k = Bu_k (f0 L)**2 / H_k, with f0 = L = 1."""
        model, _ = built
        np.testing.assert_allclose(
            np.asarray(model.strat.g_prime), [1.0 / 1.0, 0.1 / 9.0], rtol=1e-6
        )

    def test_recovering_burger_from_the_model(self, built):
        model, _ = built
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        f0 = float(model.consts.f0)
        np.testing.assert_allclose(g_prime * H / f0**2, [1.0, 0.1], rtol=1e-6)

    def test_mismatched_sequence_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="same length"):
            MultilayerShallowWater2D.from_nondimensional(
                nx=16, ny=16, rossby=0.05, burger=[1.0, 0.1], thickness_ratio=[1.0]
            )

    def test_non_positive_burger_is_rejected(self):
        with pytest.raises(ValueError, match=r"burger\[1\]"):
            MultilayerShallowWater2D.from_nondimensional(
                nx=16,
                ny=16,
                rossby=0.05,
                burger=[1.0, 0.0],
                thickness_ratio=[1.0, 9.0],
            )


class TestShallowWaterDimensionalEquivalence:
    """An inertial-set nondimensional run reproduces its dimensional twin.

    With ``L``, ``f0``, ``H`` chosen and ``U = f0 L Ro``::

        g    = Bu (f0 L)**2 / H       nu    = Ek_lat f0 L**2
        beta = beta_hat f0 / L        kappa = Ek f0

    and the model's own units are ``H`` for thickness and ``L f0`` for
    velocity.

    The regime is deliberately Ro = 0.2, Bu = 1: the Bernoulli term
    carries ``g*h`` about its mean, so at small Rossby number the mean
    thickness swamps the anomaly and float32 cancellation alone puts a
    few times 1e-4 between two algebraically identical runs. At Ro = 0.2
    the mean is only five times the anomaly and the comparison is clean.
    Thickness error is measured against the anomaly for the same reason:
    dividing by a mean of 500 would flatter any disagreement.
    """

    L = 1.0e6
    F0 = 1.0e-4
    H = 500.0
    ROSSBY = 0.2
    BURGER = 1.0
    EKMAN_LATERAL = 2.0e-3
    NX = NY = 24
    T_ND = 1.0
    N_STEPS = 80

    def gravity(self):
        return self.BURGER * (self.F0 * self.L) ** 2 / self.H

    def dimensional(self):
        return NonlinearShallowWater2D.create(
            nx=self.NX,
            ny=self.NY,
            Lx=self.L,
            Ly=self.L,
            g=self.gravity(),
            f0=self.F0,
            H0=self.H,
            lateral_viscosity=self.EKMAN_LATERAL * self.F0 * self.L**2,
        )

    def nondimensional(self):
        return NonlinearShallowWater2D.from_nondimensional(
            nx=self.NX,
            ny=self.NY,
            rossby=self.ROSSBY,
            burger=self.BURGER,
            ekman_lateral=self.EKMAN_LATERAL,
        )

    def dimensional_state(self, model, scales):
        rng = np.random.RandomState(3)
        shape = (model.grid.Ny, model.grid.Nx)
        return NonlinearSW2DState(
            h=jnp.asarray(self.H + 0.02 * scales.eta * rng.randn(*shape)),
            u=jnp.asarray(0.02 * scales.U * rng.randn(*shape)),
            v=jnp.asarray(0.02 * scales.U * rng.randn(*shape)),
        )

    def to_model_units(self, state):
        """Dimensional state -> the nondimensional model's own units."""
        velocity_unit = self.L * self.F0
        return NonlinearSW2DState(
            h=state.h / self.H,
            u=state.u / velocity_unit,
            v=state.v / velocity_unit,
        )

    def to_dimensional(self, state):
        velocity_unit = self.L * self.F0
        return NonlinearSW2DState(
            h=state.h * self.H,
            u=state.u * velocity_unit,
            v=state.v * velocity_unit,
        )

    def scales(self):
        return Scales.inertial(
            L=self.L,
            f0=self.F0,
            H=self.H,
            rossby=self.ROSSBY,
            g=self.gravity(),
        )

    def test_model_units_match_the_scale_set(self):
        """The unit conversion above is the one ``Scales`` describes."""
        scales = self.scales()
        assert pytest.approx(self.F0 * self.L * self.ROSSBY) == scales.U
        assert pytest.approx(1.0 / self.F0) == scales.T

    def test_tendencies_agree(self):
        """The sharpest check: one RHS evaluation, no accumulation."""
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        state_dim = self.dimensional_state(dim, self.scales())
        state_nd = self.to_model_units(state_dim)

        acceleration_unit = self.L * self.F0 * self.F0
        thickness_rate_unit = self.H * self.F0
        got = nd.vector_field(0.0, state_nd)
        expected = dim.vector_field(0.0, state_dim)
        for field, unit in (
            ("h", thickness_rate_unit),
            ("u", acceleration_unit),
            ("v", acceleration_unit),
        ):
            scaled = np.asarray(getattr(got, field)) * unit
            assert relative_error(scaled, getattr(expected, field)) < 1e-4, field

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        scales = self.scales()
        state_dim = self.dimensional_state(dim, scales)
        state_nd = self.to_model_units(state_dim)

        sol_nd = nd.integrate(
            state_nd, t0=0.0, t1=self.T_ND, dt=self.T_ND / self.N_STEPS
        ).ys
        sol_dim = dim.integrate(
            state_dim,
            t0=0.0,
            t1=self.T_ND * scales.T,
            dt=self.T_ND * scales.T / self.N_STEPS,
        ).ys
        recovered = self.to_dimensional(sol_nd)

        # Thickness is compared against its anomaly about the mean.
        anomaly = np.asarray(sol_dim.h) - self.H
        assert (
            np.abs(np.asarray(recovered.h) - np.asarray(sol_dim.h)).max()
            / np.abs(anomaly).max()
            < 1e-3
        )
        for field in ("u", "v"):
            assert (
                relative_error(getattr(recovered, field), getattr(sol_dim, field))
                < 1e-3
            ), field

    def test_the_states_actually_evolve(self):
        dim = self.dimensional()
        scales = self.scales()
        state = self.dimensional_state(dim, scales)
        evolved = dim.integrate(
            state,
            t0=0.0,
            t1=self.T_ND * scales.T,
            dt=self.T_ND * scales.T / self.N_STEPS,
        ).ys
        change = np.abs(np.asarray(evolved.u) - np.asarray(state.u)).max()
        assert change > 0.1 * np.abs(np.asarray(state.u)).max()


class TestMultilayerStateTransform:
    def test_per_layer_thickness_scales_broadcast(self):
        """The (nl, 1, 1) override path, on a real multilayer state."""
        model, scales = MultilayerShallowWater2D.from_nondimensional(
            nx=12,
            ny=12,
            rossby=0.05,
            burger=[1.0, 0.1],
            thickness_ratio=[1.0, 9.0],
        )
        g_prime = jnp.asarray(model.strat.g_prime)[:, None, None]
        H_k = jnp.asarray(model.strat.H)[:, None, None]
        dH = scales.f0 * scales.U * scales.L / g_prime
        transform = StateAffine.from_scales(
            MultilayerSW2DState, scales, h={"loc": H_k, "scale": dH}
        )
        rng = np.random.RandomState(6)
        shape = (2, model.grid.Ny, model.grid.Nx)
        state = MultilayerSW2DState(
            h=H_k + jnp.asarray(0.01 * rng.randn(*shape)),
            u=jnp.asarray(0.01 * rng.randn(*shape)),
            v=jnp.asarray(0.01 * rng.randn(*shape)),
        )
        assert transform.scale.h.shape == (2, 1, 1)
        recovered = transform.inverse(transform.forward(state))
        np.testing.assert_allclose(recovered.h, state.h, rtol=1e-5)


def test_the_shallow_water_factory_guards_too():
    with pytest.raises(ValueError, match="derived from the dimensionless"):
        NonlinearShallowWater2D.from_nondimensional(
            nx=32, ny=32, rossby=0.1, burger=1.0, g=9.81
        )


class TestMultilayerWindCarriesTheTopLayerThickness:
    """The RHS divides by ``H[0]``, so the factory must multiply by it.

    Otherwise the realised acceleration is ``wind_hat / H[0]`` whenever
    ``thickness_ratio[0]`` is not 1 — silently not the wind that was
    asked for.
    """

    def build(self, first_layer):
        return MultilayerShallowWater2D.from_nondimensional(
            nx=32,
            ny=32,
            rossby=0.1,
            burger=[1.0, 0.05],
            thickness_ratio=[first_layer, 4.0],
            wind_hat=0.5,
        )

    def realised(self, model):
        """The acceleration the RHS actually applies to the top layer."""
        tau0 = float(np.asarray(model.params.wind_amplitude))
        return tau0 / float(np.asarray(model.strat.H)[0])

    def test_the_requested_wind_is_what_the_rhs_applies(self):
        model, scales = self.build(first_layer=2.0)
        assert self.realised(model) == pytest.approx(0.5 * scales.U, rel=1e-6)

    def test_a_unit_top_layer_is_unchanged(self):
        """The case that used to work, so the fix is not a regression."""
        model, scales = self.build(first_layer=1.0)
        assert self.realised(model) == pytest.approx(0.5 * scales.U, rel=1e-6)

    def test_the_thickness_actually_varies_the_amplitude(self):
        thin, _ = self.build(first_layer=1.0)
        thick, _ = self.build(first_layer=2.0)
        assert float(np.asarray(thick.params.wind_amplitude)) == pytest.approx(
            2.0 * float(np.asarray(thin.params.wind_amplitude)), rel=1e-6
        )
