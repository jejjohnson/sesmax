"""Tests for the baroclinic QG nondimensional factory.

Burger numbers are **per interface**, which invert to the stratification
one-to-one; the per-mode deformation radii that follow from the eigenproblem
are a different quantity and are checked separately.

Moved from somax 9cb19b3: the cases on this model of
``tests/test_layered_nondim.py``.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core import Scales, StateAffine
from sesmax_ocean.qg import BaroclinicQG, BaroclinicQGState, ReparameterizedQG
from sesmax_ocean.resolution import AssertionFailedError, check_deformation_radius


# These grids are marginal on purpose, and the guards say so. somax's suite had
# loguru, which takes the message; without it the message is a warning.
pytestmark = pytest.mark.filterwarnings("ignore:.*marginally resolved:UserWarning")


def relative_error(got, expected):
    got, expected = np.asarray(got), np.asarray(expected)
    return np.abs(got - expected).max() / np.abs(expected).max()


class TestBaroclinicQGMapping:
    KW: ClassVar[dict] = {
        "nx": 64,
        "ny": 64,
        "rossby": 0.02,
        "beta_hat": 20.0,
        "burger": [1.0, 0.02],
        "thickness_ratio": [1.0, 4.0],
        "delta_M": 0.06,
    }

    @pytest.fixture
    def built(self):
        return BaroclinicQG.from_nondimensional(**self.KW)

    def test_rossby_sets_the_coriolis_parameter(self, built):
        model, _ = built
        assert float(model.consts.f0) == pytest.approx(1.0 / 0.02)

    def test_delta_M_sets_the_viscosity(self, built):
        model, _ = built
        assert float(model.params.lateral_viscosity) == pytest.approx(
            0.06**3 * 20.0, rel=1e-5
        )

    def test_wind_defaults_to_the_sverdrup_value(self, built):
        model, _ = built
        assert float(model.params.wind_amplitude) == pytest.approx(20.0)

    def test_advective_unit_scales(self, built):
        _, scales = built
        assert scales.kind == "advective"
        assert scales.T == 1.0
        assert scales.rossby == pytest.approx(0.02)

    def test_burger_inverts_with_the_model_f0(self, built):
        model, _ = built
        f0 = float(model.consts.f0)
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        np.testing.assert_allclose(g_prime * H / f0**2, [1.0, 0.02], rtol=1e-5)

    def test_modal_radii_are_in_units_of_L(self, built):
        """So they are directly comparable with dx, which is what the guard does.

        With a free surface (``g_prime[0]`` finite) every mode has a finite
        radius: the gravest one is the external radius ``sqrt(g H) / f0``,
        of order the basin, and the internal one is a small fraction of L.
        """
        model, _ = built
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        f0 = float(model.consts.f0)
        radii = np.sort(np.asarray(model.modal.rossby_radii))[::-1]
        assert radii.size == 2
        assert np.all(np.isfinite(radii))
        external = np.sqrt(g_prime[0] * H.sum()) / f0
        assert radii[0] == pytest.approx(external, rel=1e-2)
        assert 0.0 < radii[1] < 1.0

    def test_modal_radius_is_not_the_interface_burger(self):
        """The two conventions genuinely differ; the docs say so, so test it."""
        model, _ = BaroclinicQG.from_nondimensional(**self.KW)
        radii = np.asarray(model.modal.rossby_radii)
        internal = float(radii.min())
        # sqrt of the second interface Burger number, the naive guess.
        assert internal != pytest.approx(np.sqrt(0.02), rel=1e-2)

    def test_modal_radius_is_near_the_rigid_lid_two_layer_formula(self):
        """L_d**2 = g' H1 H2 / (f0**2 (H1 + H2)), up to the free surface.

        The eigenproblem includes the free-surface mode through
        ``g_prime[0]``, so the internal radius sits a few percent below
        the rigid-lid value rather than on it.
        """
        model, _ = BaroclinicQG.from_nondimensional(**self.KW)
        g_prime = np.asarray(model.strat.g_prime)
        H = np.asarray(model.strat.H)
        f0 = float(model.consts.f0)
        rigid_lid = np.sqrt(g_prime[1] * H[0] * H[1] / (f0**2 * (H[0] + H[1])))
        radii = np.asarray(model.modal.rossby_radii)
        internal = float(radii.min())
        assert internal == pytest.approx(rigid_lid, rel=0.1)
        assert internal < rigid_lid

    def test_reparameterized_uses_the_same_mapping(self):
        bc, _ = BaroclinicQG.from_nondimensional(**self.KW)
        rp, _ = ReparameterizedQG.from_nondimensional(**self.KW)
        np.testing.assert_allclose(
            np.asarray(rp.strat.g_prime), np.asarray(bc.strat.g_prime), rtol=1e-6
        )
        assert float(rp.consts.f0) == pytest.approx(float(bc.consts.f0))


class TestLayeredResolutionGuards:
    def test_unresolved_deformation_radius_is_rejected(self):
        """A tiny Burger number means the internal radius is sub-grid."""
        with pytest.raises(
            AssertionFailedError, match="deformation_radius check FAILED"
        ):
            BaroclinicQG.from_nondimensional(
                nx=32,
                ny=32,
                rossby=0.02,
                beta_hat=20.0,
                burger=[1.0, 1e-6],
                thickness_ratio=[1.0, 4.0],
                delta_M=0.1,
            )

    def test_unresolved_munk_layer_is_rejected(self):
        with pytest.raises(AssertionFailedError, match="munk_width check FAILED"):
            BaroclinicQG.from_nondimensional(
                nx=32,
                ny=32,
                rossby=0.02,
                beta_hat=20.0,
                burger=[1.0, 0.05],
                thickness_ratio=[1.0, 4.0],
                delta_M=0.005,
            )

    def test_check_resolution_false_skips_the_guards(self):
        model, _ = BaroclinicQG.from_nondimensional(
            nx=16,
            ny=16,
            rossby=0.02,
            beta_hat=20.0,
            burger=[1.0, 1e-6],
            thickness_ratio=[1.0, 4.0],
            delta_M=0.001,
            check_resolution=False,
        )
        assert model is not None


class TestBaroclinicDimensionalEquivalence:
    """An advective-set nondimensional QG run reproduces its dimensional twin."""

    L = 1.0e6
    U = 0.05
    ROSSBY = 0.02
    BETA_HAT = 15.0
    BURGER = (1.0, 0.05)
    THICKNESS = (1.0, 4.0)
    DELTA_M = 0.08
    NX = NY = 48

    def dimensional(self):
        f0 = self.U / (self.ROSSBY * self.L)
        beta = self.BETA_HAT * self.U / self.L**2
        H = tuple(h * 1000.0 for h in self.THICKNESS)
        g_prime = tuple(
            bu * (f0 * self.L) ** 2 / h for bu, h in zip(self.BURGER, H, strict=True)
        )
        return BaroclinicQG.create(
            nx=self.NX,
            ny=self.NY,
            Lx=self.L,
            Ly=self.L,
            f0=f0,
            beta=beta,
            n_layers=2,
            H=H,
            g_prime=g_prime,
            lateral_viscosity=self.DELTA_M**3 * self.L**3 * beta,
            # tau_hat = tau0 L**2 / (U**2 H_1): the RHS divides by H[0].
            wind_amplitude=self.BETA_HAT * self.U**2 * H[0] / self.L**2,
        )

    def nondimensional(self):
        return BaroclinicQG.from_nondimensional(
            nx=self.NX,
            ny=self.NY,
            rossby=self.ROSSBY,
            beta_hat=self.BETA_HAT,
            burger=list(self.BURGER),
            thickness_ratio=list(self.THICKNESS),
            delta_M=self.DELTA_M,
        )

    def test_burger_numbers_match_between_the_two(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        for model in (dim, nd):
            f0 = float(model.consts.f0)
            g_prime = np.asarray(model.strat.g_prime)
            H = np.asarray(model.strat.H)
            L = model.grid.Lx
            np.testing.assert_allclose(
                g_prime * H / (f0 * L) ** 2, self.BURGER, rtol=1e-4
            )

    def test_modal_radii_match_after_rescaling(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        dim_radii = np.asarray(dim.modal.rossby_radii) / self.L
        nd_radii = np.asarray(nd.modal.rossby_radii)
        finite = np.isfinite(dim_radii)
        np.testing.assert_allclose(dim_radii[finite], nd_radii[finite], rtol=1e-3)

    def test_trajectories_agree_under_the_scales_mapping(self):
        dim = self.dimensional()
        nd, _ = self.nondimensional()
        scales = Scales.advective(L=self.L, U=self.U, f0=float(dim.consts.f0))
        transform = StateAffine.from_scales(BaroclinicQGState, scales)

        rng = np.random.RandomState(5)
        field = rng.randn(2, dim.grid.Ny, dim.grid.Nx)
        field -= field.mean(axis=(1, 2), keepdims=True)
        state_dim = BaroclinicQGState(q=jnp.asarray(0.02 * scales.vorticity * field))
        state_nd = transform.forward(state_dim)

        t_nd, n = 0.3, 30
        sol_nd = nd.integrate(state_nd, t0=0.0, t1=t_nd, dt=t_nd / n).ys
        sol_dim = dim.integrate(
            state_dim, t0=0.0, t1=t_nd * scales.T, dt=t_nd * scales.T / n
        ).ys
        assert relative_error(transform.inverse(sol_nd).q, sol_dim.q) < 5e-3

    def test_the_states_actually_evolve(self):
        dim = self.dimensional()
        scales = Scales.advective(L=self.L, U=self.U, f0=float(dim.consts.f0))
        rng = np.random.RandomState(5)
        field = rng.randn(2, dim.grid.Ny, dim.grid.Nx)
        field -= field.mean(axis=(1, 2), keepdims=True)
        state = BaroclinicQGState(q=jnp.asarray(0.02 * scales.vorticity * field))
        evolved = dim.integrate(
            state, t0=0.0, t1=0.3 * scales.T, dt=0.3 * scales.T / 30
        ).ys
        change = np.abs(np.asarray(evolved.q) - np.asarray(state.q)).max()
        assert change > 0.01 * np.abs(np.asarray(state.q)).max()


class TestReturnedScalesUseTheInterfaceGravity:
    """``Scales.g`` must be ``g_prime[0]``, not standard gravity.

    The scales describe the *nondimensional* model, whose surface-mode
    gravity is the first reduced gravity. Left at 9.81 they disagree
    with the Burger number that was asked for, and the thickness-anomaly
    scale ``f0 U L / g`` comes out wrong by the ratio between them —
    which can be orders of magnitude.
    """

    def build(self, cls, **kw):
        return cls.from_nondimensional(
            nx=64,
            ny=64,
            rossby=0.02,
            beta_hat=20.0,
            burger=[1.0, 0.02],
            thickness_ratio=[1.0, 4.0],
            delta_M=0.06,
            **kw,
        )

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_burger_round_trips(self, cls):
        _, scales = self.build(cls)
        assert scales.burger == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_gravity_is_the_first_reduced_gravity(self, cls):
        model, scales = self.build(cls)
        g_prime = float(np.asarray(model.strat.g_prime)[0])
        assert scales.g == pytest.approx(g_prime, rel=1e-6)

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_gravity_is_not_standard_gravity(self, cls):
        _, scales = self.build(cls)
        assert scales.g != pytest.approx(9.81)

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_the_height_scale_follows(self, cls):
        """``eta = f0 U L / g`` is what a StateAffine gives ``h``."""
        _, scales = self.build(cls)
        assert scales.eta == pytest.approx(scales.f0 / scales.g, rel=1e-6)

    def test_a_different_burger_gives_a_different_gravity(self):
        _, low = self.build(BaroclinicQG)
        _, high = BaroclinicQG.from_nondimensional(
            nx=64,
            ny=64,
            rossby=0.02,
            beta_hat=20.0,
            burger=[4.0, 0.02],
            thickness_ratio=[1.0, 4.0],
            delta_M=0.06,
        )
        assert high.g == pytest.approx(4.0 * low.g, rel=1e-6)


class TestForwardedKwargsCannotOverrideDerivedValues:
    """``**create_kw`` must not smuggle in a second stratification.

    ``create`` prefers an explicit ``stratification`` over the ``H`` /
    ``g_prime`` / ``n_layers`` the factory derived, while the returned
    scales and the wind normalisation still describe the derived ones —
    so the model and its scales would be different systems.
    """

    def base(self):
        return dict(
            nx=64,
            ny=64,
            rossby=0.02,
            beta_hat=20.0,
            burger=[1.0, 0.02],
            thickness_ratio=[1.0, 4.0],
            delta_M=0.06,
        )

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    @pytest.mark.parametrize(
        "name", ["stratification", "H", "g_prime", "n_layers", "f0", "beta"]
    )
    def test_a_derived_argument_is_rejected(self, cls, name):
        with pytest.raises(ValueError, match="derived from the dimensionless"):
            cls.from_nondimensional(**self.base(), **{name: None})

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_an_unrelated_argument_still_goes_through(self, cls):
        model, _ = cls.from_nondimensional(**self.base(), wind_profile="single")
        assert model is not None

    def test_the_message_names_the_offending_argument(self):
        with pytest.raises(ValueError, match=r"\['stratification'\]"):
            BaroclinicQG.from_nondimensional(**self.base(), stratification=None)


class TestExplicitWindAmplitudeIsValidated:
    """Every other coefficient is checked; ``wind_hat`` was not.

    A non-finite value went straight into ``wind_amplitude`` and made
    the tendencies non-finite on the first step.
    """

    def base(self):
        return dict(
            nx=64,
            ny=64,
            rossby=0.02,
            beta_hat=20.0,
            burger=[1.0, 0.02],
            thickness_ratio=[1.0, 4.0],
            delta_M=0.06,
        )

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_a_bad_wind_hat_is_rejected(self, cls, bad):
        with pytest.raises(ValueError, match="wind_hat"):
            cls.from_nondimensional(**self.base(), wind_hat=bad)

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_zero_is_still_allowed(self, cls):
        model, _ = cls.from_nondimensional(**self.base(), wind_hat=0.0)
        assert float(np.asarray(model.params.wind_amplitude)) == 0.0

    @pytest.mark.parametrize("cls", [BaroclinicQG, ReparameterizedQG])
    def test_the_default_still_works(self, cls):
        model, _ = cls.from_nondimensional(**self.base())
        assert float(np.asarray(model.params.wind_amplitude)) > 0.0


class TestDeformationRadiusUsesTheCoarserSpacing:
    """A deformation radius is isotropic, so both directions must span it.

    Taking the finer spacing passes a grid that resolves it along one
    axis only.
    """

    def model(self, nx, ny):
        return BaroclinicQG.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=1.0,
            f0=50.0,
            beta=20.0,
            n_layers=2,
            H=(1.0, 4.0),
            g_prime=(2500.0, 50.0),
        )

    def test_an_anisotropic_grid_no_longer_passes(self):
        with pytest.raises(AssertionFailedError, match="L_d/dx"):
            check_deformation_radius(None, self.model(nx=4, ny=256))

    def test_the_isotropic_grid_at_the_finer_spacing_passes(self):
        """So the failure above is about the coarse axis, not the model."""
        check_deformation_radius(None, self.model(nx=256, ny=256))

    def test_the_reported_spacing_is_the_coarser_one(self):
        model = self.model(nx=4, ny=256)
        with pytest.raises(AssertionFailedError) as excinfo:
            check_deformation_radius(None, model)
        assert f"{max(model.grid.spacing('x'), model.grid.spacing('y')):.4g}" in str(
            excinfo.value
        )


class TestGuardsDoNotNeedTheCliExtras:
    """A base install must reach the resolution guards.

    ``loguru`` is not a dependency of the package, so the factories and
    their guards have to work without it.
    """

    @pytest.mark.parametrize(
        "factory",
        ["BarotropicQG", "BaroclinicQG", "ReparameterizedQG"],
    )
    def test_the_factory_builds_without_loguru(self, factory):
        import subprocess
        import sys
        import textwrap

        layered = """burger=[1.0, 0.02], thickness_ratio=[1.0, 4.0], beta_hat=20.0"""
        args = "beta_hat=50.0" if factory == "BarotropicQG" else layered
        script = textwrap.dedent(f"""
            import builtins
            real = builtins.__import__
            def guarded(name, *a, **k):
                if name == "loguru":
                    raise ModuleNotFoundError("No module named 'loguru'")
                return real(name, *a, **k)
            builtins.__import__ = guarded
            from sesmax_ocean.qg import {factory}
            {factory}.from_nondimensional(
                nx=128, ny=128, rossby=0.02, delta_M=0.06, {args}
            )
            print("ok")
        """)
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert out.stdout.strip() == "ok", out.stderr[-800:]


class TestArrayBackedSequencesAreAccepted:
    """``burger`` and ``thickness_ratio`` are sequences of numbers.

    ``if not burger`` raises on a multi-element array — ambiguous truth
    value — so array inputs were rejected before the conversion that
    would have handled them.
    """

    def build(self, maker):
        return BaroclinicQG.from_nondimensional(
            nx=128,
            ny=128,
            rossby=0.02,
            beta_hat=20.0,
            burger=maker([1.0, 0.02]),
            thickness_ratio=maker([1.0, 4.0]),
            delta_M=0.06,
        )

    def test_a_numpy_array_works(self):
        _, scales = self.build(np.asarray)
        assert scales.burger == pytest.approx(1.0, rel=1e-6)

    def test_a_jax_array_works(self):
        _, scales = self.build(jnp.asarray)
        assert scales.burger == pytest.approx(1.0, rel=1e-6)

    def test_a_list_still_works(self):
        _, scales = self.build(list)
        assert scales.burger == pytest.approx(1.0, rel=1e-6)

    def test_all_three_agree(self):
        values = [
            float(self.build(m)[1].burger) for m in (np.asarray, jnp.asarray, list)
        ]
        assert values[0] == pytest.approx(values[1]) == pytest.approx(values[2])

    def test_an_empty_sequence_is_still_rejected(self):
        for maker in (np.asarray, list):
            with pytest.raises(ValueError, match="must not be empty"):
                BaroclinicQG.from_nondimensional(
                    nx=64,
                    ny=64,
                    rossby=0.02,
                    beta_hat=20.0,
                    burger=maker([]),
                    thickness_ratio=maker([]),
                    delta_M=0.06,
                )


class TestTheModelSharesTheScalesGravity:
    """``consts.gravity`` must agree with ``Scales.g``.

    Passing ``g_prime`` alone left ``ReparameterizedQG.create`` at its
    9.81 default, so the model's constants contradicted its own
    stratification and the scales it was handed back with. Diagnostics
    that read ``consts.gravity`` — ``geostrophic_imbalance`` among
    them — then used the wrong coefficient.
    """

    @staticmethod
    def build(**kw):
        return ReparameterizedQG.from_nondimensional(
            nx=64,
            ny=64,
            rossby=0.02,
            beta_hat=1.0,
            burger=(1.0, 0.1),
            thickness_ratio=(1.0, 3.0),
            delta_M=0.02,
            delta_S=0.0,
            check_resolution=False,
            **kw,
        )

    def test_they_match(self):
        model, scales = self.build()
        assert float(model.swm.consts.gravity) == pytest.approx(scales.g)

    def test_it_is_not_the_default(self):
        """Otherwise the test above could pass by coincidence."""
        _, scales = self.build()
        assert scales.g != pytest.approx(9.81)

    def test_a_forwarded_gravity_is_rejected(self):
        with pytest.raises(ValueError, match="g"):
            self.build(g=9.81)


class TestTheDefaultWindGivesAUnitSverdrupVelocity:
    """The default is a Sverdrup-balance choice, and that balances curl.

    ``beta v = curl(tau) / H_1``, but the wind group this factory takes
    is a *stress*. The underlying SWM profile is ``-cos(2 pi y / Ly)``,
    whose curl has amplitude ``2 pi / Ly``, so passing ``beta_hat``
    through unchanged overshot the interior velocity by that factor.
    """

    @staticmethod
    def build(**kw):
        return ReparameterizedQG.from_nondimensional(
            nx=64,
            ny=64,
            rossby=0.01,
            beta_hat=1.0,
            burger=(1.0, 0.1),
            thickness_ratio=(1.0, 3.0),
            delta_M=0.02,
            delta_S=0.0,
            check_resolution=False,
            **kw,
        )

    @staticmethod
    def sverdrup_velocity(model):
        swm = model.swm
        tau = np.asarray(swm.wind_stress_x)
        curl = -np.gradient(tau, float(swm.grid.spacing("y")), axis=0)
        # Trim the ghost ring and its one-sided differences.
        amplitude = float(np.abs(curl[3:-3]).max())
        return (
            amplitude
            * float(model.params.wind_amplitude)
            / (float(swm.strat.H[0]) * float(swm.consts.beta))
        )

    def test_the_double_gyre_default_is_order_one(self):
        model, _ = self.build()
        assert self.sverdrup_velocity(model) == pytest.approx(1.0, rel=5e-3)

    def test_the_single_gyre_default_is_too(self):
        model, _ = self.build(wind_profile="single")
        assert self.sverdrup_velocity(model) == pytest.approx(1.0, rel=5e-3)

    def test_an_explicit_wind_hat_is_a_stress_and_passes_through(self):
        """The override is the dimensionless stress group, not a curl."""
        model, _ = self.build(wind_hat=0.5)
        assert float(model.params.wind_amplitude) == pytest.approx(0.5 * 1.0)

    def test_the_aspect_ratio_enters_the_default(self):
        """``Ly = aspect``, and the curl factor is ``2 pi / Ly``."""
        square, _ = self.build()
        tall, _ = self.build(aspect=2.0)
        assert float(tall.params.wind_amplitude) == pytest.approx(
            2.0 * float(square.params.wind_amplitude)
        )
