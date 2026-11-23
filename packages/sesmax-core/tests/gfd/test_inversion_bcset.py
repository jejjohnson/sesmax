"""A ``gridax.BC`` as ``bc=`` of the elliptic inversions.

Moved from gridax tests/test_elliptic_bcset.py at gridax 01d8320 (sesmax#126). Only the
imports changed: the functions come from ``sesmax_core.gfd`` (their calls were already
keyword-only past ``dy``) and the rest from public ``gridax`` and ``gridax.solvers``
names.
The mask is an argument of the solve: ``BC`` holds conditions only. In
``test_neumann_array_zero_is_dct`` the zero face values are built inside the test,
because the x64 fixture is not on while the module is collected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gridax import BC, Dirichlet, Mask2D, Neumann, Outflow, Periodic
from gridax.solvers import coast_ring, masked_laplacian

from sesmax_core.gfd import (
    pressure_from_divergence,
    pv_inversion,
    streamfunction_from_vorticity,
)


NY, NX = 20, 24
DX, DY = 0.05, 0.04
CG_TOL = 1e-6


def _rhs() -> jnp.ndarray:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.standard_normal((NY, NX)))


def _basin_mask() -> jnp.ndarray:
    # mask[j, i] = 1 for 1 <= j <= NY-2, 1 <= i <= NX-2
    return jnp.zeros((NY, NX)).at[1:-1, 1:-1].set(1.0)


def _basin_mask2d() -> Mask2D:
    """Mask2D of the rectangle: psi (X) wet on [1:-2, 1:-2], walls at 0 and N-2."""
    return Mask2D.from_mask(np.asarray(_basin_mask(), dtype=bool))


def _laplacian(psi: jnp.ndarray, lambda_: float) -> jnp.ndarray:
    """Unmasked 5-point (A - lambda) psi: reads the wall values as they are."""
    return masked_laplacian(psi, jnp.ones_like(psi), DX, DY, lambda_=lambda_)


def _dirichlet(south=0.0, north=0.0, west=0.0, east=0.0):
    return BC(
        x=(Dirichlet(west), Dirichlet(east)), y=(Dirichlet(south), Dirichlet(north))
    )


def _neumann(value=0.0):
    return BC(x=Neumann(value), y=Neumann(value))


class TestTransformDispatch:
    @pytest.mark.slow
    def test_closed_is_dst(self):
        rhs = _rhs()
        psi = streamfunction_from_vorticity(rhs, DX, DY, bc=_dirichlet(), lambda_=2.0)
        ref = streamfunction_from_vorticity(rhs, DX, DY, bc="dst", lambda_=2.0)
        np.testing.assert_array_equal(psi, ref)

    @pytest.mark.slow
    def test_zero_neumann_is_dct(self):
        rhs = _rhs() - jnp.mean(_rhs())
        p = pressure_from_divergence(rhs, DX, DY, bc=_neumann())
        ref = pressure_from_divergence(rhs, DX, DY, bc="dct")
        np.testing.assert_array_equal(p, ref)

    def test_periodic_is_fft(self):
        rhs = _rhs()
        psi = streamfunction_from_vorticity(
            rhs, DX, DY, bc=BC(x=Periodic(), y=Periodic()), lambda_=3.0
        )
        ref = streamfunction_from_vorticity(rhs, DX, DY, bc="fft", lambda_=3.0)
        np.testing.assert_array_equal(psi, ref)

    @pytest.mark.slow
    def test_pv_inversion_array_lambda(self):
        pv = jnp.stack([_rhs(), 2.0 * _rhs()])
        lambdas = jnp.array([1.0, 4.0])
        psi = pv_inversion(pv, DX, DY, lambdas, bc=_dirichlet())
        ref = pv_inversion(pv, DX, DY, lambdas, bc="dst")
        np.testing.assert_array_equal(psi, ref)


class TestMask:
    @pytest.mark.slow
    def test_cg_takes_the_mask_beside_the_bc(self):
        mask = _basin_mask()
        rhs = _rhs() * mask
        psi = streamfunction_from_vorticity(
            rhs, DX, DY, bc=_dirichlet(), method="cg", mask=mask
        )
        ref = streamfunction_from_vorticity(rhs, DX, DY, method="cg", mask=mask)
        np.testing.assert_allclose(psi, ref, atol=1e-12)


class TestInhomogeneousDirichletFaces:
    @pytest.mark.slow
    def test_face_values_pin_wall_rows(self):
        bc = _dirichlet(south=-0.2, north=0.1, west=0.05, east=0.3)
        psi = streamfunction_from_vorticity(_rhs(), DX, DY, bc=bc, lambda_=4.0)
        # The values sit on the psi wall nodes (the coast of the X basin
        # [1:-2, 1:-2]); the corner nodes are not on the coast.
        # psi[0, i]    = south  for 1 <= i <= NX-3
        np.testing.assert_allclose(psi[0, 1:-2], -0.2, atol=1e-12)
        # psi[NY-2, i] = north  for 1 <= i <= NX-3
        np.testing.assert_allclose(psi[-2, 1:-2], 0.1, atol=1e-12)
        # psi[j, 0]    = west   for 1 <= j <= NY-3
        np.testing.assert_allclose(psi[1:-2, 0], 0.05, atol=1e-12)
        # psi[j, NX-2] = east   for 1 <= j <= NY-3
        np.testing.assert_allclose(psi[1:-2, -2], 0.3, atol=1e-12)

    def test_matches_explicit_known_values(self):
        bc = _dirichlet(north=0.1)
        # g[NY-2, i] = 0.1 for 1 <= i <= NX-3  (the north wall nodes)
        g = jnp.zeros((NY, NX)).at[-2, 1:-2].set(0.1)
        rhs = _rhs()
        psi = streamfunction_from_vorticity(rhs, DX, DY, bc=bc)
        ref = streamfunction_from_vorticity(rhs, DX, DY, known_values=g)
        np.testing.assert_array_equal(psi, ref)

    def test_cg_with_a_mask_matches_spectral(self):
        mask = _basin_mask2d()
        rhs = _rhs()
        psi_cg = streamfunction_from_vorticity(
            rhs, DX, DY, bc=_dirichlet(north=0.1), method="cg", mask=mask
        )
        psi_sp = streamfunction_from_vorticity(rhs, DX, DY, bc=_dirichlet(north=0.1))
        np.testing.assert_allclose(psi_cg, psi_sp, atol=CG_TOL)

    def test_known_values_take_precedence(self):
        rhs = _rhs()
        g = jnp.full((NY, NX), 0.7)
        psi = streamfunction_from_vorticity(
            rhs, DX, DY, bc=_dirichlet(north=0.1), known_values=g
        )
        ref = streamfunction_from_vorticity(rhs, DX, DY, known_values=g)
        np.testing.assert_array_equal(psi, ref)

    @pytest.mark.slow
    def test_pv_inversion_face_values(self):
        pv = jnp.stack([_rhs(), _rhs()])
        psi = pv_inversion(pv, DX, DY, jnp.array([1.0, 9.0]), bc=_dirichlet(east=0.25))
        # psi[k, j, NX-2] = east for every layer k and 1 <= j <= NY-3
        np.testing.assert_allclose(psi[:, 1:-2, -2], 0.25, atol=1e-12)

    @pytest.mark.parametrize("method", ["spectral", "cg"])
    def test_all_four_faces_on_mask2d_wall(self, method):
        """#303: every face value lands on its wall nodes, none is dropped.

        Before, north / east were written on row / column N-2, which is dry
        in ``xy_corner_strict``, and silently dropped by the lifting.
        """
        mask = _basin_mask2d()
        wet = jnp.asarray(mask.wet(at="X", kind="strict"))
        lam = 4.0
        faces = {"south": -0.2, "north": 1.0, "west": 0.5, "east": 2.0}
        kw = {"mask": mask} if method == "cg" else {}
        bc = _dirichlet(**faces)
        rhs = _rhs()
        psi = streamfunction_from_vorticity(
            rhs, DX, DY, bc=bc, lambda_=lam, method=method, **kw
        )
        # The coast ring carries the requested values exactly (set by the lift).
        np.testing.assert_allclose(psi[0, 1:-2], faces["south"], atol=1e-12)
        np.testing.assert_allclose(psi[-2, 1:-2], faces["north"], atol=1e-12)
        np.testing.assert_allclose(psi[1:-2, 0], faces["west"], atol=1e-12)
        np.testing.assert_allclose(psi[1:-2, -2], faces["east"], atol=1e-12)
        # PDE residual on the solve domain (every wet node), the stencil
        # reading the wall values: exact for the spectral solve (round-off),
        # CG's default rtol / atol (1e-6) for cg.
        res = jnp.where(wet, _laplacian(psi, lam) - rhs, 0.0)
        tol = 1e-10 if method == "spectral" else CG_TOL * 10
        assert float(jnp.max(jnp.abs(res))) < tol

    def test_pressure_rejects_nonzero_dirichlet_faces(self):
        """T-point Dirichlet sits on the faces; the lift cannot put it there."""
        with pytest.raises(ValueError, match="T-point pressure"):
            pressure_from_divergence(_rhs(), DX, DY, bc=_dirichlet(north=1.0))

    def test_curved_coast_raises(self):
        """Face values on a non-rectangular coast would mostly land on no face."""
        yy, xx = np.mgrid[0:NY, 0:NX]
        m = (xx - NX / 2) ** 2 / (0.4 * NX) ** 2 + (yy - NY / 2) ** 2 / (
            0.4 * NY
        ) ** 2 < 1
        mask = Mask2D.from_mask(m)
        with pytest.raises(ValueError, match="outer coast nodes are on no face"):
            streamfunction_from_vorticity(
                jnp.zeros((NY, NX)),
                DX,
                DY,
                bc=_dirichlet(1.0, 1.0, 1.0, 1.0),
                method="cg",
                mask=mask,
            )

    def test_island_in_rectangle_is_accepted(self):
        """Islands do not touch the outer land: their coast stays at zero."""
        m = np.asarray(_basin_mask(), dtype=bool).copy()
        m[8:12, 10:14] = False
        mask = Mask2D.from_mask(m)
        psi = streamfunction_from_vorticity(
            jnp.zeros((NY, NX)),
            DX,
            DY,
            bc=_dirichlet(1.0, 1.0, 1.0, 1.0),
            method="cg",
            mask=mask,
        )
        wet = np.asarray(mask.wet(at="X", kind="strict")) > 0.5
        coast = np.asarray(coast_ring(wet))
        island_coast = coast.copy()
        island_coast[[0, NY - 2], :] = False
        island_coast[:, [0, NX - 2]] = False
        assert island_coast.any()
        np.testing.assert_array_equal(np.asarray(psi)[island_coast], 0.0)
        np.testing.assert_allclose(np.asarray(psi)[coast & ~island_coast], 1.0)


class TestErrors:
    def test_mixed_face_types(self):
        bc = BC(x=Periodic(), y=Dirichlet(0.0))
        with pytest.raises(ValueError, match="no mixing"):
            streamfunction_from_vorticity(_rhs(), DX, DY, bc=bc)

    def test_missing_face(self):
        bc = BC(south=Dirichlet(0.0))
        with pytest.raises(ValueError, match="all four faces"):
            streamfunction_from_vorticity(_rhs(), DX, DY, bc=bc)

    def test_unsupported_face_type(self):
        with pytest.raises(ValueError, match="Outflow"):
            streamfunction_from_vorticity(
                _rhs(), DX, DY, bc=BC(x=Outflow(), y=Outflow())
            )

    def test_inhomogeneous_neumann(self):
        with pytest.raises(ValueError, match="Inhomogeneous Neumann"):
            pressure_from_divergence(_rhs(), DX, DY, bc=_neumann(0.5))


class TestArrayZeroFaceValues:
    """Concrete array zeros count as homogeneous face values (#310)."""

    @pytest.mark.parametrize(
        "make_zero",
        [
            lambda: jnp.array(0.0),
            lambda: jnp.float64(0.0),
            lambda: np.zeros(()),
            lambda: np.float64(0.0),
        ],
        ids=["jnp.array", "jnp.float64", "np.zeros", "np.float64"],
    )
    def test_neumann_array_zero_is_dct(self, make_zero):
        # Built in the test, not at collection: the module's x64 is on only here.
        zero = make_zero()
        rhs = _rhs() - jnp.mean(_rhs())
        p = pressure_from_divergence(rhs, DX, DY, bc=_neumann(zero))
        ref = pressure_from_divergence(rhs, DX, DY, bc="dct")
        np.testing.assert_array_equal(p, ref)

    @pytest.mark.slow
    def test_neumann_array_zero_cg_on_mask(self):
        rhs = _rhs()
        bc = _neumann(jnp.array(0.0))
        p = pressure_from_divergence(
            rhs, DX, DY, bc=bc, method="cg", mask=_basin_mask2d()
        )
        ref = pressure_from_divergence(
            rhs, DX, DY, bc="dct", method="cg", mask=_basin_mask2d()
        )
        np.testing.assert_array_equal(p, ref)

    def test_dirichlet_array_zero_is_homogeneous(self):
        """A zero jnp face value must not switch on the known-value lift."""
        z = jnp.array(0.0)
        psi = streamfunction_from_vorticity(
            _rhs(), DX, DY, bc=_dirichlet(z, z, z, z), lambda_=2.0
        )
        ref = streamfunction_from_vorticity(_rhs(), DX, DY, bc="dst", lambda_=2.0)
        np.testing.assert_array_equal(psi, ref)

    def test_nonzero_array_still_raises(self):
        with pytest.raises(ValueError, match="Inhomogeneous Neumann"):
            pressure_from_divergence(_rhs(), DX, DY, bc=_neumann(jnp.array(0.5)))

    def test_traced_value_counts_as_nonzero(self):
        """A tracer cannot be inspected, so it is refused rather than assumed zero."""

        def solve(value):
            return pressure_from_divergence(_rhs(), DX, DY, bc=_neumann(value))

        with pytest.raises(ValueError, match="Inhomogeneous Neumann"):
            jax.jit(solve)(jnp.array(0.0))
