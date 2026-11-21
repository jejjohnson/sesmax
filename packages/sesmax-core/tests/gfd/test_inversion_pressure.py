"""The masked pressure solve honours its boundary condition; a traced lambda works.

Moved from gridax tests/test_pressure_cg.py at gridax 01d8320 (sesmax#126).
``pressure_from_divergence(method="cg")`` imposes the Neumann or Dirichlet wall of
``bc`` on the T faces and rejects the combinations it cannot honour, and
``pv_inversion`` accepts a traced scalar ``lambda_`` under ``jit`` and ``vmap``.

What changed: the imports (the functions from ``sesmax_core.gfd``, the grid and mask
from ``gridax``, the T-point Neumann operator from ``gridax_jax``'s private iterative
module). Dropped: ``TestNeumannOperator``, which tests gridax's private T-point
operators and stays there, and ``test_pressure_rejects_known_values``, which only
checked that ``pressure_from_divergence`` rejects ``known_values=`` and
``known_mask=``, parameters it no longer has.
``test_streamfunction_known_values_reject_neumann`` is kept.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gridax import CartesianGrid2D, Mask2D
from gridax_jax._src.solvers.iterative import _masked_neumann_laplacian

from sesmax_core.gfd import (
    pressure_from_divergence,
    pv_inversion,
    streamfunction_from_vorticity,
)


N = 34


def _grid():
    g = CartesianGrid2D.from_interior(N - 2, N - 2, Lx=1.0, Ly=1.3)
    return g.dx, g.dy


def _rect_mask():
    m = np.zeros((N, N), dtype=bool)
    m[1:-1, 1:-1] = True
    return Mask2D.from_mask(m)


def _island_basin():
    """Elliptic basin with a 4x4 island, dry ghost ring: one wet component."""
    yy, xx = np.mgrid[0:N, 0:N]
    m = (xx - N / 2) ** 2 / (0.42 * N) ** 2 + (yy - N / 2) ** 2 / (0.4 * N) ** 2 < 1
    m[14:18, 15:19] = False
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return Mask2D.from_mask(m)


def _rhs(seed=0):
    return jnp.asarray(np.random.default_rng(seed).standard_normal((N, N)))


def _demeaned(field, wet):
    vals = np.asarray(field)[wet]
    return vals - vals.mean()


@pytest.mark.slow  # jit-compile bound (> 1 s each)
class TestPressureCG:
    def test_agrees_with_spectral_dct_on_rectangle(self):
        dx, dy = _grid()
        mask = _rect_mask()
        wet = np.asarray(mask.h) > 0.5
        b = _rhs()
        p_sp = pressure_from_divergence(b, dx, dy, bc="dct")
        p_cg = pressure_from_divergence(b, dx, dy, method="cg", mask=mask)
        # CG stops at rtol 1e-6 of |b|; |p| ~ 1e-2 here, so 1e-8 is that tolerance
        np.testing.assert_allclose(
            _demeaned(p_cg, wet), _demeaned(p_sp, wet), atol=1e-8
        )

    def test_neumann_on_masked_basin(self):
        """Zero flux through every coast face: the solution satisfies the
        Neumann operator with the (projected) rhs and has zero wet-mean."""
        dx, dy = _grid()
        mask = _island_basin()
        wet = np.asarray(mask.h) > 0.5
        b = _rhs(1)
        p = pressure_from_divergence(b, dx, dy, method="cg", mask=mask)
        assert abs(np.asarray(p)[wet].mean()) < 1e-14
        np.testing.assert_array_equal(np.asarray(p)[~wet], 0.0)
        lap = _masked_neumann_laplacian(p, mask.h.astype(b.dtype), dx, dy)
        # residual of the CG solve: rtol 1e-6 of |b| ~ 3
        np.testing.assert_allclose(np.asarray(lap)[wet], _demeaned(b, wet), atol=1e-5)

    def test_two_components_each_zero_mean(self):
        """Disconnected wet regions each fix p up to their own constant."""
        dx, dy = _grid()
        m = np.zeros((N, N), dtype=bool)
        m[1:-1, 1:-1] = True
        m[:, N // 2] = False  # a dry wall splits the basin in two
        mask = Mask2D.from_mask(m)
        wet = np.asarray(mask.h) > 0.5
        p = pressure_from_divergence(_rhs(11), dx, dy, method="cg", mask=mask)
        left, right = np.asarray(p)[:, : N // 2], np.asarray(p)[:, N // 2 + 1 :]
        assert abs(left[wet[:, : N // 2]].mean()) < 1e-14
        assert abs(right[wet[:, N // 2 + 1 :]].mean()) < 1e-14

    def test_dirichlet_agrees_with_spectral_dst_on_rectangle(self):
        """bc="dst": CG puts p = 0 on the faces, where DST-II does."""
        dx, dy = _grid()
        mask = _rect_mask()
        b = _rhs(12)
        p_sp = pressure_from_divergence(b, dx, dy, bc="dst")
        p_cg = pressure_from_divergence(b, dx, dy, bc="dst", method="cg", mask=mask)
        # CG rtol 1e-6 of |b|; |p| ~ 1e-2 here
        np.testing.assert_allclose(np.asarray(p_cg), np.asarray(p_sp), atol=1e-8)

    def test_dirichlet_is_second_order(self):
        """Manufactured p = sin(pi x) sin(pi y), zero on the faces x, y in {0, 1}."""
        errs = []
        for n in (16, 32):
            g = CartesianGrid2D.from_interior(n, n, Lx=1.0, Ly=1.0)
            x = (np.arange(n + 2) - 0.5) * g.dx  # T centres; faces at 0 and 1
            xx, yy = np.meshgrid(x, x)
            exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
            m = np.zeros((n + 2, n + 2), dtype=bool)
            m[1:-1, 1:-1] = True
            p = pressure_from_divergence(
                jnp.asarray(-2 * np.pi**2 * exact),
                g.dx,
                g.dx,
                bc="dst",
                method="cg",
                mask=Mask2D.from_mask(m),
            )
            errs.append(np.abs(np.asarray(p) - exact)[m].max())
        assert np.log2(errs[0] / errs[1]) > 1.9

    def test_differs_from_the_dirichlet_solve(self):
        """bc decides the operator: Neumann and Dirichlet give different p."""
        dx, dy = _grid()
        mask = _island_basin()
        b = _rhs(2)
        p_n = pressure_from_divergence(b, dx, dy, method="cg", mask=mask)
        p_d = pressure_from_divergence(b, dx, dy, bc="dst", method="cg", mask=mask)
        assert float(jnp.max(jnp.abs(p_n - p_d))) > 1e-3


class TestBcIsHonouredOrRejected:
    def test_cg_rejects_periodic(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="periodic 'fft'"):
            pressure_from_divergence(
                _rhs(), dx, dy, bc="fft", method="cg", mask=_rect_mask()
            )

    def test_cg_streamfunction_rejects_neumann(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="only defined for the T-point"):
            streamfunction_from_vorticity(
                _rhs(), dx, dy, bc="dct", method="cg", mask=_rect_mask()
            )

    def test_pressure_rejects_capacitance(self):
        """Its wall is the psi wall (dry cell centres), not the T faces."""
        dx, dy = _grid()
        with pytest.raises(ValueError, match="not available for the T-point"):
            pressure_from_divergence(_rhs(), dx, dy, bc="dst", method="capacitance")

    def test_streamfunction_capacitance_rejects_neumann(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="needs bc='dst'"):
            streamfunction_from_vorticity(
                _rhs(), dx, dy, bc="dct", method="capacitance"
            )

    def test_streamfunction_known_values_reject_neumann(self):
        dx, dy = _grid()
        with pytest.raises(ValueError, match="known_values prescribe Dirichlet"):
            streamfunction_from_vorticity(
                _rhs(),
                dx,
                dy,
                bc="dct",
                method="cg",
                mask=_rect_mask(),
                known_values=_rhs(3),
            )


@pytest.mark.slow  # jit-compile bound (> 1 s each)
class TestTracedScalarLambda:
    """#309: a traced scalar lambda_ works and changes nothing."""

    @pytest.mark.parametrize("method", ["spectral", "cg"])
    def test_jit(self, method):
        dx, dy = _grid()
        kw = {} if method == "spectral" else {"mask": _island_basin()}
        rhs = jnp.stack([_rhs(4), _rhs(5)])
        eager = pv_inversion(rhs, dx, dy, lambda_=4.0, method=method, **kw)
        jitted = jax.jit(
            lambda r, lam: pv_inversion(r, dx, dy, lambda_=lam, method=method, **kw)
        )(rhs, 4.0)
        np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))

    def test_vmap(self):
        dx, dy = _grid()
        rhs = _rhs(6)
        lams = jnp.array([1.0, 4.0])
        out = jax.vmap(lambda lam: pv_inversion(rhs, dx, dy, lambda_=lam))(lams)
        for k, lam in enumerate([1.0, 4.0]):
            np.testing.assert_array_equal(
                np.asarray(out[k]),
                np.asarray(pv_inversion(rhs, dx, dy, lambda_=lam)),
            )

    def test_scalar_array_and_python_float_agree(self):
        dx, dy = _grid()
        rhs = jnp.stack([_rhs(7), _rhs(8)])
        a = pv_inversion(rhs, dx, dy, lambda_=4.0)
        b = pv_inversion(rhs, dx, dy, lambda_=jnp.asarray(4.0))
        c = pv_inversion(rhs, dx, dy, lambda_=jnp.array([4.0, 4.0]))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(np.asarray(a), np.asarray(c), rtol=1e-13, atol=1e-16)
