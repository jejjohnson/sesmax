"""The ocean forcing presets, with somax's tests of them.

Moved from somax 9cb19b3: the preset classes of ``tests/core/test_basis.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from sesmax_core import (
    ForcingTerm,
    VectorBasisForcing,
    add_vector_to,
    control_filter,
    spatial_from_divfree,
)
from sesmax_core.domain import Domain
from sesmax_ocean.forcing import (
    geostrophic_currents,
    ssh_geostrophic,
    sss_coastal,
    sst_frontal,
)


def _domain_2d(ny: int = 5, nx: int = 7) -> Domain:
    """A small rectangular [0, 1]^2 domain with distinct axis sizes."""
    xmin, xmax = (0.0, 0.0), (1.0, 1.0)
    Nx = (ny, nx)
    dx = (1.0 / (ny - 1), 1.0 / (nx - 1))
    return Domain(xmin=xmin, xmax=xmax, dx=dx, Nx=Nx, Lx=(1.0, 1.0))


class TestSSHGeostrophicPreset:
    def test_default_static_call_shape_and_prior(self):
        from geonnax.basis import gabor_frame_grid

        domain = _domain_2d()
        bf = ssh_geostrophic(
            domain, n_scales=2, base_scale=0.25, slope=4.0, amplitude=1.0
        )
        # constant-in-time default: coeffs zero, field shaped to the grid
        np.testing.assert_array_equal(bf.coeffs, jnp.zeros_like(bf.coeffs))
        assert bf(0.0).shape == tuple(domain.Nx)
        # prior std is the bare wavenumber law (no temporal tiling)
        bounds = np.stack(
            [np.asarray(domain.xmin, float), np.asarray(domain.xmax, float)], axis=-1
        )
        _, _, _, k = gabor_frame_grid(
            domain.coords, bounds, n_scales=2, base_scale=0.25
        )
        np.testing.assert_allclose(
            bf.spatial.prior_std(), jnp.sqrt(k ** (-4.0)), atol=1e-6
        )

    def test_windowed_is_time_dependent(self):
        domain = _domain_2d()
        windows = (jnp.array([0.0, 5.0]), jnp.array([1.0, 1.0]))
        bf = ssh_geostrophic(domain, n_scales=2, base_scale=0.25, windows=windows)
        # tiling doubles the column count (2 windows)
        static = ssh_geostrophic(domain, n_scales=2, base_scale=0.25)
        m_s = static.coeffs.shape[0]
        assert bf.coeffs.shape[0] == 2 * m_s
        # Give the two temporal blocks distinct coeffs so the moving gate makes
        # the field genuinely time-dependent (equal blocks would sum the same).
        coeffs = jnp.concatenate([jnp.ones(m_s), 2.0 * jnp.ones(m_s)])
        active = eqx.tree_at(lambda f: f.coeffs, bf, coeffs)
        assert active(0.0).shape == tuple(domain.Nx)
        assert not bool(jnp.allclose(active(0.0), active(5.0)))


class TestSSSCoastalPreset:
    def test_call_shape_and_grad_to_coeffs(self):
        domain = _domain_2d()
        centers = jnp.array([[0.1, 0.1], [0.9, 0.9]])
        widths = jnp.array([0.2, 0.2])
        bf = sss_coastal(domain, centers, widths, std=2.0)
        assert bf(0.0).shape == tuple(domain.Nx)
        # gradients reach only the coefficients through control_filter
        diff, static = eqx.partition(bf, control_filter(bf))

        def loss(diff):
            forcing = eqx.combine(diff, static)
            return jnp.sum(forcing(0.0) ** 2)

        # seed non-zero coeffs so the loss has a gradient
        diff = eqx.tree_at(lambda d: d.coeffs, diff, jnp.ones_like(bf.coeffs))
        grads = jax.grad(loss)(diff)
        assert grads.spatial.Phi is None
        assert bool(jnp.any(grads.coeffs != 0.0))


class TestSSTFrontalPreset:
    def test_call_shape_and_zero_init(self):
        domain = _domain_2d(ny=8, nx=8)
        bf = sst_frontal(domain, num_basis_per_dim=4, length_scale=0.3)
        np.testing.assert_array_equal(bf.coeffs, jnp.zeros_like(bf.coeffs))
        assert bf(0.0).shape == tuple(domain.Nx)

    def test_gradient_scoped_to_coeffs(self):
        domain = _domain_2d(ny=8, nx=8)
        bf = sst_frontal(domain, num_basis_per_dim=4, length_scale=0.3)
        diff, static = eqx.partition(bf, control_filter(bf))
        diff = eqx.tree_at(lambda d: d.coeffs, diff, jnp.ones_like(bf.coeffs))

        def loss(diff):
            return jnp.sum(eqx.combine(diff, static)(0.0) ** 2)

        grads = jax.grad(loss)(diff)
        assert grads.spatial.Phi is None
        assert bool(jnp.any(grads.coeffs != 0.0))


class TestVectorDivFreeForcing:
    def test_builder_and_field_shapes(self):
        domain = _domain_2d(ny=6, nx=6)
        ngrid = domain.coords.shape[0]
        vsb = spatial_from_divfree(domain, num_basis_per_dim=3, length_scale=2.0)
        assert vsb.Phi.shape == (ngrid, 9, 2)  # (Ngrid, m, ncomp)
        assert vsb.std.shape == (9,)
        vf = geostrophic_currents(domain, num_basis_per_dim=3, length_scale=2.0)
        assert isinstance(vf, VectorBasisForcing)
        assert vf(0.0).shape == (2, *tuple(domain.Nx))  # (ncomp, Ny, Nx)

    def test_add_vector_to_places_components(self):
        domain = _domain_2d(ny=6, nx=6)
        vf = geostrophic_currents(domain, num_basis_per_dim=3, length_scale=2.0)
        vf = eqx.tree_at(lambda f: f.coeffs, vf, jnp.ones_like(vf.coeffs))

        class UVH(eqx.Module):
            u: jnp.ndarray
            v: jnp.ndarray
            h: jnp.ndarray

        zeros = jnp.zeros(tuple(domain.Nx))
        state = UVH(u=zeros, v=zeros, h=zeros)
        term = ForcingTerm(vf, place=add_vector_to(("u", "v")))
        tendency = term(0.0, state)
        field = vf(0.0)
        np.testing.assert_allclose(tendency.u, field[0], atol=1e-6)
        np.testing.assert_allclose(tendency.v, field[1], atol=1e-6)
        # the untargeted component stays zero
        assert bool((tendency.h == 0).all())

    def test_field_is_divergence_free(self):
        # The forcing the bank builds is incompressible *analytically*: the
        # continuous field reconstructed with the same centring/box mapping the
        # builder uses has zero divergence (autodiff, not finite differences).
        from geonnax.basis import divfree_basis

        domain = _domain_2d(ny=6, nx=6)
        vf = geostrophic_currents(domain, num_basis_per_dim=3, length_scale=2.0)
        coeffs = jnp.ones_like(vf.coeffs)
        xmin, xmax = jnp.asarray(domain.xmin), jnp.asarray(domain.xmax)
        centre = 0.5 * (xmin + xmax)
        L = tuple(float(h) for h in 0.5 * (xmax - xmin))

        def velocity(p):
            phi, _ = divfree_basis((p - centre)[None, :], 3, L)  # (1, m, 2)
            return jnp.einsum("gmc,m->gc", phi, coeffs)[0]  # (u, v)

        for p in jnp.array([[3.0, 7.0], [5.0, 5.0], [1.0, 9.0]]):
            jac = jax.jacfwd(velocity)(p)  # (2, 2): d (u, v)_i / d (x, y)_k
            div = jac[0, 0] + jac[1, 1]
            np.testing.assert_allclose(div, 0.0, atol=1e-5)

    def test_whiten_and_regularization(self):
        domain = _domain_2d(ny=6, nx=6)
        vf = geostrophic_currents(domain, num_basis_per_dim=3, length_scale=2.0)
        u = jax.random.normal(jax.random.PRNGKey(11), vf.coeffs.shape)
        whitened = vf.whiten(u)
        np.testing.assert_allclose(
            whitened.coeffs, vf.spatial.prior_std() * u, atol=1e-6
        )
        expected = 0.5 * jnp.sum((whitened.coeffs / vf.spatial.prior_std()) ** 2)
        np.testing.assert_allclose(whitened.regularization(), expected, atol=1e-5)

    def test_gradient_scoped_to_coeffs(self):
        domain = _domain_2d(ny=6, nx=6)
        vf = geostrophic_currents(domain, num_basis_per_dim=3, length_scale=2.0)
        diff, static = eqx.partition(vf, control_filter(vf))
        diff = eqx.tree_at(lambda d: d.coeffs, diff, jnp.ones_like(vf.coeffs))

        def loss(diff):
            return jnp.sum(eqx.combine(diff, static)(0.0) ** 2)

        grads = jax.grad(loss)(diff)
        assert grads.spatial.Phi is None
        assert bool(jnp.any(grads.coeffs != 0.0))
