"""Tests for the multilayer() vmap helper.

Covers:
- Output shape and per-layer correctness.
- Equivalence with a 3-D grid in the horizontal interior.
- The key difference: multilayer writes to ALL layers (no ghost z-layers),
  whereas a 3-D grid only writes to k=1..Nz-2 and leaves k=0, k=Nz-1 zero.
"""

from __future__ import annotations

import gridax as fvx
import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core.vertical import multilayer


@pytest.fixture
def grid2d():
    return fvx.CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)


@pytest.fixture
def grid3d():
    # 4 interior z-levels → Nz = 6 total (including ghost layers at k=0, k=5)
    return fvx.CartesianCGrid3D.from_interior(8, 8, 4, 1.0, 1.0, 1.0)


def difference(grid, axis):
    """``gridax.difference`` along an axis, from T to the faces."""
    dst = {"x": "U", "y": "V"}[axis]
    return lambda h: fvx.difference(h, grid, axis=axis, src="T", dst=dst)


class TestMultilayerOutputShape:
    def test_single_arg_shape(self, grid2d):
        """multilayer wraps vmap: [nl, Ny, Nx] -> [nl, Ny, Nx]."""
        nl = 3
        h = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        result = multilayer(difference(grid2d, "x"))(h)
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)

    def test_multi_arg_shape(self, grid2d):
        """multilayer handles multi-argument operators passed directly."""
        nl = 5
        u = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        v = jnp.ones((nl, grid2d.Ny, grid2d.Nx))
        result = multilayer(fvx.GridOps(grid2d).divergence)(u, v)
        assert result.shape == (nl, grid2d.Ny, grid2d.Nx)


class TestMultilayerPerLayerEquivalence:
    def test_each_layer_matches_2d_call(self, grid2d):
        """Each layer in multilayer output matches a direct 2D call."""
        nl = 4
        # Each layer is a distinct multiple of the base ramp so that a bug that
        # mixes up layer indices would produce wrong values for at least one k.
        h = jnp.stack(
            [
                jnp.arange(grid2d.Ny * grid2d.Nx, dtype=float).reshape(
                    grid2d.Ny, grid2d.Nx
                )
                * (k + 1)
                for k in range(nl)
            ]
        )
        result_ml = multilayer(difference(grid2d, "x"))(h)
        for k in range(nl):
            result_2d = difference(grid2d, "x")(h[k])
            np.testing.assert_array_equal(result_ml[k], result_2d)

    def test_linear_field_all_layers(self, grid2d):
        """Interior of every layer recovers the correct finite difference."""
        c = 2.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.spacing("x")
        layer = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        nl = 3
        h = jnp.stack([layer] * nl)
        result = multilayer(difference(grid2d, "x"))(h)
        # Interior of every layer should be ~c (constant gradient)
        for k in range(nl):
            np.testing.assert_allclose(result[k, 1:-1, 1:-1], c, rtol=1e-5)


class TestMultilayerVs3DGrid:
    """Compare multilayer(vmap) with a 3-D grid to show interior agreement
    and boundary difference.

    For a field of shape [Nz, Ny, Nx] with no special z-ghost treatment:

    - The horizontal interior [k, 1:-1, 1:-1] is IDENTICAL for both approaches
      when k is an interior z-level (k=1..Nz-2).
    - a 3-D grid leaves k=0 and k=Nz-1 as zero (ghost z-layers).
    - multilayer(fn) writes to ALL layers including k=0 and k=Nz-1.
    """

    def test_interior_agreement_diff_x(self, grid2d, grid3d):
        """Interior [k=1..Nz-2, 1:-1, 1:-1] matches between vmap and 3D."""
        Nz = grid3d.Nz  # total z-size including ghost layers (= 6 for interior 4)

        rng = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
            Nz, grid3d.Ny, grid3d.Nx
        )

        ml_result = multilayer(difference(grid2d, "x"))(rng)
        d3_result = difference(grid3d, "x")(rng)

        # Interior z-levels and horizontal interior match
        np.testing.assert_allclose(
            ml_result[1:-1, 1:-1, 1:-1],
            d3_result[1:-1, 1:-1, 1:-1],
            rtol=1e-6,
        )

    def test_interior_agreement_diff_y(self, grid2d, grid3d):
        """Interior [k=1..Nz-2, 1:-1, 1:-1] matches between vmap and 3D (y)."""
        Nz = grid3d.Nz

        rng = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
            Nz, grid3d.Ny, grid3d.Nx
        )

        ml_result = multilayer(difference(grid2d, "y"))(rng)
        d3_result = difference(grid3d, "y")(rng)

        np.testing.assert_allclose(
            ml_result[1:-1, 1:-1, 1:-1],
            d3_result[1:-1, 1:-1, 1:-1],
            rtol=1e-6,
        )

    def test_diff3d_ghost_z_layers_are_zero(self, grid2d, grid3d):
        """a 3-D grid leaves k=0 and k=Nz-1 as zero (ghost z-shells)."""
        Nz = grid3d.Nz

        rng = jnp.arange(Nz * grid3d.Ny * grid3d.Nx, dtype=float).reshape(
            Nz, grid3d.Ny, grid3d.Nx
        )

        d3_result = difference(grid3d, "x")(rng)

        # Top and bottom z-shells are zero by the 3D ghost-layer convention
        np.testing.assert_array_equal(d3_result[0], 0.0)
        np.testing.assert_array_equal(d3_result[-1], 0.0)

    def test_multilayer_all_z_levels_written(self, grid2d, grid3d):
        """multilayer writes to ALL layers — there is no ghost z-layer concept."""
        Nz = grid3d.Nz

        # Use a field that produces non-zero differences everywhere
        c = 1.0
        x = jnp.arange(grid2d.Nx, dtype=float) * grid2d.spacing("x")
        layer = jnp.broadcast_to(c * x, (grid2d.Ny, grid2d.Nx))
        h = jnp.stack([layer] * Nz)  # [Nz, Ny, Nx]

        ml_result = multilayer(difference(grid2d, "x"))(h)

        # k=0 and k=Nz-1 are real layers: interior values are non-zero
        np.testing.assert_allclose(ml_result[0, 1:-1, 1:-1], c, rtol=1e-5)
        np.testing.assert_allclose(ml_result[-1, 1:-1, 1:-1], c, rtol=1e-5)
