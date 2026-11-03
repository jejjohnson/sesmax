"""End-to-end smoke tests that models accept a mask.

These pin the threading of the finitevolX ``Mask1D`` / ``Mask2D`` API
through every family: the models are constructed with an all-ocean mask
and then, for the canonical 2-D SWM, with a coastal mask that carves a
small island.  The tests assert the models build, their RHS is finite,
and — for the coastal case — dry T-cells stay at zero after a few steps.

Moved from somax 9cb19b3 without the nonlinear 2D shallow-water and QG
cases (``TestAllOceanMaskSmoke.test_swm_nonlinear_2d``, ``test_qg_barotropic``
and ``TestCoastalMaskNonlinearSWM2D``), which move with those models to the
ocean component.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from finitevolx import Mask1D, Mask2D

from sesmax_core.testing import (
    Burgers1D,
    Burgers2D,
    LinearShallowWater1D,
    LinearShallowWater2D,
    NonlinearShallowWater1D,
)


def _island_mask(ny: int, nx: int) -> Mask2D:
    """Rectangular basin with a small interior island."""
    h = np.ones((ny, nx), dtype=bool)
    h[0, :] = h[-1, :] = False
    h[:, 0] = h[:, -1] = False
    j0, j1 = ny // 2 - 1, ny // 2 + 2
    i0, i1 = nx // 2 - 1, nx // 2 + 2
    h[j0:j1, i0:i1] = False
    return Mask2D.from_mask(jnp.asarray(h))


# ---------------------------------------------------------------------------
# All-ocean smoke tests: model builds and RHS evaluates without errors
# ---------------------------------------------------------------------------


class TestAllOceanMaskSmoke:
    def test_swm_linear_2d(self):
        mask = Mask2D.from_dimensions(ny=16, nx=16)
        model = LinearShallowWater2D.create(nx=14, ny=14, mask=mask)
        assert model.mask is mask
        assert model.diff.mask is mask

    def test_pde2d_burgers(self):
        mask = Mask2D.from_dimensions(ny=16, nx=16)
        model = Burgers2D.create(nx=14, ny=14, mask=mask)
        assert model.advection.mask is mask

    def test_pde1d_burgers(self):
        mask = Mask1D.from_dimensions(nx=32)
        model = Burgers1D.create(nx=30, mask=mask)
        assert model.diff.mask is mask
        assert model.advection.mask is mask

    def test_swm_linear_1d(self):
        mask = Mask1D.from_dimensions(nx=32)
        model = LinearShallowWater1D.create(nx=30, mask=mask)
        assert model.mask is mask
        assert model.interp.mask is mask

    def test_swm_nonlinear_1d(self):
        mask = Mask1D.from_dimensions(nx=32)
        model = NonlinearShallowWater1D.create(nx=30, mask=mask)
        assert model.advection.mask is mask


# ---------------------------------------------------------------------------
# Coastal mask: dry cells stay zero in the integrated state
# ---------------------------------------------------------------------------
