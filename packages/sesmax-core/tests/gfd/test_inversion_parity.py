"""The moved inversions return gridax-jax's arrays bit for bit (sesmax#126).

gridax-jax (pinned at gridax ``01d8320``) still has the originals, so each
case runs both and compares with ``np.array_equal``: the spectral path, CG
(now gaussx's ``linear_solve_with_info`` in place of gridax-jax's
``solve_cg``), the capacitance correction, known values, boundary-condition
sets and the multilayer PV inversion. gridax#475 removes the originals once
this passes. The last tests check ``method="multigrid"``, which is new,
against ``method="cg"``.
"""

from __future__ import annotations

import gridax_jax as gj
import jax.numpy as jnp
import numpy as np
import pytest
from gridax import BoundaryConditionSet, Dirichlet1D, Mask2D

from sesmax_core import gfd


N = 34  # a 32 x 32 interior plus the ghost ring
DX, DY = 1.0 / 32, 1.5 / 32


def _basin() -> Mask2D:
    # a rectangle with an island and a missing corner
    h = np.zeros((N, N), dtype=bool)
    h[1:-1, 1:-1] = True
    h[9:15, 11:18] = False
    h[22:, 24:] = False
    return Mask2D.from_mask(h)


def _field(dtype=jnp.float64, seed=0, shape=(N, N)):
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape), dtype)


def _same(new, old) -> None:
    assert new.dtype == old.dtype
    assert np.array_equal(np.asarray(new), np.asarray(old))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("bc", ["dst", "dct", "fft"])
@pytest.mark.parametrize("lambda_", [0.0, 4.0])
def test_streamfunction_spectral(bc, lambda_, dtype):
    zeta = _field(dtype)
    new = gfd.streamfunction_from_vorticity(zeta, DX, DY, bc=bc, lambda_=lambda_)
    old = gj.streamfunction_from_vorticity(zeta, DX, DY, bc=bc, lambda_=lambda_)
    _same(new, old)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("lambda_", [0.0, 4.0])
def test_streamfunction_cg(lambda_, dtype):
    zeta, mask = _field(dtype), _basin()
    kwargs = {"lambda_": lambda_, "method": "cg", "mask": mask}
    new = gfd.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    old = gj.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    _same(new, old)


@pytest.mark.parametrize("lambda_", [0.0, 4.0])
def test_streamfunction_capacitance(lambda_):
    zeta, mask = _field(), _basin()
    solver = gj.build_capacitance_solver(mask, DX, DY, lambda_=lambda_)
    kwargs = {
        "lambda_": lambda_,
        "method": "capacitance",
        "capacitance_solver": solver,
    }
    new = gfd.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    old = gj.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    _same(new, old)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize(
    ("bc", "method"),
    [
        ("dct", "spectral"),
        ("dst", "spectral"),
        ("fft", "spectral"),
        ("dct", "cg"),
        ("dst", "cg"),
    ],
)
def test_pressure(bc, method, dtype):
    div = _field(dtype)
    mask = _basin() if method == "cg" else None
    new = gfd.pressure_from_divergence(div, DX, DY, bc=bc, method=method, mask=mask)
    old = gj.pressure_from_divergence(div, DX, DY, bc=bc, method=method, mask=mask)
    _same(new, old)


@pytest.mark.parametrize("method", ["spectral", "cg", "capacitance"])
def test_known_values(method):
    zeta = _field()
    known = _field(seed=1)
    mask = None if method == "spectral" else _basin()
    kwargs = {"lambda_": 2.0, "method": method, "mask": mask, "known_values": known}
    if method == "capacitance":
        kwargs["capacitance_solver"] = gj.build_capacitance_solver(
            mask, DX, DY, lambda_=2.0
        )
    new = gfd.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    old = gj.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    _same(new, old)


def test_known_mask_with_cg():
    zeta, known, mask = _field(), _field(seed=1), _basin()
    known_mask = jnp.zeros((N, N), bool).at[5, 5].set(True).at[20, 8].set(True)
    kwargs = {
        "method": "cg",
        "mask": mask,
        "known_values": known,
        "known_mask": known_mask,
    }
    new = gfd.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    old = gj.streamfunction_from_vorticity(zeta, DX, DY, **kwargs)
    _same(new, old)


def test_boundary_condition_set_with_face_values():
    zeta = _field()
    bc = BoundaryConditionSet(
        south=Dirichlet1D("south", 1.0),
        north=Dirichlet1D("north", -0.5),
        west=Dirichlet1D("west", 0.25),
        east=Dirichlet1D("east", 0.0),
    )
    new = gfd.streamfunction_from_vorticity(zeta, DX, DY, bc=bc)
    old = gj.streamfunction_from_vorticity(zeta, DX, DY, bc=bc)
    _same(new, old)


@pytest.mark.parametrize("method", ["spectral", "cg"])
@pytest.mark.parametrize("with_known", [False, True])
def test_pv_inversion_array_lambda(method, with_known):
    q = _field(shape=(2, 3, N, N))
    lam = jnp.array([0.0, 5.0, 40.0])
    kwargs = {"method": method, "mask": None if method == "spectral" else _basin()}
    if with_known:
        kwargs["known_values"] = _field(seed=1, shape=(3, N, N))
    new = gfd.pv_inversion(q, DX, DY, lam, **kwargs)
    old = gj.pv_inversion(q, DX, DY, lam, **kwargs)
    _same(new, old)


@pytest.mark.parametrize("method", ["spectral", "cg", "capacitance"])
def test_pv_inversion_scalar_lambda_batched(method):
    q, mask = _field(shape=(3, N, N)), _basin()
    kwargs = {"method": method, "mask": None if method == "spectral" else mask}
    if method == "capacitance":
        kwargs["capacitance_solver"] = gj.build_capacitance_solver(
            mask, DX, DY, lambda_=9.0
        )
    new = gfd.pv_inversion(q, DX, DY, 9.0, **kwargs)
    old = gj.pv_inversion(q, DX, DY, 9.0, **kwargs)
    _same(new, old)


# --- method="multigrid", which gridax-jax's wrappers do not have ------------


@pytest.mark.parametrize("lambda_", [0.0, 4.0])
def test_multigrid_matches_cg_on_a_masked_basin(lambda_):
    zeta, mask = _field(), _basin()
    solver = gj.build_multigrid_solver(mask, DX, DY, lambda_=lambda_)
    mg = gfd.streamfunction_from_vorticity(
        zeta,
        DX,
        DY,
        lambda_=lambda_,
        method="multigrid",
        mask=mask,
        multigrid_solver=solver,
    )
    cg = gfd.streamfunction_from_vorticity(
        zeta, DX, DY, lambda_=lambda_, method="cg", mask=mask
    )
    # both stop at |r| <= 1e-6 + 1e-6 |b|; the solutions agree to that over A
    scale = float(jnp.max(jnp.abs(cg)))
    assert float(jnp.max(jnp.abs(mg - cg))) <= 1e-6 * scale
    assert float(jnp.max(jnp.abs(jnp.where(mask.xy_corner_strict, 0.0, mg)))) == 0.0


def test_multigrid_pressure_matches_cg_with_neumann_walls():
    div, mask = _field(), _basin()
    solver = gj.build_multigrid_solver(mask.h, DX, DY, bc="neumann")
    mg = gfd.pressure_from_divergence(
        div, DX, DY, method="multigrid", mask=mask, multigrid_solver=solver
    )
    cg = gfd.pressure_from_divergence(div, DX, DY, method="cg", mask=mask)
    scale = float(jnp.max(jnp.abs(cg)))
    assert float(jnp.max(jnp.abs(mg - cg))) <= 1e-6 * scale


def test_multigrid_needs_a_matching_hierarchy():
    zeta, mask = _field(), _basin()
    with pytest.raises(ValueError, match="pre-built MultigridSolver"):
        gfd.streamfunction_from_vorticity(zeta, DX, DY, method="multigrid", mask=mask)
    neumann = gj.build_multigrid_solver(mask.h, DX, DY, bc="neumann")
    with pytest.raises(ValueError, match="bc='dirichlet'"):
        gfd.streamfunction_from_vorticity(
            zeta, DX, DY, method="multigrid", mask=mask, multigrid_solver=neumann
        )
    with pytest.raises(ValueError, match="cell faces"):
        gfd.pressure_from_divergence(
            zeta,
            DX,
            DY,
            bc="dst",
            method="multigrid",
            mask=mask,
            multigrid_solver=neumann,
        )
    with pytest.raises(ValueError, match="array-valued lambda_"):
        gfd.pv_inversion(
            _field(shape=(2, N, N)),
            DX,
            DY,
            jnp.array([0.0, 1.0]),
            method="multigrid",
            mask=mask,
            multigrid_solver=neumann,
        )
