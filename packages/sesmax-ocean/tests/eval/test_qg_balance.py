"""Tests for the QG balance residual and invariant wiring in eval metrics.

Model construction + a few diagnose/inversion calls — no integration, fast
PR lane.

Moved from somax 9cb19b3: ``tests/eval/test_qg_balance.py``.
"""

from __future__ import annotations

import jax.numpy as jnp

from sesmax_ocean.eval import compute_eval_metrics, qg_balance_residual
from sesmax_ocean.qg import BarotropicQG, BarotropicQGState


def _barotropic():
    return BarotropicQG.create(nx=32, ny=32, lateral_viscosity=100.0)


class TestQGBalanceResidual:
    def test_zero_pv_is_zero_residual(self) -> None:
        model = _barotropic()
        state = BarotropicQGState(q=jnp.zeros((model.grid.Ny, model.grid.Nx)))
        assert float(qg_balance_residual(model, state)) == 0.0

    def test_inverted_state_closes_round_trip(self) -> None:
        """psi from inversion, then q = laplacian(psi) is self-consistent."""
        model = _barotropic()
        # A nonzero interior PV bump.
        q = jnp.zeros((model.grid.Ny, model.grid.Nx)).at[16, 16].set(1e-5)
        state = BarotropicQGState(q=q)
        # The residual measures ||laplacian(invert(q)) - q|| / ||q||; for a
        # consistent elliptic operator this is small.
        assert float(qg_balance_residual(model, state)) < 1e-3


class TestComputeEvalMetricsQG:
    def test_qg_model_reports_balance_and_invariants(self) -> None:
        model = _barotropic()
        q = jnp.zeros((model.grid.Ny, model.grid.Nx)).at[16, 16].set(1e-5)
        metrics = compute_eval_metrics(model, BarotropicQGState(q=q))
        assert "qg_balance_residual" in metrics
        assert "invariant_kinetic_energy" in metrics
        assert "invariant_enstrophy" in metrics
        # QG models do NOT report the velocity-divergence metrics.
        assert "rms_divergence" not in metrics
