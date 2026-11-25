"""Public surface for sesmax-ocean's reference-free metrics.

Numbers about one state that need no truth to compare with: RMS divergence,
enstrophy, kinetic energy, the departure from geostrophic balance, and how
well a barotropic QG state closes its own inversion.
:func:`compute_eval_metrics` returns the ones that apply to a model.
"""

from __future__ import annotations

from sesmax_ocean._src.eval import (
    compute_eval_metrics,
    geostrophic_imbalance,
    kinetic_energy,
    qg_balance_residual,
    rms_divergence,
    total_enstrophy,
)


__all__ = [
    "compute_eval_metrics",
    "geostrophic_imbalance",
    "kinetic_energy",
    "qg_balance_residual",
    "rms_divergence",
    "total_enstrophy",
]
