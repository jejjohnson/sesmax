"""Public surface for sesmax-ocean's checks of a model before a run.

Does the inversion of a barotropic QG model undo its Laplacian, and does the density
of a layered model increase with depth? Each raises
:class:`sesmax_core.component.AssertionFailedError` when not. Those of the resolution
are in :mod:`sesmax_ocean.resolution`.
"""

from __future__ import annotations

from sesmax_ocean._src.checks import check_pv_inversion, check_static_stability


__all__ = ["check_pv_inversion", "check_static_stability"]
