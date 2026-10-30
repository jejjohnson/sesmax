"""Public surface for sesmax-core's non-dimensionalisation and standardisation.

:class:`Scales` holds a model's reference length, velocity, depth and time
and derives its dimensionless numbers; :class:`StateAffine` maps each state
leaf to ``(x - loc) / scale``, from a :class:`Scales` or from sample
moments; :class:`ScaledModel` integrates any model in those coordinates.

All three are also importable from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.scaled import ScaledModel
from sesmax_core._src.scales import Scales
from sesmax_core._src.state_affine import StateAffine


__all__ = ["ScaledModel", "Scales", "StateAffine"]
