"""Public surface for sesmax-core's spatial and time domains.

:class:`Domain` is a regular N-dimensional grid (bounds, spacing and size
per axis) and :class:`TimeDomain` a time axis; the forcing bank evaluates
its bases on a :class:`Domain`.
"""

from __future__ import annotations

from sesmax_core._src.domain.domain import Domain
from sesmax_core._src.domain.time import TimeDomain


__all__ = ["Domain", "TimeDomain"]
