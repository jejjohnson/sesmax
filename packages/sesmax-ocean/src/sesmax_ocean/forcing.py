"""Public surface for sesmax-ocean's forcing presets.

Each preset chooses a spatial basis of :mod:`sesmax_core.forcing` and its prior
for one ocean field, and returns a forcing with zero coefficients.
"""

from __future__ import annotations

from sesmax_ocean._src.forcing import (
    geostrophic_currents,
    ssh_geostrophic,
    sss_coastal,
    sst_frontal,
)


__all__ = [
    "geostrophic_currents",
    "ssh_geostrophic",
    "sss_coastal",
    "sst_frontal",
]
