"""sesmax-ocean: simple ocean models in JAX, on sesmax-core and gridax.

The models moved from somax at 9cb19b3 live here, with what belongs to them:
the resolution checks of their boundary layers and deformation radii
(:mod:`sesmax_ocean.resolution`), reference-free metrics of a state
(:mod:`sesmax_ocean.eval`) and forcing presets named after ocean fields
(:mod:`sesmax_ocean.forcing`).

Everything is imported from its submodule; the top level holds the version.
"""

from __future__ import annotations


__version__ = "0.0.0"  # x-release-please-version

__all__ = ["__version__"]
