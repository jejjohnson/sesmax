"""Public surface for the data of real basins; needs the ``[data]`` extra.

A bundle is the mask, the depth and the forcing of a basin on one grid, as a Zarr
store. ``BASINS`` holds the basins, and :func:`placeholder_bundle` makes a stand-in
with the layout of a real bundle. The command ``sesmax-ocean data`` builds and
fetches them with DVC.
"""

from __future__ import annotations

from sesmax_ocean._src.basins.bundle import (
    BASINS as BASINS,
    Basin,
    placeholder_bundle,
    write_bundle,
)


__all__ = ["Basin", "placeholder_bundle", "write_bundle"]
