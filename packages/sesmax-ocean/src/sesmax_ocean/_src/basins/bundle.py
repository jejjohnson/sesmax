"""The bundle of a basin: its mask, its depth and its forcing, on one grid.

Moved from somax 9cb19b3: ``scripts/data/build_basin.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from jaxtyping import Bool, Float

from sesmax_ocean import __version__


try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - the extra is installed for tests
    raise ImportError(
        "sesmax_ocean.basins needs xarray and zarr: pip install 'sesmax-ocean[data]'"
    ) from exc


#: The version of the layout of a bundle.
LAYOUT_VERSION = "0.1.0"


@dataclass(frozen=True)
class Basin:
    """A basin, the grid of its bundle and the strength of its wind.

    Args:
        nx: Cells in longitude.
        ny: Cells in latitude.
        lon_bounds: Longitudes (degrees).
        lat_bounds: Latitudes (degrees).
        geometry_kind: ``"real_basin"`` on a plane, ``"spherical_cap"`` on a band.
        Lx: Length in x (m), on a plane.
        Ly: Length in y (m), on a plane.
        ref_lat: Latitude of the projection (degrees), on a plane.
        tau0: Amplitude of the wind stress of the placeholder (Pa).
    """

    nx: int
    ny: int
    lon_bounds: tuple[float, float]
    lat_bounds: tuple[float, float]
    geometry_kind: Literal["real_basin", "spherical_cap"] = "real_basin"
    Lx: float | None = None
    Ly: float | None = None
    ref_lat: float | None = None
    tau0: float = 0.1


#: The basins, by name.
BASINS: dict[str, Basin] = {
    "gulf_stream": Basin(
        256, 128, (-82.0, -40.0), (25.0, 45.0), Lx=4.0e6, Ly=2.0e6, ref_lat=35.0
    ),
    "med_sea": Basin(
        128,
        64,
        (-6.0, 36.5),
        (30.0, 46.0),
        Lx=3.5e6,
        Ly=1.5e6,
        ref_lat=38.0,
        tau0=0.05,
    ),
    "north_atlantic": Basin(
        256, 256, (-80.0, 0.0), (10.0, 70.0), Lx=7.0e6, Ly=7.0e6, ref_lat=40.0
    ),
    "southern_ocean": Basin(
        360, 90, (0.0, 360.0), (-70.0, -30.0), "spherical_cap", tau0=0.15
    ),
}

_ATTRS = {
    "bathymetry": {"long_name": "bathymetry", "units": "m", "positive": "down"},
    "mask": {
        "long_name": "wet/dry mask",
        "flag_values": [0, 1],
        "flag_meanings": "land ocean",
    },
    "tau_x": {"long_name": "eastward surface wind stress", "units": "Pa"},
    "tau_y": {"long_name": "northward surface wind stress", "units": "Pa"},
    "heat": {
        "long_name": "net surface heat flux",
        "units": "W m-2",
        "positive": "down",
    },
    "lon": {"long_name": "longitude", "units": "degrees_east"},
    "lat": {"long_name": "latitude", "units": "degrees_north"},
}


def placeholder_bundle(
    name: str,
    *,
    nx: int | None = None,
    ny: int | None = None,
    min_depth: float = 20.0,
) -> xr.Dataset:
    """A made-up bundle of a basin, with the layout of a real one.

    Its coasts, its depths and its wind are simple shapes that only resemble the
    basin. The real data are sesmax#115.

    Args:
        name: One of ``BASINS``.
        nx: Cells in longitude; the basin's by default.
        ny: Cells in latitude; the basin's by default.
        min_depth: The smallest depth of a wet cell (m).

    Returns:
        ``bathymetry``, ``mask``, ``tau_x``, ``tau_y`` and ``heat`` on ``(y, x)``,
        with the coordinates ``lon`` and ``lat``.

    Raises:
        ValueError: If there is no such basin, or a number of cells is not positive.

    Examples:
        >>> from sesmax_ocean.basins import placeholder_bundle
        >>> bundle = placeholder_bundle("med_sea", nx=16, ny=8)
        >>> bundle["mask"].shape, sorted(bundle.data_vars)
        ((8, 16), ['bathymetry', 'heat', 'mask', 'tau_x', 'tau_y'])
    """
    if name not in BASINS:
        raise ValueError(f"unknown basin {name!r}; known: {sorted(BASINS)}")
    basin = BASINS[name]
    nx = basin.nx if nx is None else nx
    ny = basin.ny if ny is None else ny
    if nx <= 0 or ny <= 0:
        raise ValueError(f"nx and ny must be > 0, got {nx} and {ny}")
    lon, lat = np.meshgrid(
        _centres(*basin.lon_bounds, nx), _centres(*basin.lat_bounds, ny)
    )
    wet = _wet(name, lon, lat)
    south, north = basin.lat_bounds
    stress = -basin.tau0 * np.cos(np.pi * (lat - south) / (north - south))
    fields = {
        "bathymetry": _depth(wet, min_depth=min_depth),
        "mask": wet.astype(np.int8),
        "tau_x": np.where(wet, stress, 0.0),
        "tau_y": np.zeros_like(lon),
        "heat": np.zeros_like(lon),
    }
    attrs = {
        "title": f"basin bundle: {name}",
        "source": "placeholder: made-up coasts, depths and wind (sesmax#115)",
        "history": f"sesmax-ocean {__version__}: {nx=}, {ny=}, {min_depth=}",
        "conventions": "CF-1.10",
        "basin_layout_version": LAYOUT_VERSION,
        "basin_name": name,
        "geometry_kind": basin.geometry_kind,
        "lon_bounds": list(basin.lon_bounds),
        "lat_bounds": list(basin.lat_bounds),
        "Lx_m": basin.Lx,
        "Ly_m": basin.Ly,
        "ref_lat_deg": basin.ref_lat,
    }
    bundle = xr.Dataset(
        {key: (("y", "x"), _single(value)) for key, value in fields.items()},
        coords={"lon": (("y", "x"), _single(lon)), "lat": (("y", "x"), _single(lat))},
        attrs={key: value for key, value in attrs.items() if value is not None},
    )
    for key, described in _ATTRS.items():
        bundle[key].attrs.update(described)
    return bundle


def write_bundle(bundle: xr.Dataset, path: Path) -> Path:
    """Write a bundle as a Zarr store, in place of one that is there.

    Args:
        bundle: The bundle.
        path: Where to; it ends in ``.zarr``.

    Returns:
        The absolute path.

    Raises:
        ValueError: If the path does not end in ``.zarr``, or holds something else
            than a Zarr store, which would be lost.
    """
    path = Path(path)
    if path.suffix != ".zarr":
        raise ValueError(f"a bundle is a '.zarr' store, not {path}")
    if path.exists() and not (path / "zarr.json").is_file():
        raise ValueError(f"not a Zarr store, so it is not replaced: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Consolidated metadata is no part of Zarr 3.
    bundle.to_zarr(path, mode="w", zarr_format=3, consolidated=False)
    return path.resolve()


def _centres(low: float, high: float, n: int) -> Float[np.ndarray, " n"]:
    return low + (high - low) * (np.arange(n) + 0.5) / n


def _single(field: np.ndarray) -> np.ndarray:
    """Single precision, as bundles are; the mask keeps its type."""
    return field if field.dtype == np.int8 else field.astype(np.float32)


def _wet(
    name: str, lon: Float[np.ndarray, "ny nx"], lat: Float[np.ndarray, "ny nx"]
) -> Bool[np.ndarray, "ny nx"]:
    """Where a basin is wet, roughly."""
    match name:
        case "med_sea":
            # Coasts of Africa and Europe, and Italy.
            sea = (lat >= 31.0) & (lat <= 45.0)
            italy = (lon >= 7.0) & (lon <= 18.0) & (lat >= 36.0) & (lat <= 44.5)
            around = ((lon - 13.0) / 6.0) ** 2 + ((lat - 40.0) / 3.5) ** 2 < 1.0
            return sea & ~(italy & ~around)
        case "gulf_stream":
            # A coast that runs to the north-east.
            return lon > -82.0 + 0.8 * (lat - 25.0)
        case "north_atlantic":
            # The Americas, Europe with Africa, and Greenland.
            west, east = -80.0 + 0.6 * (lat - 10.0), -20.0 + 0.4 * (lat - 10.0)
            greenland = (lon > -55.0) & (lon < -20.0) & (lat > 60.0)
            return (lon > west) & (lon < east) & ~greenland
        case _:
            # Antarctica, and its peninsula.
            peninsula = (lon > 280.0) & (lon < 305.0) & (lat < -62.0)
            return (lat > -66.0) & ~peninsula


def _depth(
    wet: Bool[np.ndarray, "ny nx"],
    *,
    min_depth: float,
    abyss: float = 4000.0,
    shelf: int = 8,
) -> Float[np.ndarray, "ny nx"]:
    """A flat bottom, with a shelf of some cells along the coast; zero on land."""
    # How many steps to the nearest land, one wave of neighbours after another. The
    # edges repeat, so they bring in no land of their own.
    steps = np.full(wet.shape, shelf)
    reached = ~wet
    for step in range(shelf):
        steps[reached & (steps == shelf)] = step
        near = np.pad(reached, 1, mode="edge")
        reached = reached | near[:-2, 1:-1] | near[2:, 1:-1]
        reached = reached | near[1:-1, :-2] | near[1:-1, 2:]
    depth = min_depth + (abyss - min_depth) * steps / shelf
    return np.where(wet, depth, 0.0)
