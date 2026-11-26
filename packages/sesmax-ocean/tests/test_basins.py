"""The bundles of the basins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from sesmax_ocean.basins import BASINS, placeholder_bundle, write_bundle
from sesmax_ocean.scenarios import SCENARIOS


@pytest.mark.parametrize("name", list(BASINS))
def test_a_basin_has_its_geometry(name: str) -> None:
    basin = BASINS[name]
    assert basin.geometry_kind == SCENARIOS[name].geometry_kind
    assert basin.lon_bounds[0] < basin.lon_bounds[1]
    assert basin.lat_bounds[0] < basin.lat_bounds[1]
    on_a_plane = basin.geometry_kind == "real_basin"
    for length in (basin.Lx, basin.Ly, basin.ref_lat):
        assert (length is not None) is on_a_plane


@pytest.mark.parametrize("name", list(BASINS))
def test_a_bundle_is_written_and_read(name: str, tmp_path: Path) -> None:
    path = write_bundle(placeholder_bundle(name, nx=32, ny=16), tmp_path / "b.zarr")
    read = xr.open_zarr(path, consolidated=False)
    assert set(read.data_vars) == {"bathymetry", "mask", "tau_x", "tau_y", "heat"}
    assert set(read.coords) == {"lon", "lat"}
    assert read.mask.shape == (16, 32)
    assert read.attrs["basin_name"] == name
    assert read.attrs["geometry_kind"] == BASINS[name].geometry_kind
    assert "placeholder" in read.attrs["source"]
    wet = np.asarray(read.mask) == 1
    assert 0 < wet.sum() < wet.size
    depth = np.asarray(read.bathymetry)
    assert depth[wet].min() >= 20.0
    for field in (read.bathymetry, read.tau_x, read.tau_y):
        np.testing.assert_array_equal(np.asarray(field)[~wet], 0.0)


def test_the_shelf_deepens_by_a_step_per_cell() -> None:
    bundle = placeholder_bundle("southern_ocean", nx=4, ny=40, min_depth=100.0)
    column = np.asarray(bundle.bathymetry)[:, 0]
    steps = np.diff(column[np.flatnonzero(column)[0] :])
    np.testing.assert_allclose(steps[:7], 3900.0 / 8.0)
    np.testing.assert_allclose(steps[7:], 0.0)
    assert column.max() == 4000.0


def test_a_bundle_has_its_basin_s_grid_by_default() -> None:
    assert placeholder_bundle("med_sea").mask.shape == (64, 128)


def test_a_wrong_bundle_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown basin"):
        placeholder_bundle("no_such_basin")
    with pytest.raises(ValueError, match="must be > 0"):
        placeholder_bundle("med_sea", nx=0)


def test_a_bundle_replaces_a_bundle(tmp_path: Path) -> None:
    path = tmp_path / "b.zarr"
    write_bundle(placeholder_bundle("med_sea", nx=16, ny=8), path)
    write_bundle(placeholder_bundle("med_sea", nx=8, ny=8), path)
    assert xr.open_zarr(path, consolidated=False).mask.shape == (8, 8)


@pytest.mark.parametrize("name", ["bundles", "looks_like.zarr"])
def test_nothing_but_a_bundle_is_replaced(name: str, tmp_path: Path) -> None:
    """A path that is one directory short would lose every bundle in it."""
    other = tmp_path / name
    other.mkdir()
    (other / "kept.txt").write_text("kept")
    with pytest.raises(ValueError, match=r"Zarr store|'\.zarr' store"):
        write_bundle(placeholder_bundle("med_sea", nx=8, ny=8), other)
    assert (other / "kept.txt").read_text() == "kept"
