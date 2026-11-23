"""The wind-stress ``rho0`` default is geophysix's reference density, 1025.0."""

from __future__ import annotations

from geophysix.seawater import LINEAR_EOS
from gridax import CartesianCGrid2D, CartesianCGrid3D

from sesmax_core.forcing import WindCurl2D, WindStress2D, WindStress3D


def test_default_rho0_is_the_linear_eos_reference_density() -> None:
    grid2d = CartesianCGrid2D.from_interior(8, 8, 1.0, 1.0)
    grid3d = CartesianCGrid3D.from_interior(6, 6, 4, 1.0, 1.0, 1.0)
    defaults = [
        WindStress2D(grid2d).rho0,
        WindCurl2D(grid2d).rho0,
        WindStress3D(grid3d)._wind2d.rho0,
    ]
    # The value did not move when the literal became a geophysix record.
    assert defaults == [LINEAR_EOS.reference_density] * 3
    assert LINEAR_EOS.reference_density == 1025.0
