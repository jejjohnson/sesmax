"""Public surface for sesmax-core's forcing and its reduced-order bases.

A forcing is a field of time (:class:`ForcingProtocol`): constant, seasonal,
interpolated from data, or a reduced-order :class:`BasisForcing`
``u(t, s) = sum_k c_k w_k(t) phi_k(s)`` whose coefficients carry a prior
standard deviation per mode, so a data-assimilation run can estimate them.
:class:`ForcingTerm` places a forcing into a model's tendency as a term. The
``spatial_from_*`` constructors build the spatial modes from geonnax bases
on a :class:`~sesmax_core.domain.Domain`.

All names are also importable from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.basis import (
    BasisForcing,
    ConstantInTime,
    ForcingTerm,
    FourierInTime,
    SpatialBasis,
    TemporalBasis,
    TransformedForcing,
    VectorBasisForcing,
    VectorSpatialBasis,
    add_to,
    add_vector_to,
    control_filter,
)
from sesmax_core._src.forcing import (
    ConstantForcing,
    ForcingProtocol,
    InterpolatedForcing,
    NoForcing,
    SeasonalWindForcing,
)
from sesmax_core._src.forcing_bank import (
    GaussianWindowsInTime,
    matern_spectral_density,
    spatial_from_divfree,
    spatial_from_eof,
    spatial_from_fourier,
    spatial_from_gabor,
    spatial_from_graph_laplacian,
    spatial_from_rbf,
    spatial_from_spherical_rbf,
    spatial_from_wavelet,
    tile_in_time,
)


__all__ = [
    "BasisForcing",
    "ConstantForcing",
    "ConstantInTime",
    "ForcingProtocol",
    "ForcingTerm",
    "FourierInTime",
    "GaussianWindowsInTime",
    "InterpolatedForcing",
    "NoForcing",
    "SeasonalWindForcing",
    "SpatialBasis",
    "TemporalBasis",
    "TransformedForcing",
    "VectorBasisForcing",
    "VectorSpatialBasis",
    "add_to",
    "add_vector_to",
    "control_filter",
    "matern_spectral_density",
    "spatial_from_divfree",
    "spatial_from_eof",
    "spatial_from_fourier",
    "spatial_from_gabor",
    "spatial_from_graph_laplacian",
    "spatial_from_rbf",
    "spatial_from_spherical_rbf",
    "spatial_from_wavelet",
    "tile_in_time",
]
