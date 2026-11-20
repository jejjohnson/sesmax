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
    HatInTime,
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
from sesmax_core._src.forcing_ops._base import (
    AbstractForcing,
    ForcingProduct,
    ForcingSum,
    TimeVaryingForcing,
)
from sesmax_core._src.forcing_ops.drag import (
    LinearDrag2D,
    LinearDrag3D,
    QuadraticDrag2D,
    QuadraticDrag3D,
)
from sesmax_core._src.forcing_ops.functional import (
    linear_drag_tendency,
    quadratic_drag_tendency,
    rayleigh_tendency,
    wind_stress_tendency,
)
from sesmax_core._src.forcing_ops.rayleigh import (
    RayleighDamping2D,
    RayleighDamping3D,
)
from sesmax_core._src.forcing_ops.relaxation import (
    linear_drag,
    rayleigh_relaxation,
)
from sesmax_core._src.forcing_ops.wind import (
    WindCurl2D,
    WindStress2D,
    WindStress3D,
)


__all__ = [
    "AbstractForcing",
    "BasisForcing",
    "ConstantForcing",
    "ConstantInTime",
    "ForcingProduct",
    "ForcingProtocol",
    "ForcingSum",
    "ForcingTerm",
    "FourierInTime",
    "GaussianWindowsInTime",
    "HatInTime",
    "InterpolatedForcing",
    "LinearDrag2D",
    "LinearDrag3D",
    "NoForcing",
    "QuadraticDrag2D",
    "QuadraticDrag3D",
    "RayleighDamping2D",
    "RayleighDamping3D",
    "SeasonalWindForcing",
    "SpatialBasis",
    "TemporalBasis",
    "TimeVaryingForcing",
    "TransformedForcing",
    "VectorBasisForcing",
    "VectorSpatialBasis",
    "WindCurl2D",
    "WindStress2D",
    "WindStress3D",
    "add_to",
    "add_vector_to",
    "control_filter",
    "linear_drag",
    "linear_drag_tendency",
    "matern_spectral_density",
    "quadratic_drag_tendency",
    "rayleigh_relaxation",
    "rayleigh_tendency",
    "spatial_from_divfree",
    "spatial_from_eof",
    "spatial_from_fourier",
    "spatial_from_gabor",
    "spatial_from_graph_laplacian",
    "spatial_from_rbf",
    "spatial_from_spherical_rbf",
    "spatial_from_wavelet",
    "tile_in_time",
    "wind_stress_tendency",
]
