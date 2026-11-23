"""The core of sesmax: what every simple Earth-system model shares.

``sesmax_core`` holds the model-agnostic machinery moved from somax (at
commit ``9cb19b3``): the state and parameter types and their constraints,
the model base classes, the term algebra, non-dimensionalisation and
standardisation (``Scales``, ``StateAffine``, ``ScaledModel``), the
layered-fluid pieces (``ModalTransform``), and forcing
with its reduced-order bases. The data-assimilation adapters, the flowjax
bridge, monitors and rollouts are in their own submodules, some behind an
extra (``[da]``, ``[flows]``); the I/O functions and the checkpointer are
also top-level and import xarray or orbax only when called (``[xarray]``,
``[checkpoint]``).

A model is an :class:`equinox.Module` whose ``vector_field`` is a pure
function of time, state and parameters, so ``jax.jit``, ``jax.grad`` and
``jax.vmap`` go through a whole integration window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from sesmax_core._src.checkpoint import SimulationCheckpointer
from sesmax_core._src.driver import solve_pde
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
from sesmax_core._src.gfd.inversion import (
    pressure_from_divergence,
    pv_inversion,
    streamfunction_from_vorticity,
)
from sesmax_core._src.gfd.modal import ModalTransform, StratificationProfile
from sesmax_core._src.io import (
    append_to_dataset,
    apply_scale_metadata,
    dataset_to_state,
    load_dataset,
    save_dataset,
    scales_attrs,
    snapshots_to_dataset,
    state_to_dataset,
    transform_attrs,
)
from sesmax_core._src.model import Model, TermModel
from sesmax_core._src.scaled import ScaledModel
from sesmax_core._src.scales import Scales
from sesmax_core._src.state_affine import StateAffine
from sesmax_core._src.terms import (
    Compose,
    Scaled,
    Sum,
    Term,
    TermFn,
    build_diffrax_terms,
    explicit,
    implicit,
    partition,
)
from sesmax_core._src.types import (
    Diagnostics,
    Params,
    PhysConsts,
    State,
    as_parameter,
    frozen,
    interval,
    positive,
    trainable_mask,
)
from sesmax_core.solvers import stable_step


if TYPE_CHECKING:
    from sesmax_core._src.operator_term import OperatorTerm


__version__ = "0.2.0"  # x-release-please-version

__all__ = [
    "AbstractForcing",
    "BasisForcing",
    "Compose",
    "ConstantForcing",
    "ConstantInTime",
    "Diagnostics",
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
    "ModalTransform",
    "Model",
    "NoForcing",
    "OperatorTerm",
    "Params",
    "PhysConsts",
    "QuadraticDrag2D",
    "QuadraticDrag3D",
    "RayleighDamping2D",
    "RayleighDamping3D",
    "Scaled",
    "ScaledModel",
    "Scales",
    "SeasonalWindForcing",
    "SimulationCheckpointer",
    "SpatialBasis",
    "State",
    "StateAffine",
    "StratificationProfile",
    "Sum",
    "TemporalBasis",
    "Term",
    "TermFn",
    "TermModel",
    "TimeVaryingForcing",
    "TransformedForcing",
    "VectorBasisForcing",
    "VectorSpatialBasis",
    "WindCurl2D",
    "WindStress2D",
    "WindStress3D",
    "__version__",
    "add_to",
    "add_vector_to",
    "append_to_dataset",
    "apply_scale_metadata",
    "as_parameter",
    "build_diffrax_terms",
    "control_filter",
    "dataset_to_state",
    "explicit",
    "frozen",
    "implicit",
    "interval",
    "linear_drag",
    "linear_drag_tendency",
    "load_dataset",
    "matern_spectral_density",
    "partition",
    "positive",
    "pressure_from_divergence",
    "pv_inversion",
    "quadratic_drag_tendency",
    "rayleigh_relaxation",
    "rayleigh_tendency",
    "save_dataset",
    "scales_attrs",
    "snapshots_to_dataset",
    "solve_pde",
    "spatial_from_divfree",
    "spatial_from_eof",
    "spatial_from_fourier",
    "spatial_from_gabor",
    "spatial_from_graph_laplacian",
    "spatial_from_rbf",
    "spatial_from_spherical_rbf",
    "spatial_from_wavelet",
    "stable_step",
    "state_to_dataset",
    "streamfunction_from_vorticity",
    "tile_in_time",
    "trainable_mask",
    "transform_attrs",
    "wind_stress_tendency",
]


def __getattr__(name: str) -> object:
    """Import ``OperatorTerm`` on first use.

    It needs pipekit, so a plain ``import sesmax_core`` stays pipekit-free.
    """
    if name == "OperatorTerm":
        from sesmax_core._src.operator_term import OperatorTerm

        return OperatorTerm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
