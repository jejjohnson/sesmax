"""What a scenario hands to a model, and the two registry entries.

A *scenario* is what we simulate: the geometry, the constants, the forcing and the
shape of the initial condition. A *model* is how: the equations, on its state. The
builder of a model reads a built scenario and the ``model`` block of a run spec.

Moved from somax 9cb19b3: ``somax/_src/cli/scenarios/_types.py`` and
``somax/_src/cli/models_registry/_types.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from jaxtyping import Array, Float

from sesmax_core import Model, State


_EMPTY: Mapping[str, Any] = MappingProxyType({})

GeometryKind = Literal["rectangular", "spherical_cap", "real_basin"]
Coordinates = Literal["cartesian", "spherical"]
InitialConditionKind = Literal["at_rest", "jet", "gaussian_eddy"]


@dataclass(frozen=True)
class Geometry:
    """The domain of a scenario.

    Args:
        kind: ``"rectangular"`` for a box, ``"real_basin"`` for a Cartesian basin
            with coasts, ``"spherical_cap"`` for a band of the sphere.
        nx: Interior cells in x.
        ny: Interior cells in y.
        Lx: Length in x (m), for a Cartesian geometry.
        Ly: Length in y (m), for a Cartesian geometry.
        lon_bounds: Longitudes (degrees), for a spherical geometry.
        lat_bounds: Latitudes (degrees), for a spherical geometry.
        mask: The wet cells, one on water, on the interior T-points.
        bathymetry: The depth (m), on the interior T-points.
    """

    kind: GeometryKind
    nx: int
    ny: int
    Lx: float | None = None
    Ly: float | None = None
    lon_bounds: tuple[float, float] | None = None
    lat_bounds: tuple[float, float] | None = None
    mask: Float[Array, "ny nx"] | None = None
    bathymetry: Float[Array, "ny nx"] | None = None


@dataclass(frozen=True)
class Constants:
    """The constants of a basin.

    Args:
        f0: Reference Coriolis parameter (1/s).
        beta: Meridional gradient of ``f`` (1/(m s)).
        rho0: Reference density (kg/m³).
        g: Gravitational acceleration (m/s²).
    """

    f0: float = 1.0e-4
    beta: float = 1.6e-11
    rho0: float = 1025.0
    g: float = 9.81


@dataclass(frozen=True)
class ForcingFields:
    """Forcing fields computed ahead of the run, on the interior T-points.

    Args:
        tau_x: Zonal wind stress.
        tau_y: Meridional wind stress.
        heat: Surface heat flux.
    """

    tau_x: Float[Array, "ny nx"] | None = None
    tau_y: Float[Array, "ny nx"] | None = None
    heat: Float[Array, "ny nx"] | None = None


@dataclass(frozen=True)
class InitialCondition:
    """The shape of an initial condition, which a model turns into a state.

    Args:
        kind: The shape; every model knows ``"at_rest"``.
        params: The numbers of the shape, such as ``jet_speed``.
    """

    kind: InitialConditionKind = "at_rest"
    params: Mapping[str, Any] = _EMPTY


@dataclass(frozen=True)
class Scenario:
    """Everything a model needs from a scenario.

    Args:
        name: The name of the scenario.
        geometry: The domain.
        constants: The constants of the basin.
        forcing: Forcing fields computed ahead of the run.
        initial_condition: The shape of the initial state.
        forcing_params: Numbers and names the model's own wind reads
            (``wind_amplitude``, ``wind_profile``); copied, and read-only.
    """

    name: str
    geometry: Geometry
    constants: Constants = field(default_factory=Constants)
    forcing: ForcingFields = field(default_factory=ForcingFields)
    initial_condition: InitialCondition = field(default_factory=InitialCondition)
    forcing_params: Mapping[str, Any] = _EMPTY

    def __post_init__(self) -> None:
        frozen = MappingProxyType(dict(self.forcing_params))
        object.__setattr__(self, "forcing_params", frozen)


@dataclass(frozen=True)
class ScenarioEntry:
    """A scenario that can be asked for.

    Args:
        name: Its name in a run spec.
        geometry_kind: The kind of its geometry, known before it is built.
        build: Builds it from the ``scenario`` block of a run spec, or ``None``
            while it waits for data.
        needs: What it waits for, if it cannot be built yet.
    """

    name: str
    geometry_kind: GeometryKind
    build: Callable[[Mapping[str, Any]], Scenario] | None = field(
        default=None, repr=False
    )
    needs: str = ""


@dataclass(frozen=True)
class ModelEntry:
    """A model that can be asked for.

    Args:
        name: Its name in a run spec.
        coordinates: The coordinates of its grid.
        masks: Whether it runs on a domain with land.
        build: Builds the model and its initial state from a scenario and the
            ``model`` block of a run spec.
        from_nondimensional: The same from a scenario and its ``nondim`` block, or
            ``None`` if the model is not built from dimensionless numbers.
    """

    name: str
    coordinates: Coordinates
    masks: bool
    build: Callable[[Scenario, Mapping[str, Any]], tuple[Model, State]] = field(
        repr=False
    )
    from_nondimensional: (
        Callable[[Scenario, Mapping[str, Any]], tuple[Model, State]] | None
    ) = field(default=None, repr=False)
