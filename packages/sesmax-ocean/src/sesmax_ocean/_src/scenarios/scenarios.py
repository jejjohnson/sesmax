"""The scenarios: what we simulate.

Moved from somax 9cb19b3: ``somax/_src/cli/scenarios/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, get_args

from sesmax_ocean._src.scenarios.blocks import known, read
from sesmax_ocean._src.scenarios.types import (
    Constants,
    Geometry,
    GeometryKind,
    InitialCondition,
    InitialConditionKind,
    Scenario,
    ScenarioEntry,
)


def double_gyre(params: Mapping[str, Any]) -> Scenario:
    """A rectangular basin, driven by the curl of an idealised wind.

    Args:
        params: The ``scenario`` block of a run spec: ``grid`` with ``nx`` and
            ``ny``, and ``Lx`` and ``Ly`` (10⁶ m each unless given); ``consts``
            with any of ``f0``, ``beta``, ``rho0`` and ``g``; ``forcing`` with any
            of ``wind_amplitude`` and ``wind_profile``; and ``initial_condition``
            with a ``type`` (``"at_rest"``, ``"jet"`` or ``"gaussian_eddy"``) and
            its ``params``.

    Returns:
        The scenario.

    Raises:
        ValueError: If a key or the initial condition is unknown, or ``nx`` or
            ``ny`` is missing.
    """
    where = "double_gyre: scenario"
    grid = read(
        f"{where}.grid",
        params.get("grid", {}),
        {"nx": int, "ny": int, "Lx": float, "Ly": float},
    )
    if missing := sorted({"nx", "ny"} - set(grid)):
        raise ValueError(f"{where}.grid missing required key(s) {missing}")
    consts = read(
        f"{where}.consts",
        params.get("consts", {}),
        dict.fromkeys(("f0", "beta", "rho0", "g"), float),
    )
    forcing = read(
        f"{where}.forcing",
        params.get("forcing", {}),
        {"wind_amplitude": float, "wind_profile": str},
    )
    initial = params.get("initial_condition", {})
    known(f"{where}.initial_condition", initial, ("type", "params"))
    kind = initial.get("type", "at_rest")
    if kind not in get_args(InitialConditionKind):
        raise ValueError(
            f"{where}: unknown initial_condition.type {kind!r}; supported: "
            f"{sorted(get_args(InitialConditionKind))}"
        )
    return Scenario(
        name="double_gyre",
        geometry=Geometry(kind="rectangular", **{"Lx": 1.0e6, "Ly": 1.0e6} | grid),
        constants=Constants(**consts),
        initial_condition=InitialCondition(kind, dict(initial.get("params", {}))),
        forcing_params=forcing,
    )


_DATA = "its coastline, depths and wind, regridded onto its grid (sesmax#115)"


def _planned(name: str, kind: GeometryKind) -> ScenarioEntry:
    return ScenarioEntry(name=name, geometry_kind=kind, needs=_DATA)


#: The scenarios, by name.
SCENARIOS: dict[str, ScenarioEntry] = {
    entry.name: entry
    for entry in (
        ScenarioEntry("double_gyre", "rectangular", double_gyre),
        _planned("global_ocean", "spherical_cap"),
        _planned("gulf_stream", "real_basin"),
        _planned("med_sea", "real_basin"),
        _planned("north_atlantic", "real_basin"),
        _planned("southern_ocean", "spherical_cap"),
    )
}


def get_scenario(name: str) -> ScenarioEntry:
    """The scenario of a name.

    Raises:
        KeyError: If there is none.
    """
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


def build_scenario(params: Mapping[str, Any]) -> Scenario:
    """Build the scenario a ``scenario`` block names.

    Raises:
        KeyError: If there is no such scenario.
        NotImplementedError: If it waits for data.
        ValueError: If the block has a key or a value it does not take.
    """
    entry = get_scenario(params["name"])
    if entry.build is None:
        raise NotImplementedError(f"the scenario {entry.name!r} needs {entry.needs}")
    return entry.build(params)
