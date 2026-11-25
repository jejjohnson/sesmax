"""Which scenario goes with which model, and the run the two make.

Moved from somax 9cb19b3: ``somax/_src/cli/_compatibility.py`` and
``_factories.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sesmax_core.component import BuiltRun
from sesmax_ocean._src.scenarios.cartesian import CARTESIAN
from sesmax_ocean._src.scenarios.scenarios import (
    build_scenario,
    get_scenario,
)
from sesmax_ocean._src.scenarios.sphere import SPHERICAL
from sesmax_ocean._src.scenarios.types import ModelEntry, Scenario, ScenarioEntry


#: The models, by name.
MODELS: dict[str, ModelEntry] = {
    entry.name: entry
    for entry in sorted((*CARTESIAN, *SPHERICAL), key=lambda m: m.name)
}

#: Pairs the rules let through and the models refuse. A model on the sphere closes
#: its band with walls, so it cannot reach a pole, as the whole ocean does.
INCOMPATIBLE: frozenset[tuple[str, str]] = frozenset(
    {("global_ocean", "spherical_swm"), ("global_ocean", "spherical_qg")}
)


class IncompatiblePairError(ValueError):
    """A scenario and a model that do not go together."""


def get_model(name: str) -> ModelEntry:
    """The model of a name.

    Raises:
        KeyError: If there is none.
    """
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name]


def why_not(scenario: ScenarioEntry, model: ModelEntry) -> str | None:
    """Why a scenario and a model do not go together, or ``None`` if they do.

    A box and a basin with coasts need Cartesian coordinates, a band of the sphere
    spherical ones, and whatever has coasts a model that takes a mask.
    """
    if (scenario.name, model.name) in INCOMPATIBLE:
        return "the domain reaches a pole, where the model has a wall"
    needed = "spherical" if scenario.geometry_kind == "spherical_cap" else "cartesian"
    if model.coordinates != needed:
        return (
            f"the geometry {scenario.geometry_kind!r} needs {needed} coordinates, and "
            f"the model has {model.coordinates} ones"
        )
    if scenario.geometry_kind != "rectangular" and not model.masks:
        return "the scenario has land, and the model takes no mask"
    return None


def check_compatible(scenario: str, model: str) -> None:
    """A scenario and a model go together.

    Args:
        scenario: The name of the scenario.
        model: The name of the model.

    Raises:
        KeyError: If there is no such scenario or model.
        IncompatiblePairError: If they do not go together.
    """
    reason = why_not(get_scenario(scenario), get_model(model))
    if reason is not None:
        raise IncompatiblePairError(
            f"the scenario {scenario!r} does not go with the model {model!r}: {reason}"
        )


def compatible_models(scenario: str) -> tuple[str, ...]:
    """The models a scenario goes with, by name.

    Raises:
        KeyError: If there is no such scenario.

    Examples:
        >>> from sesmax_ocean.scenarios import compatible_models
        >>> compatible_models("southern_ocean")
        ('spherical_qg', 'spherical_swm')
    """
    entry = get_scenario(scenario)
    return tuple(
        name for name, model in MODELS.items() if why_not(entry, model) is None
    )


def build_ocean_run(spec: Mapping[str, Any]) -> BuiltRun:
    """Build the model and the initial state of a run.

    Args:
        spec: The ``scenario`` and ``model`` blocks of a run spec. A ``scenario``
            block with a ``nondim`` block makes a dimensionless run.

    Returns:
        The model and its initial state, with its boundary conditions.

    Raises:
        KeyError: If there is no such scenario or model.
        IncompatiblePairError: If they do not go together.
        NotImplementedError: If the scenario waits for data, or the model has no such
            initial condition.
        ValueError: If a block has a key or a value that is not known, or the model
            is not built from dimensionless numbers.

    Examples:
        >>> from sesmax_ocean.scenarios import build_ocean_run
        >>> run = build_ocean_run(
        ...     {
        ...         "scenario": {
        ...             "name": "double_gyre",
        ...             "grid": {"nx": 16, "ny": 16, "Lx": 1.0e6, "Ly": 1.0e6},
        ...             "forcing": {"wind_amplitude": 1.0e-12},
        ...         },
        ...         "model": {
        ...             "name": "barotropic_qg",
        ...             "params": {"lateral_viscosity": 500.0},
        ...         },
        ...     }
        ... )
        >>> type(run.model).__name__, run.state0.q.shape
        ('BarotropicQG', (18, 18))
    """
    model, block = spec["model"], spec["scenario"]
    check_compatible(block["name"], model["name"])
    entry, scenario = get_model(model["name"]), build_scenario(block)
    if nondim := block.get("nondim"):
        if entry.from_nondimensional is None:
            raise ValueError(
                f"the model {entry.name!r} is not built from dimensionless numbers; "
                "replace scenario.nondim by scenario.consts"
            )
        _refuse_dimensional(model, scenario)
        built, state = entry.from_nondimensional(scenario, nondim)
    else:
        built, state = entry.build(scenario, model)
    return BuiltRun(
        model=built,
        state0=built.apply_boundary_conditions(state),
        coordinates="nondimensional" if nondim else "si",
    )


def _refuse_dimensional(model: Mapping[str, Any], scenario: Scenario) -> None:
    """A spec with dimensionless numbers gives nothing else a model is built from.

    Raises:
        ValueError: If it does.
    """
    given = [
        f"model.{block}.{key}"
        for block in ("params", "stratification")
        for key in model.get(block, {})
    ]
    if "wind_amplitude" in scenario.forcing_params:
        given.append("scenario.forcing.wind_amplitude")
    if given:
        raise ValueError(
            f"{model['name']}: {given} cannot be combined with scenario.nondim, whose "
            "numbers determine the model"
        )
