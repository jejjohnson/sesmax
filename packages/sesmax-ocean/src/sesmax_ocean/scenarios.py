"""Public surface for what the ocean simulates, how, and which go together.

A scenario is a geometry, its constants, its forcing and the shape of its initial
condition; a model is the equations on a state. :func:`build_ocean_run` builds the
two from the blocks of a run spec, and :func:`check_compatible` says whether they go
together. ``SCENARIOS`` and ``MODELS`` hold the entries by name.
"""

from __future__ import annotations

from sesmax_ocean._src.scenarios.pairs import (
    MODELS as MODELS,
    IncompatiblePairError,
    build_ocean_run,
    check_compatible,
    compatible_models,
    get_model,
)
from sesmax_ocean._src.scenarios.scenarios import (
    SCENARIOS as SCENARIOS,
    build_scenario,
    get_scenario,
)
from sesmax_ocean._src.scenarios.types import (
    Constants,
    ForcingFields,
    Geometry,
    InitialCondition,
    ModelEntry,
    Scenario,
    ScenarioEntry,
)


__all__ = [
    "Constants",
    "ForcingFields",
    "Geometry",
    "IncompatiblePairError",
    "InitialCondition",
    "ModelEntry",
    "Scenario",
    "ScenarioEntry",
    "build_ocean_run",
    "build_scenario",
    "check_compatible",
    "compatible_models",
    "get_model",
    "get_scenario",
]
