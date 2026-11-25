"""How a component of the Earth system offers its models to a run."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal

from sesmax_core._src.model import Model
from sesmax_core._src.types import State


@dataclasses.dataclass(frozen=True)
class BuiltRun:
    """A model and the state it starts from.

    Args:
        model: The model.
        state0: Its initial state.
        coordinates: Whether time and state are in SI units or in the units of
            the model's scales.
    """

    model: Model
    state0: State
    coordinates: Literal["si", "nondimensional"] = "si"


#: Builds a run from the ``model`` and ``scenario`` mappings of a run spec.
Builder = Callable[[Mapping[str, Any]], BuiltRun]
#: Raises if a run fails a check: called with the run spec, the model and the
#: parameters the spec gives the check.
Check = Callable[..., None]

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclasses.dataclass(frozen=True)
class Component:
    """What a component registers: its models, and what it knows about them.

    A component holds one at module level and names it under the entry-point
    group ``sesmax.components``, so the run loop finds it and imports no
    component by name.

    Args:
        name: The component, e.g. ``"ocean"``.
        models: ``name -> builder``, the names a run spec may ask for.
        checks: ``name -> check``, beyond the run loop's own.
        evaluate: Scalar metrics of a model and a state, or ``None``.
        field_units: ``model name -> field -> unit``, for logs and files.
        describe_numerics: What a manifest records of a model's numerics (its
            stepper, viscosity, scheme, …), or ``None``.
    """

    name: str
    models: Mapping[str, Builder]
    checks: Mapping[str, Check] = _EMPTY
    evaluate: Callable[[Model, State], dict[str, float]] | None = None
    field_units: Mapping[str, Mapping[str, str]] = _EMPTY
    describe_numerics: Callable[[Model], dict[str, Any]] | None = None
