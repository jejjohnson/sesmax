"""The checks a spec asks for, before a run."""

from __future__ import annotations

from sesmax_core import Model
from sesmax_core.component import Component
from sesmax_experiment._src.sim.spec import RunSpec


def preflight(spec: RunSpec, model: Model, component: Component) -> None:
    """Run the checks of ``spec.assertions`` before any step is taken.

    Args:
        spec: The spec of the run.
        model: The model that was built.
        component: Where the checks come from.

    Raises:
        KeyError: If the spec asks for a check the component does not have.
    """
    unknown = sorted(set(spec.assertions) - set(component.checks))
    if unknown:
        raise KeyError(
            f"the component {component.name!r} has no check {unknown}; it has "
            f"{sorted(component.checks)}"
        )
    for name, params in spec.assertions.items():
        component.checks[name](spec, model, **params)
