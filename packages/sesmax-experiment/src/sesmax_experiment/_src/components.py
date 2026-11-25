"""The components installed beside this package, and the models they offer."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import entry_points

from sesmax_core.component import Builder, Component


#: The entry-point group a package names its :class:`Component` under.
GROUP = "sesmax.components"


def components() -> dict[str, Component]:
    """Every installed component, by name.

    Returns:
        ``name -> component``, in the order of the names.

    Raises:
        TypeError: If an entry point of the group is not a :class:`Component`.
        ValueError: If two components have one name.

    Examples:
        >>> from sesmax_experiment.components import components
        >>> "testing" in components()
        True
    """
    found: dict[str, Component] = {}
    for entry in sorted(entry_points(group=GROUP), key=lambda e: (e.name, e.value)):
        component = entry.load()
        if not isinstance(component, Component):
            raise TypeError(
                f"the entry point {entry.name!r} of {GROUP!r} is a "
                f"{type(component).__name__}, not a Component: {entry.value}"
            )
        if component.name in found:
            raise ValueError(f"two components are named {component.name!r}")
        found[component.name] = component
    return found


def find_model(
    name: str, among: Mapping[str, Component] | None = None
) -> tuple[Component, Builder]:
    """The component that offers a model, and its builder.

    Args:
        name: The name of the model in a run spec.
        among: The components to look in; the installed ones by default.

    Returns:
        The component and the builder.

    Raises:
        KeyError: If no component offers the model.
        ValueError: If two do.
    """
    among = components() if among is None else among
    owners = [c for c in among.values() if name in c.models]
    if not owners:
        offered = sorted(model for c in among.values() for model in c.models)
        raise KeyError(f"no component offers the model {name!r}; offered: {offered}")
    if len(owners) > 1:
        names = [c.name for c in owners]
        raise ValueError(f"the model {name!r} is offered by {names}")
    return owners[0], owners[0].models[name]
