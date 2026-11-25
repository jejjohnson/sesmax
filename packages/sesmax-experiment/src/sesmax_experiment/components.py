"""Public surface for finding what can be run.

A package of models registers a :class:`sesmax_core.component.Component` under the
entry-point group ``sesmax.components``, and the run loop finds it here by the name
of a model.
"""

from __future__ import annotations

from sesmax_experiment._src.components import GROUP, components, find_model


__all__ = ["GROUP", "components", "find_model"]
