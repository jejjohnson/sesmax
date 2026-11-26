"""Public surface for a component of the Earth system that can be run.

:class:`Component` is what a package of models registers under the entry-point
group ``sesmax.components``, and :class:`BuiltRun` what one of its builders
returns: a model and the state it starts from. They sit in the core, so a
component registers without importing ``sesmax_experiment``, which runs it.

Not re-exported from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.component import (
    AssertionFailedError,
    Builder,
    BuiltRun,
    Check,
    Component,
)


__all__ = ["AssertionFailedError", "Builder", "BuiltRun", "Check", "Component"]
