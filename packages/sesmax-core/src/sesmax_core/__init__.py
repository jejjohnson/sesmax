"""The core of sesmax: what every simple Earth-system model shares.

``sesmax_core`` will hold the model-agnostic machinery moved from somax: the
state and parameter types, the term algebra, the model base classes, time
stepping, forcing and its bases, non-dimensionalisation, the layered-fluid
pieces an ocean and an atmosphere share, the data-assimilation adapters,
training rollouts, monitors, guards and I/O.

It is empty until phase C1 of ``docs/design/workspace_and_core.md``; phase C0
only builds the workspace around it.
"""

from __future__ import annotations


__version__ = "0.0.0"  # x-release-please-version

__all__ = ["__version__"]
