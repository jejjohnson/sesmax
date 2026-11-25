"""sesmax-experiment: what is done with a model.

A run of a model in chunks, its monitors and its checkpoints live here, and later
the estimation of its state and parameters. The package names no component of the
Earth system: a component registers itself through
:class:`sesmax_core.component.Component`.

Everything is imported from its submodule; the top level holds the version.
"""

from __future__ import annotations


__version__ = "0.0.0"  # x-release-please-version

__all__ = ["__version__"]
