"""The ported models with the numerics somax had, where a fix has changed them.

Each fix since the port moves numbers, so a model of `sesmax_ocean` no longer
gives somax's. These subclasses put one piece of somax's back per method, and the
equality tests run them: whatever is not overridden here still reproduces somax.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx

from sesmax_core import Model
from sesmax_ocean.qg import ReparameterizedQG
from sesmax_ocean.spherical import SphericalSWM
from sesmax_ocean.swm import MultilayerShallowWater2D, NonlinearShallowWater2D


class _SomaxNumerics:
    """The pieces of somax's shallow-water models."""

    def _mass_tendency(self, h: Any, u: Any, v: Any) -> Any:
        """No flux through the edge faces of the array."""
        return self.ops.advect(h, u, v, scheme=self.scheme)  # type: ignore[attr-defined]


class _Nonlinear(_SomaxNumerics, NonlinearShallowWater2D):
    pass


class _Multilayer(_SomaxNumerics, MultilayerShallowWater2D):
    pass


class _Spherical(_SomaxNumerics, SphericalSWM):
    pass


_CLASSES = {
    NonlinearShallowWater2D: _Nonlinear,
    MultilayerShallowWater2D: _Multilayer,
    SphericalSWM: _Spherical,
}


def with_somax_numerics(model: Model) -> Model:
    """``model`` with the pieces that were fixed after the port put back."""
    if isinstance(model, ReparameterizedQG):
        return eqx.tree_at(lambda m: m.swm, model, with_somax_numerics(model.swm))
    if type(model) not in _CLASSES:
        return model
    fields = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)}
    return _CLASSES[type(model)](**fields)
