"""The ported models with the numerics somax had, where a fix has changed them.

Each fix since the port moves numbers, so a model of `sesmax_ocean` no longer
gives somax's. These subclasses put one piece of somax's back per method, and the
equality tests run them: whatever is not overridden here still reproduces somax.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from sesmax_core import Model
from sesmax_core.fields import placed, values
from sesmax_ocean.qg import ReparameterizedQG
from sesmax_ocean.spherical import SphericalSWM
from sesmax_ocean.swm import MultilayerShallowWater2D, NonlinearShallowWater2D


class _SomaxNumerics:
    """The pieces of somax's shallow-water models."""

    def _wrap(self, field: Any, at: str) -> Any:
        """The ghosts of the products of the tendency stay at zero."""
        return field

    def _laplacian(self, velocity: Any, at: str) -> Any:
        """Diffusion as a tracer at T, with closed outer faces."""
        diffusion = self.ops.diffuse(values(velocity), kappa=1.0)  # type: ignore[attr-defined]
        return placed(diffusion, velocity, at)

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
    if isinstance(model, SphericalSWM):
        # f at the corners, which the PV averages to the corners again.
        lat = jnp.asarray(model.grid.coords("y"))[:, None]
        f = 2.0 * model.consts.omega * jnp.sin(lat + 0.5 * model.grid.dlat)
        fields["f_field"] = jnp.broadcast_to(f, model.grid.shape)
    return _CLASSES[type(model)](**fields)
