"""The ported models with the numerics somax had, where a fix has changed them.

Each fix since the port moves numbers, so a model of `sesmax_ocean` no longer
gives somax's. These subclasses put one piece of somax's back per method, and the
equality tests run them: whatever is not overridden here still reproduces somax.

One change is gridax's: since v0.3 no operation with a mask reads land. finitevolX,
at the commit somax pins, computed as if all were ocean and then zeroed the dry
outputs, so a wet cell beside land read what land held. `_ReadsLand` does that
for the two operations the quasi-geostrophic models use.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from gridax import GridOps, apply_mask, grad_perp, laplacian

from sesmax_core import Model
from sesmax_core.fields import placed, values
from sesmax_ocean.qg import BaroclinicQG, BarotropicQG, ReparameterizedQG
from sesmax_ocean.spherical import SphericalQG, SphericalSWM
from sesmax_ocean.swm import MultilayerShallowWater2D, NonlinearShallowWater2D


class _ReadsLand(eqx.Module):
    """The operations of a model, with the two that read land in finitevolX."""

    ops: GridOps

    def __getattr__(self, name: str) -> Any:
        if name == "ops":
            raise AttributeError(name)
        return getattr(self.ops, name)

    def laplacian(self, field: Any, *, at: str) -> Any:
        """The cells beside land take a gradient from what land holds."""
        out = laplacian(field, self.ops.grid, at=at)
        mask = self.ops.mask
        return out if mask is None else apply_mask(out, mask, at=at)

    def grad_perp(self, psi: Any, *, at: str) -> Any:
        """A corner beside land averages what land holds."""
        mask = self.ops.mask
        u, v = grad_perp(psi, self.ops.grid, at=at)
        if mask is None:
            return u, v
        return apply_mask(u, mask, at="U"), apply_mask(v, mask, at="V")


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


class _SphericalQG(SphericalQG):
    """Velocities and fluxes that stop at the edge of the array."""

    def velocities(self, psi: Any) -> Any:
        dpsi_dlat = self.ops.difference(psi, axis="y", src="T", dst="V")
        dpsi_dlon = self.ops.difference(psi, axis="x", src="T", dst="U")
        u = -self.ops.interpolate(dpsi_dlat, src="V", dst="U")
        v = self.ops.interpolate(dpsi_dlon, src="U", dst="V")
        return u, v

    def _advection(self, q: Any, u: Any, v: Any) -> Any:
        return self.ops.advect(q, u, v, scheme=self.scheme)

    def _laplacian_of_vorticity(self, q: Any) -> Any:
        return self.ops.diffuse(q, kappa=1.0)


_CLASSES = {
    NonlinearShallowWater2D: _Nonlinear,
    MultilayerShallowWater2D: _Multilayer,
    SphericalSWM: _Spherical,
    SphericalQG: _SphericalQG,
}


def with_somax_numerics(model: Model) -> Model:
    """``model`` with the pieces that were fixed after the port put back."""
    if isinstance(model, ReparameterizedQG):
        swm = with_somax_numerics(model.swm)
        swm = eqx.tree_at(lambda m: m.ops, swm, _ReadsLand(swm.ops))
        return eqx.tree_at(lambda m: m.swm, model, swm)
    if isinstance(model, BarotropicQG | BaroclinicQG | SphericalQG):
        model = eqx.tree_at(lambda m: m.ops, model, _ReadsLand(model.ops))
    if type(model) not in _CLASSES:
        return model
    fields = {f.name: getattr(model, f.name) for f in dataclasses.fields(model)}
    if isinstance(model, SphericalSWM):
        # f at the corners, which the PV averages to the corners again.
        lat = jnp.asarray(model.grid.coords("y"))[:, None]
        f = 2.0 * model.consts.omega * jnp.sin(lat + 0.5 * model.grid.dlat)
        fields["f_field"] = jnp.broadcast_to(f, model.grid.shape)
    return _CLASSES[type(model)](**fields)
