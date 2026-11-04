"""2D nonlinear convection: du/dt + u*grad(u) = 0."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    enforce_periodic,
)
from jaxtyping import Array, PyTree

from sesmax_core._src._validation import require_positive
from sesmax_core._src.model import Model
from sesmax_core._src.scales import Scales
from sesmax_core._src.testing.pde2d._advective_form import advective_tendency
from sesmax_core._src.types import Diagnostics, State


class NonlinearConvection2DState(State):
    """State for 2D nonlinear convection.

    Args:
        u: x-velocity on T-points, shape ``(Ny, Nx)``.
        v: y-velocity on T-points, shape ``(Ny, Nx)``.
    """

    u: Array
    v: Array

    # Both components are collocated at T-points, not staggered.
    mask_locations: ClassVar[dict[str, str]] = {"u": "h", "v": "h"}


class NonlinearConvection2DDiagnostics(Diagnostics):
    """Diagnostics for 2D nonlinear convection.

    Args:
        kinetic_energy: Integrated KE = 0.5 * (u^2 + v^2).
    """

    kinetic_energy: Array


class NonlinearConvection2D(Model):
    """2D nonlinear convection on an Arakawa C-grid.

    Solves the system::

        du/dt + u * du/dx + v * du/dy = 0
        dv/dt + u * dv/dx + v * dv/dy = 0

    using upwind flux reconstruction.

    Args:
        grid: 2D Arakawa C-grid.
        advection: Advection operator.
        interp: Interpolation operators.
        diff: Difference operators (for the divergence).
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        method: Reconstruction method (default ``"upwind1"``).
    """

    grid: CartesianGrid2D = eqx.field(static=True)
    advection: FVXAdvection2D
    interp: Interpolation2D
    diff: Difference2D
    mask: Mask2D | None
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearConvection2DState:
        """Compute tendency via upwind advection."""
        du_dt, dv_dt = advective_tendency(
            state.u,
            state.v,
            self.advection,
            self.interp,
            self.diff,
            method=self.method,
        )
        return NonlinearConvection2DState(u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearConvection2DState:
        """Apply periodic boundary conditions."""
        return NonlinearConvection2DState(
            u=enforce_periodic(state.u),
            v=enforce_periodic(state.v),
        )

    def diagnose(self, state: PyTree) -> NonlinearConvection2DDiagnostics:
        """Compute kinetic energy diagnostic."""
        ui, vi = state.u[1:-1, 1:-1], state.v[1:-1, 1:-1]
        ke = 0.5 * jnp.sum(ui**2 + vi**2) * self.grid.dx * self.grid.dy
        return NonlinearConvection2DDiagnostics(kinetic_energy=ke)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        aspect: float = 1.0,
        **create_kw: Any,
    ) -> tuple[NonlinearConvection2D, Scales]:
        r"""Build the model at unit scales instead of SI coefficients.

        Non-dimensional form
        --------------------
        Advective scale set (:meth:`sesmax_core.Scales.advective`) with
        ``L = U = 1``, so ``T = 1`` and the equation reads
        ``d_t u + u . grad u = 0``. Inviscid nonlinear convection has no
        coefficient to scale out, so this factory only fixes the domain
        and reports the scale set; it exists so the family presents one
        interface.


        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            **create_kw: Forwarded to :meth:`create` (``method``, ``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, (nx, ny),
            extent=(1.0, aspect))`` gives a step in the same time unit;
            pass both cell counts and the aspect ratio, since the bound
            depends on the *smaller* spacing.
        """
        context = "NonlinearConvection2D.from_nondimensional"
        require_positive(context, aspect=aspect)
        model = NonlinearConvection2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            **create_kw,
        )
        return model, Scales.advective(L=1.0, U=1.0)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        method: str = "upwind1",
        mask: Mask2D | None = None,
    ) -> NonlinearConvection2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            method: Advection reconstruction method.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``NonlinearConvection2D`` model instance.
        """
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        advection = FVXAdvection2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)
        diff = Difference2D(grid=grid, mask=mask)
        return NonlinearConvection2D(
            grid=grid,
            advection=advection,
            interp=interp,
            diff=diff,
            mask=mask,
            method=method,
        )
