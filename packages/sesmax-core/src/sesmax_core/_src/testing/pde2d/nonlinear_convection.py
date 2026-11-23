"""2D nonlinear convection: du/dt + u*grad(u) = 0."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from gridax import (
    CartesianCGrid2D,
    CField,
    GridOps,
    Mask2D,
    fill_ghosts,
)
from gridax.schemes import Scheme, Upwind
from jaxtyping import Array, PyTree

from sesmax_core._src._validation import require_positive
from sesmax_core._src.fields import values
from sesmax_core._src.model import Model
from sesmax_core._src.scales import Scales
from sesmax_core._src.testing.pde2d._advective_form import PERIODIC, advective_tendency
from sesmax_core._src.types import Diagnostics, State


#: The default scheme of the advective face values.
UPWIND = Upwind(1)


class NonlinearConvection2DState(State):
    """State for 2D nonlinear convection.

    Args:
        u: x-velocity on T-points, shape ``(Ny, Nx)``.
        v: y-velocity on T-points, shape ``(Ny, Nx)``.
    """

    u: Array | CField
    v: Array | CField


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
        ops: gridax's operations on the grid and the mask.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        scheme: Scheme of the advective face values (default ``Upwind(1)``).
    """

    grid: CartesianCGrid2D
    ops: GridOps
    mask: Mask2D | None
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearConvection2DState:
        """Compute tendency via upwind advection."""
        du_dt, dv_dt = advective_tendency(
            state.u,
            state.v,
            self.ops,
            scheme=self.scheme,
        )
        return NonlinearConvection2DState(u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearConvection2DState:
        """Apply periodic boundary conditions."""
        bcs = NonlinearConvection2DState(u=PERIODIC, v=PERIODIC)
        return fill_ghosts(state, self.grid, bcs)

    def diagnose(self, state: PyTree) -> NonlinearConvection2DDiagnostics:
        """Compute kinetic energy diagnostic."""
        ui, vi = values(state.u)[1:-1, 1:-1], values(state.v)[1:-1, 1:-1]
        ke = (
            0.5
            * jnp.sum(ui**2 + vi**2)
            * self.grid.spacing("x")
            * self.grid.spacing("y")
        )
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
            **create_kw: Forwarded to :meth:`create` (``scheme``, ``mask``).

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
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
    ) -> NonlinearConvection2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            scheme: Scheme of the advective face values.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``NonlinearConvection2D`` model instance.
        """
        grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)
        ops = GridOps(grid, mask)
        return NonlinearConvection2D(
            grid=grid,
            ops=ops,
            mask=mask,
            scheme=scheme,
        )
