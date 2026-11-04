"""1D nonlinear convection (inviscid Burgers): du/dt + u du/dx = 0."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import Advection1D, CartesianGrid1D, Interpolation1D, Mask1D
from jaxtyping import Array, PyTree

from sesmax_core._src.model import Model
from sesmax_core._src.scales import Scales
from sesmax_core._src.testing.pde1d._burgers_flux import burgers_tendency
from sesmax_core._src.testing.pde1d._ghosts import fill_ghosts
from sesmax_core._src.types import Diagnostics, State


class NonlinearConvection1DState(State):
    """State for 1D nonlinear convection.

    Args:
        u: Scalar field on T-points, shape ``(Nx,)`` including ghost cells.
    """

    u: Array

    # ``u`` here is a T-point scalar, not a C-grid velocity.
    mask_locations: ClassVar[dict[str, str]] = {"u": "h"}


class NonlinearConvection1DDiagnostics(Diagnostics):
    """Diagnostics for 1D nonlinear convection.

    Args:
        energy: Integrated energy 0.5 * sum(u^2).
    """

    energy: Array


class NonlinearConvection1D(Model):
    """1D nonlinear convection (inviscid Burgers) on an Arakawa C-grid.

    Solves ``du/dt + u * du/dx = 0`` using upwind flux reconstruction.

    Args:
        grid: 1D Arakawa C-grid.
        advection: Advection operator.
        interp: Interpolation operator.
        mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
        periodic: Whether to use periodic boundary conditions.
        method: Reconstruction method for advection (default ``"upwind1"``).
    """

    grid: CartesianGrid1D = eqx.field(static=True)
    advection: Advection1D
    interp: Interpolation1D
    mask: Mask1D | None
    periodic: bool = eqx.field(static=True, default=True)
    method: str = eqx.field(static=True, default="upwind1")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearConvection1DState:
        """Compute tendency: du/dt = -d(u^2 / 2)/dx via upwind flux."""
        du_dt = burgers_tendency(
            state.u,
            self.advection,
            self.interp,
            periodic=self.periodic,
            method=self.method,
        )
        return NonlinearConvection1DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearConvection1DState:
        """Apply boundary conditions to the state."""
        return NonlinearConvection1DState(
            u=fill_ghosts(state.u, periodic=self.periodic)
        )

    def diagnose(self, state: PyTree) -> NonlinearConvection1DDiagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * jnp.sum(state.u[1:-1] ** 2) * self.grid.dx
        return NonlinearConvection1DDiagnostics(energy=energy)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 100,
        **create_kw: Any,
    ) -> tuple[NonlinearConvection1D, Scales]:
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
            nx: Interior grid cells.
            **create_kw: Forwarded to :meth:`create` (``periodic``,
                ``method``, ``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, nx)`` gives a
            step in the same time unit.
        """
        model = NonlinearConvection1D.create(
            nx=nx,
            Lx=1.0,
            **create_kw,
        )
        return model, Scales.advective(L=1.0, U=1.0)

    @staticmethod
    def create(
        nx: int = 100,
        Lx: float = 2.0,
        periodic: bool = True,
        method: str = "upwind1",
        mask: Mask1D | None = None,
    ) -> NonlinearConvection1D:
        """Convenience factory.

        Args:
            nx: Number of interior grid cells.
            Lx: Domain length.
            periodic: Use periodic BCs if True, Dirichlet if False.
            method: Advection reconstruction method.
            mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``NonlinearConvection1D`` model instance.
        """
        grid = CartesianGrid1D.from_interior(nx, Lx)
        advection = Advection1D(grid=grid, mask=mask)
        interp = Interpolation1D(grid=grid, mask=mask)
        return NonlinearConvection1D(
            grid=grid,
            advection=advection,
            interp=interp,
            mask=mask,
            periodic=periodic,
            method=method,
        )
