"""1D Burgers equation: du/dt + u du/dx = nu * d2u/dx2."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from gridax import (
    CartesianCGrid1D,
    CField,
    GridOps,
    Mask1D,
)
from gridax.schemes import Scheme, Upwind
from jaxtyping import Array, PyTree

from sesmax_core._src._validation import require_positive
from sesmax_core._src.fields import each, values
from sesmax_core._src.model import Model
from sesmax_core._src.scales import Scales
from sesmax_core._src.testing.pde1d._burgers_flux import burgers_tendency
from sesmax_core._src.testing.pde1d._ghosts import fill_ends
from sesmax_core._src.types import Diagnostics, Params, State, as_parameter


#: The default scheme of the advective face values.
UPWIND = Upwind(1)


class Burgers1DParams(Params):
    """Differentiable parameters for 1D Burgers equation.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Burgers1DState(State):
    """State for 1D Burgers equation.

    Args:
        u: Scalar field on T-points, shape ``(Nx,)`` including ghost cells.
    """

    u: Array | CField


class Burgers1DDiagnostics(Diagnostics):
    """Diagnostics for 1D Burgers equation.

    Args:
        energy: Integrated energy 0.5 * sum(u^2).
    """

    energy: Array


class Burgers1D(Model):
    r"""1D Burgers equation on an Arakawa C-grid.

    Solves ``du/dt + u * du/dx = nu * d²u/dx²``, combining nonlinear
    advection with viscous diffusion. The viscosity ``nu`` is learnable
    and visible to ``jax.grad``.

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 1D Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
        periodic: Whether to use periodic boundary conditions.
        scheme: Scheme of the advective face values (default ``Upwind(1)``).
    """

    params: Burgers1DParams
    grid: CartesianCGrid1D
    ops: GridOps
    mask: Mask1D | None
    periodic: bool = eqx.field(static=True, default=True)
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Burgers1DState:
        r"""Compute tendency: du/dt = -u * du/dx + nu * d²u/dx²."""
        advection_term = burgers_tendency(
            state.u,
            self.ops,
            periodic=self.periodic,
            scheme=self.scheme,
        )
        # du/dt = -d(u²/2)/dx + nu ∇²u
        du_dt = each(
            lambda advection, lap: advection + self.params.nu * lap,
            advection_term,
            self.ops.laplacian(state.u),
        )
        return Burgers1DState(u=du_dt)

    def apply_boundary_conditions(self, state: PyTree) -> Burgers1DState:
        """Apply boundary conditions to the state."""
        return Burgers1DState(u=fill_ends(state.u, self.grid, periodic=self.periodic))

    def diagnose(self, state: PyTree) -> Burgers1DDiagnostics:
        """Compute energy diagnostic."""
        energy = 0.5 * jnp.sum(values(state.u)[1:-1] ** 2) * self.grid.spacing("x")
        return Burgers1DDiagnostics(energy=energy)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 100,
        reynolds: float,
        **create_kw: Any,
    ) -> tuple[Burgers1D, Scales]:
        r"""Build the model at unit scales instead of SI coefficients.

        Non-dimensional form
        --------------------
        Advective scale set (:meth:`sesmax_core.Scales.advective`) with
        ``L = U = 1``, so ``T = L/U = 1`` and the equation reads
        ``d_t u + u . grad u = Re**-1 laplacian(u)``. The Reynolds
        number is the only free number: ``nu = 1/Re``.


        Args:
            nx: Interior grid cells.
            reynolds: Reynolds number ``Re = U L / nu``; sets
                ``nu = 1/Re``.
            **create_kw: Forwarded to :meth:`create` (``periodic``,
                ``scheme``, ``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, nx)`` gives a
            step in the same time unit; for the diffusive bound pass
            ``mode="diffusive", diffusivity=1/reynolds``, since the
            scale set cannot know the model's Reynolds number.

        Raises:
            ValueError: If an input is not positive.
        """
        context = "Burgers1D.from_nondimensional"
        require_positive(context, reynolds=reynolds)
        model = Burgers1D.create(
            nx=nx,
            Lx=1.0,
            nu=1.0 / reynolds,
            **create_kw,
        )
        return model, Scales.advective(L=1.0, U=1.0)

    @staticmethod
    def create(
        nx: int = 100,
        Lx: float = 2.0,
        nu: float = 0.01,
        periodic: bool = True,
        scheme: Scheme = UPWIND,
        mask: Mask1D | None = None,
    ) -> Burgers1D:
        """Convenience factory.

        Args:
            nx: Number of interior grid cells.
            Lx: Domain length.
            nu: Kinematic viscosity.
            periodic: Use periodic BCs if True, Dirichlet if False.
            scheme: Scheme of the advective face values.
            mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``Burgers1D`` model instance.
        """
        grid = CartesianCGrid1D.from_interior(nx, Lx)
        params = Burgers1DParams(nu=as_parameter(nu))
        ops = GridOps(grid, mask)
        return Burgers1D(
            params=params,
            grid=grid,
            ops=ops,
            mask=mask,
            periodic=periodic,
            scheme=scheme,
        )
