"""2D Burgers equation: du/dt + u*grad(u) = nu * laplacian(u)."""

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
from sesmax_core._src.fields import each, values
from sesmax_core._src.model import Model
from sesmax_core._src.scales import Scales
from sesmax_core._src.testing.pde2d._advective_form import PERIODIC, advective_tendency
from sesmax_core._src.types import Diagnostics, Params, State, as_parameter


#: The default scheme of the advective face values.
UPWIND = Upwind(1)


class Burgers2DParams(Params):
    """Differentiable parameters for 2D Burgers equation.

    Args:
        nu: Kinematic viscosity (diffusion coefficient).
    """

    nu: Array


class Burgers2DState(State):
    """State for 2D Burgers equation.

    Args:
        u: x-velocity on T-points, shape ``(Ny, Nx)``.
        v: y-velocity on T-points, shape ``(Ny, Nx)``.
    """

    u: Array | CField
    v: Array | CField


class Burgers2DDiagnostics(Diagnostics):
    """Diagnostics for 2D Burgers equation.

    Args:
        kinetic_energy: Integrated KE.
    """

    kinetic_energy: Array


class Burgers2D(Model):
    r"""2D Burgers equation on an Arakawa C-grid.

    Solves the system::

        du/dt + u * du/dx + v * du/dy = nu * laplacian(u)
        dv/dt + u * dv/dx + v * dv/dy = nu * laplacian(v)

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 2D Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        scheme: Scheme of the advective face values (default ``Upwind(1)``).
    """

    params: Burgers2DParams
    grid: CartesianCGrid2D
    ops: GridOps
    mask: Mask2D | None
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> Burgers2DState:
        r"""Compute tendency: advection + diffusion for (u, v)."""
        # Advection
        adv_u, adv_v = advective_tendency(
            state.u,
            state.v,
            self.ops,
            scheme=self.scheme,
        )

        # Diffusion
        def total(advection: Array, lap: Array) -> Array:
            return advection + self.params.nu * lap

        return Burgers2DState(
            u=each(total, adv_u, self.ops.laplacian(state.u)),
            v=each(total, adv_v, self.ops.laplacian(state.v)),
        )

    def apply_boundary_conditions(self, state: PyTree) -> Burgers2DState:
        """Apply periodic boundary conditions."""
        bcs = Burgers2DState(u=PERIODIC, v=PERIODIC)
        return fill_ghosts(state, self.grid, bcs)

    def diagnose(self, state: PyTree) -> Burgers2DDiagnostics:
        """Compute kinetic energy diagnostic."""
        ui, vi = values(state.u)[1:-1, 1:-1], values(state.v)[1:-1, 1:-1]
        ke = (
            0.5
            * jnp.sum(ui**2 + vi**2)
            * self.grid.spacing("x")
            * self.grid.spacing("y")
        )
        return Burgers2DDiagnostics(kinetic_energy=ke)

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        aspect: float = 1.0,
        reynolds: float,
        **create_kw: Any,
    ) -> tuple[Burgers2D, Scales]:
        r"""Build the model at unit scales instead of SI coefficients.

        Non-dimensional form
        --------------------
        Advective scale set (:meth:`sesmax_core.Scales.advective`) with
        ``L = U = 1``, so ``T = L/U = 1`` and the equation reads
        ``d_t u + u . grad u = Re**-1 laplacian(u)``. The Reynolds
        number is the only free number: ``nu = 1/Re``.


        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            reynolds: Reynolds number ``Re = U L / nu``; sets
                ``nu = 1/Re``.
            **create_kw: Forwarded to :meth:`create` (``scheme``, ``mask``).

        Returns:
            ``(model, scales)``. ``scales.dt_from_cfl(C, (nx, ny),
            extent=(1.0, aspect))`` gives a step in the same time unit;
            pass both cell counts and the aspect ratio, since the bound
            depends on the *smaller* spacing. For the diffusive bound
            add ``mode="diffusive", diffusivity=1/reynolds`` — the
            scale set cannot know the model's Reynolds number.

        Raises:
            ValueError: If an input is not positive.
        """
        context = "Burgers2D.from_nondimensional"
        require_positive(context, aspect=aspect)
        require_positive(context, reynolds=reynolds)
        model = Burgers2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            nu=1.0 / reynolds,
            **create_kw,
        )
        return model, Scales.advective(L=1.0, U=1.0)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 2.0,
        Ly: float = 2.0,
        nu: float = 0.01,
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
    ) -> Burgers2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            nu: Kinematic viscosity.
            scheme: Scheme of the advective face values.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``Burgers2D`` model instance.
        """
        grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = Burgers2DParams(nu=as_parameter(nu))
        ops = GridOps(grid, mask)
        return Burgers2D(
            params=params,
            grid=grid,
            ops=ops,
            mask=mask,
            scheme=scheme,
        )
