"""1D nonlinear shallow water model."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from gridax import (
    BC,
    CartesianCGrid1D,
    CField,
    GridOps,
    Mask1D,
    laplacian,
)
from gridax.schemes import Scheme, Upwind
from jaxtyping import Array, PyTree

from sesmax_core._src.fields import each, placed, values
from sesmax_core._src.model import Model
from sesmax_core._src.testing.pde1d._ghosts import fill_ends
from sesmax_core._src.types import Diagnostics, Params, PhysConsts, State, as_parameter


#: The default scheme of the advective face values.
UPWIND = Upwind(1)


class NonlinearSW1DState(State):
    """State for the 1D nonlinear shallow water model.

    Args:
        h: Total layer thickness, shape ``(Nx,)``.
        u: x-velocity on U-points, shape ``(Nx,)``.
        v: y-velocity on T-points (Coriolis coupling), shape ``(Nx,)``.
    """

    h: Array | CField
    u: Array | CField
    v: Array | CField


class NonlinearSW1DParams(Params):
    """Differentiable parameters for the 1D nonlinear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
    """

    lateral_viscosity: Array
    bottom_drag: Array


class NonlinearSW1DPhysConsts(PhysConsts):
    """Frozen physical constants for the 1D nonlinear shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Coriolis parameter (f-plane).
        H0: Mean layer depth (for diagnostics only).
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    H0: float = eqx.field(static=True, default=100.0)


class NonlinearSW1DDiagnostics(Diagnostics):
    """Diagnostics for the 1D nonlinear shallow water model.

    Args:
        energy: Domain-integrated energy.
        mass: Domain-integrated mass (h).
    """

    energy: Array
    mass: Array


class NonlinearShallowWater1D(Model):
    r"""1D nonlinear shallow water model on an Arakawa C-grid.

    Solves the nonlinear shallow water equations::

        dh/dt = -d(h·u)/dx
        du/dt = -u·du/dx - g·dh/dx + f₀·v + nu*laplacian(u) - kappa*u
        dv/dt = -f₀·u                      + nu*laplacian(v) - kappa*v

    Args:
        params: Differentiable parameters (viscosity, drag).
        consts: Frozen physical constants (g, f₀, H₀).
        grid: 1D Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
        scheme: Scheme of the advective face values.
    """

    params: NonlinearSW1DParams
    consts: NonlinearSW1DPhysConsts = eqx.field(static=True)
    grid: CartesianCGrid1D
    ops: GridOps
    mask: Mask1D | None
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearSW1DState:
        """Compute tendencies for the nonlinear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        f0 = self.consts.f0
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag

        # The domain is periodic: a closed wall would write no tendency on the
        # outermost cells. A masked one keeps closed walls.
        bc = BC.periodic("x") if self.mask is None else None

        # Mass equation: dh/dt = -d(hu)/dx (via advection operator)
        dh_dt = self.ops.advect(h, u, None, scheme=self.scheme, bc=bc)

        # Momentum advection: -u * du/dx, implemented in conservative
        # flux form as -d(u^2 / 2)/dx using the existing advection operator.
        # The momentum cells are the U points, so u is the cell value too.
        u_cells = placed(values(u), u, "T")
        flux_form = values(self.ops.advect(u_cells, u, None, scheme=self.scheme, bc=bc))
        du_adv = placed(flux_form, u, "U")

        # Pressure gradient: -g * dh/dx
        dh_dx = self.ops.difference(h, axis="x", src="T", dst="U")

        # Coriolis
        v_on_U = self.ops.interpolate(v, src="T", dst="U")
        u_on_T = self.ops.interpolate(u, src="U", dst="T")

        # Advection, pressure gradient and Coriolis, then diffusion and drag
        du_dt = each(
            lambda adv, dh, v_u, lap, u: (
                0.5 * adv + -g * dh + f0 * v_u + nu * lap - kappa * u
            ),
            du_adv,
            dh_dx,
            v_on_U,
            laplacian(u, self.grid, at="U", mask=self.mask),
            u,
        )
        dv_dt = each(
            lambda u_t, lap, v: -f0 * u_t + nu * lap - kappa * v,
            u_on_T,
            laplacian(v, self.grid, at="T", mask=self.mask),
            v,
        )

        return NonlinearSW1DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearSW1DState:
        """Apply periodic boundary conditions."""
        return NonlinearSW1DState(
            h=fill_ends(state.h, self.grid, periodic=True, at="T"),
            u=fill_ends(state.u, self.grid, periodic=True, at="U"),
            v=fill_ends(state.v, self.grid, periodic=True, at="T"),
        )

    def diagnose(self, state: PyTree) -> NonlinearSW1DDiagnostics:
        """Compute energy and mass."""
        g = self.consts.gravity
        dx = self.grid.spacing("x")
        h, u, v = values(state.h), values(state.u), values(state.v)
        interior = slice(1, -1)
        energy = (
            0.5
            * dx
            * jnp.sum(
                g * h[interior] ** 2
                + h[interior] * (u[interior] ** 2 + v[interior] ** 2)
            )
        )
        mass = dx * jnp.sum(h[interior])
        return NonlinearSW1DDiagnostics(energy=energy, mass=mass)

    @staticmethod
    def create(
        nx: int = 200,
        Lx: float = 1e6,
        g: float = 9.81,
        f0: float = 1e-4,
        H0: float = 100.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        scheme: Scheme = UPWIND,
        mask: Mask1D | None = None,
    ) -> NonlinearShallowWater1D:
        """Convenience factory.

        Args:
            nx: Number of interior cells.
            Lx: Domain length (m).
            g: Gravitational acceleration (m/s²).
            f0: Coriolis parameter (1/s).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m²/s).
            bottom_drag: Linear bottom drag (1/s).
            scheme: Scheme of the advective face values.
            mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``NonlinearShallowWater1D`` model instance.
        """
        grid = CartesianCGrid1D.from_interior(nx, Lx)
        params = NonlinearSW1DParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
        )
        consts = NonlinearSW1DPhysConsts(gravity=g, f0=f0, H0=H0)
        ops = GridOps(grid, mask)
        return NonlinearShallowWater1D(
            params=params,
            consts=consts,
            grid=grid,
            ops=ops,
            mask=mask,
            scheme=scheme,
        )
