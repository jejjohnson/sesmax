"""2D linear shallow water model."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from geophysix.rotation import beta_plane
from gridax import (
    BC,
    CartesianCGrid2D,
    CField,
    Coriolis2D,
    Difference2D,
    Dirichlet,
    Interpolation2D,
    Mask2D,
    Neumann,
    Periodic,
    fill_ghosts,
    laplacian,
)
from jaxtyping import Array, Float, PyTree

from sesmax_core._src._diagnostic_ops import fvx_ops, masked
from sesmax_core._src.fields import each, placed, values
from sesmax_core._src.model import Model
from sesmax_core._src.types import Diagnostics, Params, PhysConsts, State, as_parameter


class LinearSW2DState(State):
    """State for the 2D linear shallow water model.

    Args:
        h: Height perturbation (η = h - H₀), shape ``(Ny, Nx)``.
        u: x-velocity on U-points, shape ``(Ny, Nx)``.
        v: y-velocity on V-points, shape ``(Ny, Nx)``.
    """

    h: Float[Array, "Ny Nx"] | CField
    u: Float[Array, "Ny Nx"] | CField
    v: Float[Array, "Ny Nx"] | CField

    # ``h`` is the perturbation, so it is already centred on zero.
    scale_kinds: ClassVar[dict[str, str]] = {"h": "height_anomaly"}


#: The point of each field.
POINTS = LinearSW2DState(h="T", u="U", v="V")
_BOX = BC(x=Periodic(), y=Periodic())
_FLAT, _WALL = Neumann(), Dirichlet(0.0)
#: The boundary conditions of each ``bc_type``: the doubly periodic box, and
#: closed walls with no normal flow and free slip.
BCS = {
    "periodic": LinearSW2DState(h=_BOX, u=_BOX, v=_BOX),
    "wall": LinearSW2DState(
        h=BC(x=_FLAT, y=_FLAT), u=BC(x=_WALL, y=_FLAT), v=BC(x=_FLAT, y=_WALL)
    ),
}


class LinearSW2DParams(Params):
    """Differentiable parameters for the 2D linear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
    """

    lateral_viscosity: Array
    bottom_drag: Array


class LinearSW2DPhysConsts(PhysConsts):
    """Frozen physical constants for the 2D linear shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Reference Coriolis parameter.
        beta: Meridional gradient of f (β-plane).
        H0: Mean layer depth.
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    beta: float = eqx.field(static=True, default=0.0)
    H0: float = eqx.field(static=True, default=100.0)


class LinearSW2DDiagnostics(Diagnostics):
    """Diagnostics for the 2D linear shallow water model.

    Args:
        energy: Domain-integrated energy.
        relative_vorticity: ζ = dv/dx - du/dy at X-points.
    """

    energy: Array
    relative_vorticity: Float[Array, "Ny Nx"]


class LinearShallowWater2D(Model):
    r"""2D linear shallow water model on an Arakawa C-grid.

    Solves the linearised shallow water equations::

        dh/dt = -H₀ · (du/dx + dv/dy)
        du/dt = -g · dh/dx + f·v + nu*laplacian(u) - kappa*u
        dv/dt = -g · dh/dy - f·u + nu*laplacian(v) - kappa*v

    Supports both f-plane (β=0) and β-plane Coriolis.

    Args:
        params: Differentiable parameters (viscosity, drag).
        consts: Frozen physical constants (g, f₀, β, H₀).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        coriolis: Coriolis operator.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        f_field: Precomputed Coriolis parameter field f(y).
        bc_type: Boundary condition type (``"periodic"`` or ``"wall"``).
    """

    params: LinearSW2DParams
    consts: LinearSW2DPhysConsts = eqx.field(static=True)
    grid: CartesianCGrid2D
    diff: Difference2D
    interp: Interpolation2D
    coriolis: Coriolis2D
    mask: Mask2D | None
    f_field: Float[Array, "Ny Nx"]
    bc_type: str = eqx.field(static=True, default="periodic")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> LinearSW2DState:
        """Compute tendencies for the linear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        H0 = self.consts.H0
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag

        # Mass equation: dh/dt = -H0 * div(u, v)
        dh_dt = each(lambda div: -H0 * div, self.diff.divergence(u, v))

        # Coriolis: du += f*v, dv -= f*u
        f = placed(self.f_field, u, "T")
        du_cor, dv_cor = self.coriolis(u, v, f)

        # Pressure gradient and Coriolis, then diffusion and bottom drag
        def momentum(dh: Array, cor: Array, lap: Array, velocity: Array) -> Array:
            return -g * dh + cor + nu * lap - kappa * velocity

        du_dt = each(
            momentum,
            self.diff.diff_x_T_to_U(h),
            du_cor,
            laplacian(u, self.grid, at="U", mask=self.mask),
            u,
        )
        dv_dt = each(
            momentum,
            self.diff.diff_y_T_to_V(h),
            dv_cor,
            laplacian(v, self.grid, at="V", mask=self.mask),
            v,
        )

        return LinearSW2DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> LinearSW2DState:
        """Apply boundary conditions (periodic or free-slip wall)."""
        return fill_ghosts(state, self.grid, BCS[self.bc_type], stagger=POINTS)

    def diagnose(self, state: PyTree) -> LinearSW2DDiagnostics:
        """Compute energy and relative vorticity (finitevolX Tier-B ops).

        E = ∑ (H0 KE + ½ g h²) ΔA with KE = ½(ū² + v̄²) at T-points; equal
        to ½ ∑ (g h² + H0 (u² + v²)) ΔA once the ghost ring holds the
        boundary condition (periodic or wall).
        """
        fp = fvx_ops()
        G = self.grid
        h, u, v = values(state.h), values(state.u), values(state.v)
        ke = fp.KineticEnergy(grid=G)(u, v)  # T
        ape = fp.AvailablePE(grid=G, g_prime=self.consts.gravity)(h)  # T
        energy = fp.TotalEnergy(grid=G)(self.consts.H0 * ke, ape)
        zeta = masked(self.mask, "X", fp.RelativeVorticity(grid=G)(u, v))
        return LinearSW2DDiagnostics(energy=energy, relative_vorticity=zeta)

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1e6,
        Ly: float = 1e6,
        g: float = 9.81,
        f0: float = 1e-4,
        beta: float = 0.0,
        H0: float = 100.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        bc: str = "periodic",
        mask: Mask2D | None = None,
    ) -> LinearShallowWater2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s²).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m·s)).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m²/s).
            bottom_drag: Linear bottom drag (1/s).
            bc: Boundary condition type (``"periodic"`` or ``"wall"``).
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``LinearShallowWater2D`` model instance.
        """
        grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = LinearSW2DParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
        )
        consts = LinearSW2DPhysConsts(gravity=g, f0=f0, beta=beta, H0=H0)
        diff = Difference2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)
        coriolis = Coriolis2D(grid=grid, mask=mask)

        # Build f(y) = f0 + beta * (y - y0) on full grid
        y = jnp.arange(grid.Ny) * grid.spacing("y")
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        f_field = beta_plane(Y, f0=f0, beta=beta, y0=y0)

        return LinearShallowWater2D(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            coriolis=coriolis,
            mask=mask,
            f_field=f_field,
            bc_type=bc,
        )
