"""1D linear shallow water model."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
from gridax import (
    CartesianCGrid1D,
    CField,
    Difference1D,
    Interpolation1D,
    Mask1D,
    laplacian,
)
from jaxtyping import Array, PyTree

from sesmax_core._src.fields import each, values
from sesmax_core._src.model import Model
from sesmax_core._src.testing.pde1d._ghosts import fill_ends
from sesmax_core._src.types import Diagnostics, Params, PhysConsts, State, as_parameter


class LinearSW1DState(State):
    """State for the 1D linear shallow water model.

    Args:
        h: Height perturbation (η = h - H₀), shape ``(Nx,)``.
        u: x-velocity on U-points, shape ``(Nx,)``.
        v: y-velocity on T-points (Coriolis coupling), shape ``(Nx,)``.
    """

    h: Array | CField
    u: Array | CField
    v: Array | CField

    # ``h`` is the perturbation, so it is already centred on zero.
    scale_kinds: ClassVar[dict[str, str]] = {"h": "height_anomaly"}


class LinearSW1DParams(Params):
    """Differentiable parameters for the 1D linear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
    """

    lateral_viscosity: Array
    bottom_drag: Array


class LinearSW1DPhysConsts(PhysConsts):
    """Frozen physical constants for the 1D linear shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Coriolis parameter (f-plane).
        H0: Mean layer depth.
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    H0: float = eqx.field(static=True, default=100.0)


class LinearSW1DDiagnostics(Diagnostics):
    """Diagnostics for the 1D linear shallow water model.

    Args:
        energy: Domain-integrated energy.
        phase_speed: Gravity wave phase speed √(gH₀).
    """

    energy: Array
    phase_speed: Array


class LinearShallowWater1D(Model):
    r"""1D linear shallow water model on an Arakawa C-grid.

    Solves the linearised shallow water equations::

        dh/dt = -H₀ · du/dx
        du/dt = -g · dh/dx + f₀·v + nu*laplacian(u) - kappa*u
        dv/dt = -f₀·u           + nu*laplacian(v) - kappa*v

    where h is the height perturbation, (u, v) are velocities,
    and the Coriolis term couples u ↔ v even in 1D.

    Args:
        params: Differentiable parameters (viscosity, drag).
        consts: Frozen physical constants (g, f₀, H₀).
        grid: 1D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).
    """

    params: LinearSW1DParams
    consts: LinearSW1DPhysConsts = eqx.field(static=True)
    grid: CartesianCGrid1D
    diff: Difference1D
    interp: Interpolation1D
    mask: Mask1D | None

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> LinearSW1DState:
        """Compute tendencies for the linear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        f0 = self.consts.f0
        H0 = self.consts.H0
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag

        # Mass equation: dh/dt = -H0 * du/dx
        dh_dt = each(lambda du: -H0 * du, self.diff.diff_x_U_to_T(u))

        # Momentum equations
        # Pressure gradient: -g * dh/dx
        dh_dx = self.diff.diff_x_T_to_U(h)

        # Coriolis: du/dt += f0*v, dv/dt -= f0*u
        # v lives on T-points, interpolate to U-points for du/dt
        v_on_U = self.interp.T_to_U(v)
        u_on_T = self.interp.U_to_T(u)

        # Pressure gradient and Coriolis, then diffusion and bottom drag
        du_dt = each(
            lambda dh, v_u, lap, u: -g * dh + f0 * v_u + nu * lap - kappa * u,
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

        return LinearSW1DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> LinearSW1DState:
        """Apply periodic boundary conditions."""
        return LinearSW1DState(
            h=fill_ends(state.h, self.grid, periodic=True, at="T"),
            u=fill_ends(state.u, self.grid, periodic=True, at="U"),
            v=fill_ends(state.v, self.grid, periodic=True, at="T"),
        )

    def diagnose(self, state: PyTree) -> LinearSW1DDiagnostics:
        """Compute energy and phase speed."""
        g = self.consts.gravity
        H0 = self.consts.H0
        h, u, v = values(state.h), values(state.u), values(state.v)
        interior = slice(1, -1)
        dx = self.grid.spacing("x")
        # Energy: 0.5 * (g*h² + H0*(u² + v²)) * dx
        energy = (
            0.5
            * dx
            * jnp.sum(g * h[interior] ** 2 + H0 * (u[interior] ** 2 + v[interior] ** 2))
        )
        phase_speed = jnp.sqrt(g * H0)
        return LinearSW1DDiagnostics(energy=energy, phase_speed=phase_speed)

    @staticmethod
    def create(
        nx: int = 200,
        Lx: float = 1e6,
        g: float = 9.81,
        f0: float = 1e-4,
        H0: float = 100.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        mask: Mask1D | None = None,
    ) -> LinearShallowWater1D:
        """Convenience factory.

        Args:
            nx: Number of interior cells.
            Lx: Domain length (m).
            g: Gravitational acceleration (m/s²).
            f0: Coriolis parameter (1/s).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m²/s).
            bottom_drag: Linear bottom drag (1/s).
            mask: Optional 1-D Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``LinearShallowWater1D`` model instance.
        """
        grid = CartesianCGrid1D.from_interior(nx, Lx)
        params = LinearSW1DParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
        )
        consts = LinearSW1DPhysConsts(gravity=g, f0=f0, H0=H0)
        diff = Difference1D(grid=grid, mask=mask)
        interp = Interpolation1D(grid=grid, mask=mask)
        return LinearShallowWater1D(
            params=params,
            consts=consts,
            grid=grid,
            diff=diff,
            interp=interp,
            mask=mask,
        )
