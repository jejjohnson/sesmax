"""2D nonlinear shallow water model (vector-invariant form)."""

from __future__ import annotations

import operator
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from geophysix.rotation import beta_plane
from gridax import (
    CartesianCGrid2D,
    CField,
    GridOps,
    Mask2D,
    area_sum,
    kinetic_energy,
)
from gridax.schemes import Scheme
from jaxtyping import Array, Float, PyTree

from sesmax_core import (
    Diagnostics,
    Model,
    Params,
    PhysConsts,
    Scales,
    State,
    as_parameter,
)
from sesmax_core.fields import each, placed, values
from sesmax_core.gfd import resolve_burger
from sesmax_core.validation import (
    reject_derived_kwargs,
    require_non_negative,
    require_positive,
)
from sesmax_ocean._src.swm._shared import BCS, UPWIND, WRAPS, fill_box, wrap


class NonlinearSW2DState(State):
    """State for the 2D nonlinear shallow water model.

    Args:
        h: Total layer thickness, shape ``(Ny, Nx)``.
        u: x-velocity on U-points, shape ``(Ny, Nx)``.
        v: y-velocity on V-points, shape ``(Ny, Nx)``.
    """

    h: Float[Array, "Ny Nx"] | CField
    u: Float[Array, "Ny Nx"] | CField
    v: Float[Array, "Ny Nx"] | CField


class NonlinearSW2DParams(Params):
    """Differentiable parameters for the 2D nonlinear shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu.
        bottom_drag: Linear bottom drag coefficient kappa.
        wind_amplitude: Wind stress amplitude (m/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class NonlinearSW2DPhysConsts(PhysConsts):
    """Frozen physical constants for the 2D nonlinear shallow water model.

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


class NonlinearSW2DDiagnostics(Diagnostics):
    """Diagnostics for the 2D nonlinear shallow water model.

    Args:
        energy: Domain-integrated total energy.
        enstrophy: Domain-integrated potential enstrophy.
        mass: Domain-integrated mass/volume ``dx*dy * sum h`` — conserved
            to machine precision by the flux-form mass equation.
        casimir_q3: PV Casimir ``dx*dy * sum q^3 * h`` — higher-moment
            material invariant.
        potential_vorticity: PV field q = (ζ+f)/h at X-points.
        relative_vorticity: ζ = dv/dx - du/dy at X-points.
        kinetic_energy_field: KE at T-points.
    """

    energy: Array
    enstrophy: Array
    mass: Array
    casimir_q3: Array
    potential_vorticity: Float[Array, "Ny Nx"]
    relative_vorticity: Float[Array, "Ny Nx"]
    kinetic_energy_field: Float[Array, "Ny Nx"]

    def invariants(self) -> dict[str, Array]:
        """Mass (tight), total energy, potential enstrophy, PV Casimir."""
        return {
            "mass": self.mass,
            "total_energy": self.energy,
            "potential_enstrophy": self.enstrophy,
            "casimir_q3": self.casimir_q3,
        }


class NonlinearShallowWater2D(Model):
    r"""2D nonlinear shallow water model (vector-invariant form).

    Solves the rotating shallow water equations in vector-invariant
    form on an Arakawa C-grid::

        dh/dt = -div(h*u)
        du/dt = +q·h̄v - ∂P/∂x + nu*laplacian(u) - kappa*u
        dv/dt = -q·h̄u - ∂P/∂y + nu*laplacian(v) - kappa*v

    where q = (ζ+f)/h is potential vorticity and P = KE + g·h
    is the Bernoulli potential.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: 2D Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        f_field: Precomputed Coriolis field f(y).
        wind_stress_x: Precomputed x-wind stress pattern (normalised).
        wind_stress_y: Precomputed y-wind stress pattern (normalised).
        bc_type: Boundary condition type.
        scheme: Scheme of the face values in the mass equation.
    """

    params: NonlinearSW2DParams
    consts: NonlinearSW2DPhysConsts = eqx.field(static=True)
    grid: CartesianCGrid2D
    ops: GridOps
    mask: Mask2D | None
    f_field: Float[Array, "Ny Nx"]
    wind_stress_x: Float[Array, "Ny Nx"]
    wind_stress_y: Float[Array, "Ny Nx"]
    bc_type: str = eqx.field(static=True, default="periodic")
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def _wrap(self, field: Any, at: str) -> Any:
        """``field`` with its ghosts filled on the periodic faces."""
        return wrap(field, self.grid, WRAPS[self.bc_type], at)

    def _mass_tendency(self, h: Any, u: Any, v: Any) -> Any:
        """``-div(h u)``, with the boundary condition of ``h`` at the edge faces."""
        return self.ops.advect(h, u, v, scheme=self.scheme, bc=BCS[self.bc_type]["h"])

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NonlinearSW2DState:
        """Compute tendencies for the nonlinear shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # 1. Mass equation: dh/dt = -div(hu, hv)
        dh_dt = self._mass_tendency(h, u, v)

        # 2. Potential vorticity: q = (zeta + f) / h at X-points
        f = placed(self.f_field, h, "T")
        q = self.ops.sw_potential_vorticity(u, v, h, f)

        # 3. Mass fluxes h_bar * u at U-points and h_bar * v at V-points, then
        #    the cross-velocity PV fluxes (vector-invariant form):
        #    du/dt = +q_bar * (hv)_bar   (v drives u via Coriolis)
        #    dv/dt = -q_bar * (hu)_bar   (u drives v via Coriolis)
        uh = each(operator.mul, self.ops.interpolate(h, src="T", dst="U"), u)
        vh = each(operator.mul, self.ops.interpolate(h, src="T", dst="V"), v)
        # The next averages read the ghosts of these, which the operators leave
        # at zero.
        q = self._wrap(q, "X")
        uh, vh = self._wrap(uh, "U"), self._wrap(vh, "V")
        q_on_U = self.ops.interpolate(q, src="X", dst="U")
        q_on_V = self.ops.interpolate(q, src="X", dst="V")
        vh_on_U = self.ops.interpolate(vh, src="V", dst="U")
        uh_on_V = self.ops.interpolate(uh, src="U", dst="V")

        # 4. Bernoulli potential: P = KE + g*h
        ke = kinetic_energy(u, v, self.grid)
        P = each(lambda k, thickness: k + g * thickness, ke, h)

        # 5. Diffusion, by the operator for T-points on the array of a velocity
        diffusion_u = self.ops.diffuse(values(u), kappa=nu)
        diffusion_v = self.ops.diffuse(values(v), kappa=nu)

        # 6. Momentum: PV flux and pressure, then wind, diffusion, bottom drag
        def x_momentum(pv: Array, flux: Array, dP: Array, velocity: Array) -> Array:
            return (
                pv * flux
                - dP
                + tau0 * self.wind_stress_x
                + diffusion_u
                - kappa * velocity
            )

        def y_momentum(pv: Array, flux: Array, dP: Array, velocity: Array) -> Array:
            return (
                -pv * flux
                - dP
                + tau0 * self.wind_stress_y
                + diffusion_v
                - kappa * velocity
            )

        P = self._wrap(P, "T")
        dP_dx = self.ops.difference(P, axis="x", src="T", dst="U")
        dP_dy = self.ops.difference(P, axis="y", src="T", dst="V")
        du_dt = each(x_momentum, q_on_U, vh_on_U, dP_dx, u)
        dv_dt = each(y_momentum, q_on_V, uh_on_V, dP_dy, v)

        return NonlinearSW2DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> NonlinearSW2DState:
        """Apply boundary conditions to all state fields.

        For periodic: ghost cells wrap around.
        For wall (free-slip, no-normal-flow):
          - u = 0 at x-boundaries (no normal flow at east/west walls)
          - v = 0 at y-boundaries (no normal flow at north/south walls)
          - Tangential velocity: zero-gradient (free-slip)
          - h: zero-gradient at all walls
        """
        return fill_box(state, self.grid, self.bc_type)

    def diagnose(self, state: PyTree) -> NonlinearSW2DDiagnostics:
        """Compute energy, enstrophy, PV, vorticity, and KE fields."""
        h, u, v = values(state.h), values(state.u), values(state.v)
        grid = self.grid
        ke = kinetic_energy(u, v, grid)  # T
        zeta = self.ops.curl(u, v)  # X
        q = self.ops.sw_potential_vorticity(u, v, h, self.f_field)  # X
        # Total energy: KE + PE
        energy = area_sum(ke + 0.5 * self.consts.gravity * h**2, grid)
        # Potential enstrophy: 0.5 * q^2 * h
        h_on_X = self.ops.interpolate(h, src="T", dst="X")
        enstrophy = area_sum(0.5 * q**2 * h_on_X, grid)
        # Mass/volume (flux-form invariant) and the q^3 Casimir.
        mass = area_sum(h, grid)
        casimir_q3 = area_sum(q**3 * h_on_X, grid)

        return NonlinearSW2DDiagnostics(
            energy=energy,
            enstrophy=enstrophy,
            mass=mass,
            casimir_q3=casimir_q3,
            potential_vorticity=q,
            relative_vorticity=zeta,
            kinetic_energy_field=ke,
        )

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        rossby: float,
        burger: float | None = None,
        froude: float | None = None,
        beta_hat: float = 0.0,
        ekman: float = 0.0,
        ekman_lateral: float = 0.0,
        wind_hat: float = 0.0,
        aspect: float = 1.0,
        **create_kw: Any,
    ) -> tuple[NonlinearShallowWater2D, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **inertial** (:meth:`sesmax_core.Scales.inertial`), with
        ``L = f0 = H = 1`` so that ``T = 1/f0 = 1`` and the velocity
        scale is ``U = f0 L Ro = Ro``. The model's own velocity fields
        are therefore ``O(Ro)``; wrap it in a
        :class:`~sesmax_core.ScaledModel` built from the returned ``Scales``
        to work in ``O(1)`` state instead.

        ==================  ==============================  ====================
        Input               Definition                      ``create()`` kwarg
        ==================  ==============================  ====================
        ``rossby``          :math:`Ro = U/(f_0 L)`          sets ``U``
        ``burger``          :math:`Bu = gH/(f_0 L)^2`       ``g``
        ``froude``          :math:`Fr = U/\sqrt{gH}`         ``g`` (via ``Bu``)
        ``beta_hat``        :math:`\beta L/f_0`              ``beta``
        ``ekman``           :math:`\kappa/f_0`               ``bottom_drag``
        ``ekman_lateral``   :math:`\nu/(f_0 L^2)`            ``lateral_viscosity``
        ``wind_hat``        :math:`\tau_0/(f_0 U)`           ``wind_amplitude``
        ==================  ==============================  ====================

        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            rossby: Rossby number; sets the velocity scale.
            burger: Burger number. Give this or ``froude``, not both.
            froude: Froude number. Equivalent to ``burger`` through
                ``Bu = (Ro/Fr)**2``.
            beta_hat: Dimensionless planetary vorticity gradient. Zero
                (the default) is an f-plane.
            ekman: Linear bottom-drag Ekman number.
            ekman_lateral: Lateral-viscosity Ekman number.
            wind_hat: Dimensionless wind acceleration.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            **create_kw: Forwarded to :meth:`create` (``bc``,
                ``scheme``, ``wind_profile``, ``mask``).

        Returns:
            ``(model, scales)`` with ``scales.kind == "inertial"``.

        Raises:
            ValueError: If an input is out of range, or if neither or
                both of ``burger`` and ``froude`` is given.
        """
        context = "NonlinearShallowWater2D.from_nondimensional"
        reject_derived_kwargs(
            context,
            (
                "Lx",
                "Ly",
                "g",
                "f0",
                "beta",
                "H0",
                "lateral_viscosity",
                "bottom_drag",
                "wind_amplitude",
            ),
            **create_kw,
        )
        require_positive(context, rossby=rossby, aspect=aspect)
        require_non_negative(
            context,
            beta_hat=beta_hat,
            ekman=ekman,
            ekman_lateral=ekman_lateral,
            wind_hat=wind_hat,
        )
        bu = resolve_burger(context, burger, froude, rossby)

        # Unit scales: L = f0 = H0 = 1, so every coefficient below is
        # the dimensionless group itself. The exception is the wind,
        # which enters the momentum equation as an acceleration and so
        # carries the velocity scale U = Ro.
        model = NonlinearShallowWater2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            g=bu,
            f0=1.0,
            beta=beta_hat,
            H0=1.0,
            lateral_viscosity=ekman_lateral,
            bottom_drag=ekman,
            wind_amplitude=wind_hat * rossby,
            **create_kw,
        )
        scales = Scales.inertial(L=1.0, f0=1.0, H=1.0, rossby=rossby, g=bu)
        return model, scales

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
        wind_amplitude: float = 0.0,
        wind_profile: str = "doublegyre",
        bc: str = "periodic",
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
    ) -> NonlinearShallowWater2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s^2).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m*s)).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind stress amplitude (m/s^2).
            wind_profile: Wind stress pattern. ``"doublegyre"``
                gives tau_x = -cos(2*pi*y/Ly), ``"single"`` gives
                tau_x = -cos(pi*y/Ly).
            bc: Boundary condition type.
            scheme: Scheme of the face values in the mass equation.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).

        Returns:
            A ``NonlinearShallowWater2D`` model instance.
        """
        grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)
        params = NonlinearSW2DParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
            wind_amplitude=as_parameter(wind_amplitude),
        )
        consts = NonlinearSW2DPhysConsts(gravity=g, f0=f0, beta=beta, H0=H0)

        y = jnp.arange(grid.Ny) * grid.spacing("y")
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        f_field = beta_plane(Y, f0=f0, beta=beta, y0=y0)

        # Wind stress pattern (normalised)
        if wind_profile == "single":
            wind_stress_x = -jnp.cos(jnp.pi * Y / Ly)
        else:
            # Double gyre: tau_x = -cos(2*pi*y/Ly)
            wind_stress_x = -jnp.cos(2.0 * jnp.pi * Y / Ly)
        wind_stress_y = jnp.zeros_like(wind_stress_x)

        return NonlinearShallowWater2D(
            params=params,
            consts=consts,
            grid=grid,
            ops=GridOps(grid, mask),
            mask=mask,
            f_field=f_field,
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y,
            bc_type=bc,
            scheme=scheme,
        )
