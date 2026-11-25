"""Multilayer 2D nonlinear shallow water model (vector-invariant form)."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from functools import partial
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
    ModalTransform,
    Model,
    Params,
    PhysConsts,
    Scales,
    State,
    StratificationProfile,
    as_parameter,
)
from sesmax_core.fields import each, placed, values
from sesmax_core.gfd import burger_to_g_prime
from sesmax_core.guards import guard_finite, guard_positive
from sesmax_core.validation import (
    reject_derived_kwargs,
    require_non_negative,
    require_positive,
)
from sesmax_ocean._src.swm._shared import (
    BCS,
    UPWIND,
    WALLS,
    WRAPS,
    FreeSlip,
    fill_box,
    velocity_laplacian,
    wrap,
)


class MultilayerSW2DState(State):
    """State for the multilayer 2D nonlinear shallow water model.

    Args:
        h: Total layer thickness per layer, shape ``(nl, Ny, Nx)``.
        u: x-velocity on U-points per layer, shape ``(nl, Ny, Nx)``.
        v: y-velocity on V-points per layer, shape ``(nl, Ny, Nx)``.
    """

    h: Float[Array, "nl Ny Nx"] | CField
    u: Float[Array, "nl Ny Nx"] | CField
    v: Float[Array, "nl Ny Nx"] | CField


class MultilayerSW2DParams(Params):
    """Differentiable parameters for the multilayer 2D shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind stress amplitude (m/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class MultilayerSW2DPhysConsts(PhysConsts):
    """Frozen physical constants for the multilayer 2D shallow water model.

    Args:
        gravity: Gravitational acceleration g.
        f0: Reference Coriolis parameter.
        beta: Meridional gradient of f (beta-plane).
        n_layers: Number of vertical layers.
    """

    gravity: float = eqx.field(static=True, default=9.81)
    f0: float = eqx.field(static=True, default=1e-4)
    beta: float = eqx.field(static=True, default=0.0)
    n_layers: int = eqx.field(static=True, default=2)


class MultilayerSW2DDiagnostics(Diagnostics):
    """Diagnostics for the multilayer 2D shallow water model.

    Args:
        energy: Domain-integrated total energy per layer.
        total_energy: Domain-integrated total energy (scalar).
        enstrophy: Domain-integrated potential enstrophy per layer.
        total_enstrophy: Total potential enstrophy (scalar).
        mass: Domain-integrated layer mass/volume per layer,
            ``M_k = dx*dy * sum h_k``. Conserved to machine precision by
            the flux-form mass equation in a closed/periodic basin.
        casimir_q3: PV Casimir per layer, ``dx*dy * sum q^3 * h`` — a
            higher-moment material invariant, more sensitive than enstrophy.
        potential_vorticity: PV field q = (zeta+f)/h at X-points per layer.
        relative_vorticity: Relative vorticity zeta per layer.
        kinetic_energy_field: KE at T-points per layer.
    """

    energy: Array
    total_energy: Array
    enstrophy: Array
    total_enstrophy: Array
    mass: Array
    casimir_q3: Array
    potential_vorticity: Float[Array, "nl Ny Nx"]
    relative_vorticity: Float[Array, "nl Ny Nx"]
    kinetic_energy_field: Float[Array, "nl Ny Nx"]

    def invariants(self) -> dict[str, Array]:
        """Mass (tight), total energy, potential enstrophy, PV Casimir.

        Mass is conserved to machine precision (flux-form continuity); the
        energy/enstrophy/Casimir entries drift under the FV upwind PV
        advection's implicit dissipation and are drift signals, not zero
        targets.
        """
        return {
            "mass": jnp.sum(self.mass),
            "total_energy": self.total_energy,
            "potential_enstrophy": self.total_enstrophy,
            "casimir_q3": jnp.sum(self.casimir_q3),
        }


class MultilayerShallowWater2D(Model):
    r"""Multilayer 2D nonlinear shallow water model (vector-invariant form).

    Solves the rotating shallow water equations per layer k::

        dh_k/dt = -div(h_k * u_k)
        du_k/dt = +q_k * (h_k v_k)_bar - dP_k/dx + forcing
        dv_k/dt = -q_k * (h_k u_k)_bar - dP_k/dy + forcing

    where q_k = (zeta_k + f) / h_k is potential vorticity and
    P_k = KE_k + p_k is the Bernoulli potential with hydrostatic
    pressure coupling between layers:
    p_k = sum_{j=0}^{k} g'_j * h_j (cumulative).

    Wind forcing is applied to the top layer only; bottom drag
    to the bottom layer only.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: 2D Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        strat: Stratification profile (layer depths and reduced gravities).
        modal: Precomputed modal transform.
        f_field: Precomputed Coriolis field f(y) at T-points.
        f_field_ml: Coriolis field broadcast to ``(nl, Ny, Nx)``.
        free_slip: The potential vorticity on the walls and coasts.
        wind_stress_x: Precomputed x-wind stress pattern (normalised).
        wind_stress_y: Precomputed y-wind stress pattern (normalised).
        bc_type: Boundary condition type.
        scheme: Scheme of the face values in the mass equation.
    """

    params: MultilayerSW2DParams
    consts: MultilayerSW2DPhysConsts = eqx.field(static=True)
    grid: CartesianCGrid2D
    ops: GridOps
    mask: Mask2D | None
    strat: StratificationProfile
    modal: ModalTransform
    f_field: Float[Array, "Ny Nx"]
    free_slip: FreeSlip
    f_field_ml: Float[Array, "nl Ny Nx"]
    wind_stress_x: Float[Array, "Ny Nx"]
    wind_stress_y: Float[Array, "Ny Nx"]
    bc_type: str = eqx.field(static=True, default="periodic")
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def _wrap(self, field: Any, at: str) -> Any:
        """``field`` with its ghosts filled on the periodic faces."""
        return wrap(field, self.grid, WRAPS[self.bc_type], at)

    def _potential_vorticity(self, u: Any, v: Any, h: Any) -> Any:
        """``(zeta + f) / h`` at X-points, with ``f / h`` on walls and coasts."""
        q = self.ops.sw_potential_vorticity(u, v, h, placed(self.f_field_ml, h, "T"))
        q = each(lambda array: self.free_slip(array, values(h)), q)
        return self._wrap(q, "X")

    def _laplacian(self, velocity: Any, at: str) -> Any:
        """The Laplacian of a velocity component, with free slip along coasts."""
        return velocity_laplacian(
            velocity,
            self.grid,
            self.mask,
            at,
            thickness_bc=BCS[self.bc_type]["h"],
            wraps=WRAPS[self.bc_type],
        )

    def _mass_tendency(self, h: Any, u: Any, v: Any) -> Any:
        """``-div(h u)``, with the boundary condition of ``h`` at the edge faces."""
        return self.ops.advect(h, u, v, scheme=self.scheme, bc=BCS[self.bc_type]["h"])

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> MultilayerSW2DState:
        """Compute tendencies for the multilayer shallow water equations."""
        h, u, v = state.h, state.u, state.v
        # FAIL-HARD (in-JIT): layer thickness must stay strictly positive.
        # The PV q_k = (zeta_k + f) / h_k below is singular as h_k -> 0+, so a
        # non-positive thickness makes the rest of the tendency meaningless;
        # halt at this step rather than integrate a blow-up to the chunk end.
        h = each(partial(guard_positive, where="layer thickness h_k"), h)
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # --- 1. Mass equation: dh_k/dt = -div(h_k * u_k, h_k * v_k) ---
        dh_dt = self._mass_tendency(h, u, v)

        # --- 2. Potential vorticity: q_k = (zeta_k + f) / h_k at X-points ---
        q = self._potential_vorticity(u, v, h)

        # --- 3. Mass fluxes, then the cross-velocity PV fluxes ---
        uh = each(operator.mul, self.ops.interpolate(h, src="T", dst="U"), u)
        vh = each(operator.mul, self.ops.interpolate(h, src="T", dst="V"), v)
        # The next averages read the ghosts of these, which the operators leave
        # at zero.
        uh, vh = self._wrap(uh, "U"), self._wrap(vh, "V")
        q_on_U = self.ops.interpolate(q, src="X", dst="U")
        q_on_V = self.ops.interpolate(q, src="X", dst="V")
        vh_on_U = self.ops.interpolate(vh, src="V", dst="U")
        uh_on_V = self.ops.interpolate(uh, src="U", dst="V")

        # --- 4. Bernoulli potential: P_k = KE_k + p_k ---
        # Hydrostatic pressure coupling: p_k = sum_{j=0}^{k} g'_j * h_j
        g_prime = self.strat.g_prime[:, None, None]

        def bernoulli(ke: Array, thickness: Array) -> Array:
            return ke + jnp.cumsum(g_prime * thickness, axis=0)

        P = each(bernoulli, kinetic_energy(u, v, self.grid), h)

        # --- 5. Momentum: PV flux and pressure, wind on the top layer, diffusion,
        #        bottom drag on the bottom layer ---
        def x_momentum(
            pv: Array, flux: Array, dP: Array, lap: Array, velocity: Array
        ) -> Array:
            tendency = pv * flux - dP
            tendency = tendency.at[0].add(tau0 * self.wind_stress_x / self.strat.H[0])
            tendency = tendency + nu * lap
            return tendency.at[-1].add(-kappa * velocity[-1])

        def y_momentum(
            pv: Array, flux: Array, dP: Array, lap: Array, velocity: Array
        ) -> Array:
            tendency = -pv * flux - dP
            tendency = tendency.at[0].add(tau0 * self.wind_stress_y / self.strat.H[0])
            tendency = tendency + nu * lap
            return tendency.at[-1].add(-kappa * velocity[-1])

        P = self._wrap(P, "T")
        dP_dx = self.ops.difference(P, axis="x", src="T", dst="U")
        dP_dy = self.ops.difference(P, axis="y", src="T", dst="V")
        du_dt = each(x_momentum, q_on_U, vh_on_U, dP_dx, self._laplacian(u, "U"), u)
        dv_dt = each(y_momentum, q_on_V, uh_on_V, dP_dy, self._laplacian(v, "V"), v)

        # FAIL-HARD (in-JIT): halt at the first step that produces a
        # non-finite tendency, earlier than a host-side check, which only
        # runs once the chunk has fully integrated.
        dh_dt = each(partial(guard_finite, where="RHS dh/dt"), dh_dt)
        du_dt = each(partial(guard_finite, where="RHS du/dt"), du_dt)
        dv_dt = each(partial(guard_finite, where="RHS dv/dt"), dv_dt)

        return MultilayerSW2DState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> MultilayerSW2DState:
        """Apply boundary conditions to all state fields per layer.

        For periodic: ghost cells wrap around.
        For wall (free-slip, no-normal-flow):
          - u = 0 at x-boundaries per layer
          - v = 0 at y-boundaries per layer
          - Tangential velocity: zero-gradient (free-slip)
          - h: zero-gradient at all walls
        """
        return fill_box(state, self.grid, self.bc_type)

    def diagnose(self, state: PyTree) -> MultilayerSW2DDiagnostics:
        """Compute energy, enstrophy, PV, vorticity, and KE per layer."""
        h, u, v = values(state.h), values(state.u), values(state.v)
        grid = self.grid
        g_prime = self.strat.g_prime[:, None, None]  # [nl, 1, 1]
        ke = kinetic_energy(u, v, grid)  # T
        zeta = self.ops.curl(u, v)  # X
        q = self.ops.sw_potential_vorticity(u, v, h, self.f_field_ml)  # X
        h_on_X = self.ops.interpolate(h, src="T", dst="X")
        # The sums keep the layer axis, so every quantity below is per layer.
        # Energy per layer: KE + PE
        energy_per_layer = area_sum(ke + 0.5 * g_prime * h**2, grid)
        # Potential enstrophy per layer: 0.5 * q^2 * h
        enstrophy_per_layer = area_sum(0.5 * q**2 * h_on_X, grid)
        # Mass/volume per layer (flux-form invariant) and the q^3 Casimir.
        mass_per_layer = area_sum(h, grid)
        casimir_q3_per_layer = area_sum(q**3 * h_on_X, grid)

        return MultilayerSW2DDiagnostics(
            energy=energy_per_layer,
            total_energy=jnp.sum(energy_per_layer),
            enstrophy=enstrophy_per_layer,
            total_enstrophy=jnp.sum(enstrophy_per_layer),
            mass=mass_per_layer,
            casimir_q3=casimir_q3_per_layer,
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
        burger: Sequence[float],
        thickness_ratio: Sequence[float],
        beta_hat: float = 0.0,
        ekman: float = 0.0,
        ekman_lateral: float = 0.0,
        wind_hat: float = 0.0,
        aspect: float = 1.0,
        **create_kw: Any,
    ) -> tuple[MultilayerShallowWater2D, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **inertial** (:meth:`sesmax_core.Scales.inertial`), with
        ``L = f0 = 1`` and the depth scale set by the first layer, so
        ``T = 1/f0 = 1`` and ``U = Ro``.

        Stratification comes from one Burger number **per interface**,
        :math:`Bu_k = g'_k H_k/(f_0 L)^2`, inverted to
        ``g_prime[k] = Bu_k / thickness_ratio[k]``. That is the
        interface convention, not the per-mode one: the deformation
        radii of the vertical modes are a combination of the interface
        values, and are available on the built model as
        ``model.modal.rossby_radii`` once the eigenproblem is solved.

        Args:
            nx: Interior cells in x.
            ny: Interior cells in y.
            rossby: Rossby number; sets the velocity scale.
            burger: Per-interface Burger numbers, top to bottom.
            thickness_ratio: Layer thicknesses relative to the depth
                scale, ``H_k / H_1``. Same length as ``burger``.
            beta_hat: Dimensionless planetary vorticity gradient.
            ekman: Linear bottom-drag Ekman number.
            ekman_lateral: Lateral-viscosity Ekman number.
            wind_hat: Dimensionless wind acceleration.
            aspect: ``Ly / Lx``; the domain is ``Lx = 1``.
            **create_kw: Forwarded to :meth:`create`.

        Returns:
            ``(model, scales)`` with ``scales.kind == "inertial"``.

        Raises:
            ValueError: If an input is out of range or the sequences
                disagree in length.
        """
        context = "MultilayerShallowWater2D.from_nondimensional"
        reject_derived_kwargs(
            context,
            (
                "Lx",
                "Ly",
                "f0",
                "beta",
                "n_layers",
                "H",
                "g_prime",
                "stratification",
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
        g_prime = burger_to_g_prime(
            context, burger, thickness_ratio, f0=1.0, length=1.0
        )
        thickness = tuple(float(h) for h in thickness_ratio)

        model = MultilayerShallowWater2D.create(
            nx=nx,
            ny=ny,
            Lx=1.0,
            Ly=aspect,
            g=g_prime[0],
            f0=1.0,
            beta=beta_hat,
            n_layers=len(thickness),
            H=thickness,
            g_prime=g_prime,
            lateral_viscosity=ekman_lateral,
            bottom_drag=ekman,
            # The RHS applies the wind as tau0 * F / H[0], so the
            # amplitude carries the top-layer thickness; without it
            # the realised acceleration is wind_hat / H[0] whenever
            # thickness_ratio[0] is not 1. Same convention as the
            # layered QG factories.
            wind_amplitude=wind_hat * rossby * thickness[0],
            **create_kw,
        )
        scales = Scales.inertial(
            L=1.0, f0=1.0, H=thickness[0], rossby=rossby, g=g_prime[0]
        )
        return model, scales

    @staticmethod
    def create(
        nx: int = 64,
        ny: int = 64,
        Lx: float = 4e6,
        Ly: float = 4e6,
        g: float = 9.81,
        f0: float = 9.375e-5,
        beta: float = 1.754e-11,
        n_layers: int = 3,
        H: tuple[float, ...] = (400.0, 1100.0, 2600.0),
        g_prime: tuple[float, ...] = (9.81, 0.025, 0.0125),
        stratification: StratificationProfile | None = None,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "doublegyre",
        bc: str = "periodic",
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
    ) -> MultilayerShallowWater2D:
        """Convenience factory for the multilayer shallow water model.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x (m).
            Ly: Domain length in y (m).
            g: Gravitational acceleration (m/s^2) (only used
                if ``stratification`` is None to set g_prime[0]).
            f0: Reference Coriolis parameter (1/s).
            beta: Meridional gradient of f (1/(m*s)).
            n_layers: Number of layers (ignored if ``stratification`` given).
            H: Layer thicknesses (m), top to bottom
                (ignored if ``stratification`` given).
            g_prime: Reduced gravities (m/s^2) at each interface;
                g_prime[0] = g (full gravity)
                (ignored if ``stratification`` given).
            stratification: Pre-built ``StratificationProfile``, or None
                to build from ``H`` and ``g_prime``.
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
            A ``MultilayerShallowWater2D`` model instance.

        Raises:
            ValueError: If ``n_layers``, ``H``, and ``g_prime`` have
                inconsistent lengths (when ``stratification`` is None).
        """
        grid = CartesianCGrid2D.from_interior(nx, ny, Lx, Ly)

        # Stratification
        if stratification is not None:
            strat = stratification
        else:
            if len(H) != n_layers or len(g_prime) != n_layers:
                msg = (
                    f"n_layers ({n_layers}), len(H) ({len(H)}), and "
                    f"len(g_prime) ({len(g_prime)}) must all be equal"
                )
                raise ValueError(msg)
            strat = StratificationProfile.from_layers(H=list(H), g_prime=list(g_prime))

        nl = strat.nl

        # Modal transform (for diagnostics / optional barotropic filter)
        modal = ModalTransform.from_stratification(strat, f0)

        params = MultilayerSW2DParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
            wind_amplitude=as_parameter(wind_amplitude),
        )
        consts = MultilayerSW2DPhysConsts(gravity=g, f0=f0, beta=beta, n_layers=nl)

        # Precompute Coriolis field
        y = jnp.arange(grid.Ny) * grid.spacing("y")
        y0 = Ly / 2.0
        Y = jnp.broadcast_to(y[:, None], (grid.Ny, grid.Nx))
        f_field = beta_plane(Y, f0=f0, beta=beta, y0=y0)
        f_field_ml = jnp.broadcast_to(f_field[None], (nl, grid.Ny, grid.Nx)).copy()

        # Wind stress pattern (normalised)
        if wind_profile == "single":
            wind_stress_x = -jnp.cos(jnp.pi * Y / Ly)
        else:
            wind_stress_x = -jnp.cos(2.0 * jnp.pi * Y / Ly)
        wind_stress_y = jnp.zeros_like(wind_stress_x)

        return MultilayerShallowWater2D(
            params=params,
            consts=consts,
            grid=grid,
            ops=GridOps(grid, mask),
            mask=mask,
            strat=strat,
            modal=modal,
            f_field=f_field,
            free_slip=FreeSlip.build(
                grid,
                mask,
                BCS[bc]["h"],
                f_field,
                walls=WALLS[bc],
            ),
            f_field_ml=f_field_ml,
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y,
            bc_type=bc,
            scheme=scheme,
        )
