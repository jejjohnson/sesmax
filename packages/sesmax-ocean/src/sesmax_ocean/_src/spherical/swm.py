"""Single-layer shallow water on a sphere (vector-invariant form)."""

from __future__ import annotations

import operator
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from geophysix.earth import SPHERE
from gridax import (
    BC,
    CField,
    Dirichlet,
    GridOps,
    Mask2D,
    Neumann,
    Periodic,
    SphericalCGrid2D,
    apply_mask,
    area_sum,
    fill_ghosts,
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
from sesmax_core.gfd import resolve_burger_spherical
from sesmax_core.validation import require_non_negative, require_positive
from sesmax_ocean._src.resolution import check_equatorial_deformation_radius
from sesmax_ocean._src.spherical._geometry import (
    latitudes,
    require_global_longitude,
    require_open_poles,
)
from sesmax_ocean._src.swm._shared import POINTS, UPWIND


# Periodic in longitude, solid walls with free slip in latitude.
_BAND = {
    "h": BC(x=Periodic(), y=Neumann()),
    "u": BC(x=Periodic(), y=Neumann()),
    "v": BC(x=Periodic(), y=Dirichlet(0.0)),
}


class SphericalSWMState(State):
    """State for the spherical shallow water model.

    Args:
        h: Layer thickness at T-points, shape ``(Ny, Nx)``.
        u: Zonal velocity at U-points, shape ``(Ny, Nx)``.
        v: Meridional velocity at V-points, shape ``(Ny, Nx)``.
    """

    h: Float[Array, "Ny Nx"] | CField
    u: Float[Array, "Ny Nx"] | CField
    v: Float[Array, "Ny Nx"] | CField


class SphericalSWMParams(Params):
    """Differentiable parameters for the spherical shallow water model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind stress amplitude (m/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class SphericalSWMPhysConsts(PhysConsts):
    """Frozen physical constants for the spherical shallow water model.

    Note the absence of ``f0`` and ``beta``. On a sphere the Coriolis
    parameter is a *field*, ``f(phi) = 2 Omega sin(phi)``, and its
    meridional gradient follows from the geometry rather than being a
    free constant — which is the whole point of leaving the beta plane.

    Args:
        gravity: Gravitational acceleration g (m/s^2).
        omega: Planet angular velocity (rad/s).
        radius: Planet radius (m).
        H0: Mean layer depth (m).
    """

    gravity: float = eqx.field(static=True, default=9.81)
    omega: float = eqx.field(static=True, default=SPHERE.rotation_rate)
    radius: float = eqx.field(static=True, default=SPHERE.semi_major)
    H0: float = eqx.field(static=True, default=1000.0)


class SphericalSWMDiagnostics(Diagnostics):
    """Diagnostics for the spherical shallow water model.

    Integrals use the spherical cell area ``R^2 cos(lat) dlon dlat``
    rather than a uniform ``dx dy``, so a conservation statement means
    the same thing at every latitude.

    Args:
        energy: Area-integrated total energy,
            ``h * KE + g h^2 / 2``.
        enstrophy: Area-integrated potential enstrophy.
        mass: Area-integrated mass.
        casimir_q3: PV Casimir, the ``q^3`` moment.
        potential_vorticity: PV at X-points.
        relative_vorticity: Relative vorticity at X-points.
        kinetic_energy_field: Kinetic energy at T-points.
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


class SphericalSWM(Model):
    r"""Shallow water on a sphere, vector-invariant form.

    Solves the rotating shallow-water equations on a spherical Arakawa
    C-grid::

        dh/dt = -div_sphere(h u)
        du/dt = +q (h v)_bar - (1/(R cos(phi))) dP/dlon + nu lap(u) - kappa u
        dv/dt = -q (h u)_bar - (1/R) dP/dlat            + nu lap(v) - kappa v

    with ``q = (zeta + f)/h`` the potential vorticity and
    ``P = KE + g h`` the Bernoulli potential. Every horizontal
    derivative carries the spherical metric: the ``1/(R cos(phi))``
    factor in longitude and ``1/R`` in latitude, supplied by the
    grid.

    Coriolis is the full ``f(phi) = 2 Omega sin(phi)``, not a beta-plane
    expansion about a reference latitude. The planetary vorticity
    gradient ``beta = 2 Omega cos(phi)/R`` is then implicit in the field
    and varies correctly from equator to pole.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: Spherical Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional land/ocean mask (``None`` = all-ocean).
        f_field: Precomputed Coriolis field at X-points.
        wind_stress_x: Normalised zonal wind-stress pattern.
        wind_stress_y: Normalised meridional wind-stress pattern.
        scheme: Scheme of the face values in the mass equation.
    """

    params: SphericalSWMParams
    consts: SphericalSWMPhysConsts = eqx.field(static=True)
    grid: SphericalCGrid2D
    ops: GridOps
    mask: Mask2D | None
    f_field: Float[Array, "Ny Nx"]
    wind_stress_x: Float[Array, "Ny Nx"]
    wind_stress_y: Float[Array, "Ny Nx"]
    scheme: Scheme = eqx.field(static=True, default=UPWIND)

    def _masked(self, field: Float[Array, "Ny Nx"], at: str) -> Float[Array, "Ny Nx"]:
        """``field`` zeroed on land at the given C-grid stagger."""
        return field if self.mask is None else apply_mask(field, self.mask, at=at)

    def _mass_tendency(self, h: Any, u: Any, v: Any) -> Any:
        """``-div(h u)``, with the boundary condition of ``h`` at the edge faces."""
        return self.ops.advect(h, u, v, scheme=self.scheme, bc=_BAND["h"])

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> SphericalSWMState:
        """Compute tendencies for the spherical shallow water equations."""
        h, u, v = state.h, state.u, state.v
        g = self.consts.gravity
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        # 1. Mass: dh/dt = -div_sphere(h u)
        dh_dt = self._mass_tendency(h, u, v)

        # 2. Potential vorticity q = (zeta + f)/h at X-points.
        f = placed(self.f_field, h, "T")
        q = self.ops.sw_potential_vorticity(u, v, h, f)

        # 3. Mass fluxes, then the cross-velocity PV fluxes (vector-invariant
        # Coriolis term).
        uh = each(operator.mul, self.ops.interpolate(h, src="T", dst="U"), u)
        vh = each(operator.mul, self.ops.interpolate(h, src="T", dst="V"), v)
        q_on_U = self.ops.interpolate(q, src="X", dst="U")
        q_on_V = self.ops.interpolate(q, src="X", dst="V")
        vh_on_U = self.ops.interpolate(vh, src="V", dst="U")
        uh_on_V = self.ops.interpolate(uh, src="U", dst="V")

        # 4. Bernoulli potential P = KE + g h.
        ke = kinetic_energy(u, v, self.grid)
        P = each(lambda k, thickness: k + g * thickness, ke, h)

        # 5. Wind forcing. Masked at its stagger: every other operator
        # here is, so an unmasked source would be the one thing giving
        # dry U/V cells a velocity — and nothing projects it back out,
        # so it would persist into the saved states.
        wind_u = tau0 * self._masked(self.wind_stress_x, "U")
        wind_v = tau0 * self._masked(self.wind_stress_y, "V")

        # 6. Lateral diffusion, by the operator for T-points on the array of a
        # velocity.
        diffusion_u = self.ops.diffuse(values(u), kappa=nu)
        diffusion_v = self.ops.diffuse(values(v), kappa=nu)

        # 7. Momentum: PV flux and pressure, then wind, diffusion and linear
        # bottom drag. The pressure gradients carry the spherical metric.
        def zonal(pv: Array, flux: Array, dP: Array, velocity: Array) -> Array:
            return pv * flux - dP + wind_u + diffusion_u - kappa * velocity

        def meridional(pv: Array, flux: Array, dP: Array, velocity: Array) -> Array:
            tendency = -pv * flux - dP + wind_v + diffusion_v - kappa * velocity
            # Hold the constrained rows fixed. ``apply_boundary_conditions``
            # zeroes the wall-normal v before each evaluation, but the
            # integrator projects only the RHS *input* — the returned and
            # saved states are not projected — so a nonzero dv_dt on those
            # rows would accumulate there and the snapshots would show
            # through-flow at a wall the model says is closed.
            return tendency.at[0, :].set(0.0).at[-2, :].set(0.0).at[-1, :].set(0.0)

        dP_dx = self.ops.difference(P, axis="x", src="T", dst="U")
        dP_dy = self.ops.difference(P, axis="y", src="T", dst="V")
        du_dt = each(zonal, q_on_U, vh_on_U, dP_dx, u)
        dv_dt = each(meridional, q_on_V, uh_on_V, dP_dy, v)

        return SphericalSWMState(h=dh_dt, u=du_dt, v=dv_dt)

    def apply_boundary_conditions(self, state: PyTree) -> SphericalSWMState:
        """Periodic in longitude, solid walls in latitude.

        Longitude wraps because a lat-lon grid spanning the full globe
        is genuinely periodic there. Latitude does not: the poles are
        coordinate singularities, not a boundary the flow crosses, so
        the meridional velocity is set to zero at the northern and
        southern edges and thickness is given a zero-gradient condition.
        """
        return fill_ghosts(
            state,
            self.grid,
            SphericalSWMState(**_BAND),
            stagger=SphericalSWMState(**POINTS),
        )

    def diagnose(self, state: PyTree) -> SphericalSWMDiagnostics:
        """Area-weighted invariants plus PV, vorticity and KE fields."""
        g = self.consts.gravity
        h, u, v = values(state.h), values(state.u), values(state.v)
        grid = self.grid

        ke = kinetic_energy(u, v, grid)
        zeta = self.ops.curl(u, v)
        q = self.ops.sw_potential_vorticity(u, v, h, self.f_field)

        # The PV invariants live at X-points, whose dual cells sit half
        # a latitude step north of the T-cells: their area goes as
        # cos(lat_T + dlat/2), not cos(lat_T). Reusing the T weights
        # biases enstrophy and the cubic Casimir, and increasingly so
        # towards the poles.
        interior = (slice(1, -1), slice(1, -1))
        area_X = grid.area(at="X")[interior]
        h_on_X = self.ops.interpolate(h, src="T", dst="X")[interior]
        q_on_X = q[interior]

        # ``kinetic_energy`` returns the *specific* KE, 0.5(u^2+v^2),
        # so it has to be weighted by the layer thickness before it can
        # be added to the potential term — otherwise the two have
        # different units and the kinetic part is undercounted by a
        # factor of h.
        energy = area_sum(h * ke + 0.5 * g * h**2, grid)
        enstrophy = 0.5 * jnp.sum(area_X * q_on_X**2 * h_on_X)
        mass = area_sum(h, grid)
        casimir_q3 = jnp.sum(area_X * q_on_X**3 * h_on_X)

        return SphericalSWMDiagnostics(
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
        nx: int = 128,
        ny: int = 64,
        rossby: float,
        burger: float | None = None,
        froude: float | None = None,
        lamb: float | None = None,
        ekman: float = 0.0,
        ekman_lateral: float = 0.0,
        wind_hat: float = 0.0,
        lat_range: tuple[float, float] = (-80.0, 80.0),
        lon_range: tuple[float, float] = (0.0, 360.0),
        check_resolution: bool = True,
        **create_kw: Any,
    ) -> tuple[SphericalSWM, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **planetary** (:meth:`sesmax_core.Scales.planetary`), with
        ``a = Omega = H = 1``. The planet radius is the length scale and
        the rotation period the time scale, so ``T = 1/Omega = 1`` and
        ``f0 = 2 Omega = 2``; the velocity scale follows from the Rossby
        number as ``U = 2 Omega a Ro = 2 Ro``.

        The factor of two is not cosmetic. On a sphere the Coriolis
        parameter is ``f(phi) = 2 Omega sin(phi)``, so the quantity that
        plays the role of a Cartesian ``f0`` is ``2 Omega``, and the
        conventional spherical Rossby number is ``U/(2 Omega a)``.
        Writing it that way keeps :attr:`Scales.rossby` the same
        ``U/(f0 L)`` it is for every other family, at the cost of the
        twos that appear in the table below.

        ==================  ==================================  ====================
        Input               Definition                          ``create()`` kwarg
        ==================  ==================================  ====================
        ``rossby``          :math:`Ro = U/(2\Omega a)`           sets ``U``
        ``burger``          :math:`Bu = gH/(2\Omega a)^2`        ``g = 4Bu``
        ``froude``          :math:`Fr = U/\sqrt{gH}`             ``g`` (via ``Bu``)
        ``lamb``            :math:`\varepsilon = 1/Bu`           ``g = 4/\varepsilon``
        ``ekman``           :math:`\kappa/(2\Omega)`             ``bottom_drag``
        ``ekman_lateral``   :math:`\nu/(2\Omega a^2)`            ``lateral_viscosity``
        ``wind_hat``        :math:`\tau_0/(2\Omega U)`           ``wind_amplitude``
        ==================  ==================================  ====================

        The latitude range stays in degrees: it is a geometric choice,
        not a scale, and a nondimensional sphere is still a sphere.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            rossby: Rossby number; sets the velocity scale.
            burger: Burger number. Give exactly one of this,
                ``froude`` or ``lamb``.
            froude: Froude number. Equivalent through
                ``Bu = (Ro/Fr)**2``.
            lamb: Lamb parameter ``eps = 4 Omega**2 a**2/(gH)``, the
                inverse Burger number and the usual spelling in the
                spherical literature.
            ekman: Linear bottom-drag Ekman number.
            ekman_lateral: Lateral-viscosity Ekman number.
            wind_hat: Dimensionless wind acceleration.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            check_resolution: Run the equatorial-deformation-radius
                guard and raise if the grid cannot span it. Set
                ``False`` only to build a deliberately coarse model.
            **create_kw: Forwarded to :meth:`create` (``wind_profile``,
                ``scheme``, ``mask``).

        Returns:
            ``(model, scales)`` with ``scales.kind == "planetary"``.

        Raises:
            ValueError: If an input is out of range, or if not exactly
                one of ``burger``, ``froude`` and ``lamb`` is given.
            AssertionFailedError: If ``check_resolution`` is set and the
                equatorial deformation radius is unresolved.

        Example:
            >>> model, scales = SphericalSWM.from_nondimensional(
            ...     nx=128,
            ...     ny=64,
            ...     rossby=0.05,
            ...     lamb=10.0,
            ... )
            >>> scales.kind
            'planetary'
        """
        context = "SphericalSWM.from_nondimensional"
        require_positive(context, rossby=rossby)
        require_non_negative(
            context,
            ekman=ekman,
            ekman_lateral=ekman_lateral,
            wind_hat=wind_hat,
        )
        bu = resolve_burger_spherical(context, burger, froude, lamb, rossby)

        # Unit scales: a = Omega = H0 = 1, so f0 = 2 and U = 2 Ro. Rate
        # coefficients carry one factor of f0; the wind is an
        # acceleration and so carries f0 * U.
        f0 = 2.0
        velocity = f0 * rossby
        model = SphericalSWM.create(
            nx=nx,
            ny=ny,
            lon_range=lon_range,
            lat_range=lat_range,
            radius=1.0,
            omega=1.0,
            g=f0**2 * bu,
            H0=1.0,
            lateral_viscosity=f0 * ekman_lateral,
            bottom_drag=f0 * ekman,
            wind_amplitude=wind_hat * f0 * velocity,
            **create_kw,
        )
        scales = Scales.planetary(a=1.0, Omega=1.0, H=1.0, rossby=rossby, g=f0**2 * bu)

        if check_resolution:
            check_equatorial_deformation_radius(None, model)

        return model, scales

    @staticmethod
    def create(
        nx: int = 128,
        ny: int = 64,
        lon_range: tuple[float, float] = (0.0, 360.0),
        lat_range: tuple[float, float] = (-80.0, 80.0),
        radius: float = SPHERE.semi_major,
        omega: float = SPHERE.rotation_rate,
        g: float = 9.81,
        H0: float = 1000.0,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "zonal",
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
        wind_stress_x: Float[Array, "Ny Nx"] | None = None,
        wind_stress_y: Float[Array, "Ny Nx"] | None = None,
    ) -> SphericalSWM:
        """Convenience factory.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            radius: Planet radius (m).
            omega: Planet angular velocity (rad/s).
            g: Gravitational acceleration (m/s^2).
            H0: Mean layer depth (m).
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind stress amplitude (m/s^2).
            wind_profile: ``"zonal"`` gives a ``-cos(2 lat)`` zonal
                stress (trades and westerlies); ``"none"`` gives no
                pattern.
            scheme: Scheme of the face values in the mass equation.
            mask: Optional land/ocean mask.
            wind_stress_x: Precomputed zonal stress pattern on the
                model grid, overriding ``wind_profile``. For a scenario
                that supplies its own forcing field rather than one of
                the analytic profiles.
            wind_stress_y: Precomputed meridional stress pattern.

        Returns:
            A ``SphericalSWM`` instance.
        """
        context = "SphericalSWM.create"
        require_global_longitude(context, lon_range)
        require_open_poles(context, lat_range)
        grid = SphericalCGrid2D.from_interior(nx, ny, lon_range, lat_range, R=radius)

        params = SphericalSWMParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
            wind_amplitude=as_parameter(wind_amplitude),
        )
        consts = SphericalSWMPhysConsts(gravity=g, omega=omega, radius=radius, H0=H0)

        # Coriolis on X-points, where the PV lives: f = 2 Omega sin(lat).
        lat_T = latitudes(grid)
        f_field = 2.0 * omega * jnp.sin(lat_T + 0.5 * grid.dlat)

        if wind_stress_x is None:
            if wind_profile == "none":
                wind_stress_x = jnp.zeros_like(lat_T)
            else:
                # Trades / westerlies: eastward at mid-latitudes,
                # westward at the equator and poles.
                wind_stress_x = -jnp.cos(2.0 * lat_T)
        else:
            wind_stress_x = jnp.asarray(wind_stress_x)
        wind_stress_y = (
            jnp.zeros_like(lat_T)
            if wind_stress_y is None
            else jnp.asarray(wind_stress_y)
        )

        return SphericalSWM(
            params=params,
            consts=consts,
            grid=grid,
            ops=GridOps(grid, mask),
            mask=mask,
            f_field=f_field,
            wind_stress_x=wind_stress_x,
            wind_stress_y=wind_stress_y,
            scheme=scheme,
        )
