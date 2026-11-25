"""Barotropic quasi-geostrophic flow on a sphere."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import gaussx
import jax
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
    area_weights,
    embed,
    fill_ghosts,
    interior,
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
from sesmax_core.fields import each, values
from sesmax_core.validation import require_non_negative, require_positive
from sesmax_ocean._src.spherical._geometry import (
    latitudes,
    require_global_longitude,
    require_open_poles,
)
from sesmax_ocean._src.swm._shared import UPWIND


# Periodic in longitude; zero on the wall of a streamfunction, which lies between
# the last cell centre and its ghost.
_STREAMFUNCTION = BC(x=Periodic(), y=Dirichlet(0.0))
_WRAP = BC(x=Periodic())
# No flux of vorticity through the walls.
_FLAT = BC(x=Periodic(), y=Neumann())


class SphericalQGState(State):
    """State for the spherical barotropic QG model.

    Args:
        q: Relative vorticity at T-points, shape ``(Ny, Nx)``. The
            streamfunction is recovered on the fly by inverting the
            spherical Laplacian, so ``psi`` is not carried.
    """

    q: Float[Array, "Ny Nx"] | CField


class SphericalQGParams(Params):
    """Differentiable parameters for the spherical QG model.

    Args:
        lateral_viscosity: Harmonic viscosity coefficient nu (m^2/s).
        bottom_drag: Linear bottom drag coefficient kappa (1/s).
        wind_amplitude: Wind stress curl amplitude (1/s^2).
    """

    lateral_viscosity: Array
    bottom_drag: Array
    wind_amplitude: Array


class SphericalQGPhysConsts(PhysConsts):
    """Frozen physical constants for the spherical QG model.

    Args:
        omega: Planet angular velocity (rad/s).
        radius: Planet radius (m).
    """

    omega: float = eqx.field(static=True, default=SPHERE.rotation_rate)
    radius: float = eqx.field(static=True, default=SPHERE.semi_major)


class SphericalQGDiagnostics(Diagnostics):
    """Diagnostics for the spherical QG model.

    Args:
        kinetic_energy: Area-integrated kinetic energy.
        enstrophy: Area-integrated enstrophy, the ``q^2`` moment.
        circulation: Area-integrated relative vorticity.
        streamfunction: Recovered streamfunction at T-points.
        relative_vorticity: The state's vorticity field.
    """

    kinetic_energy: Array
    enstrophy: Array
    circulation: Array
    streamfunction: Float[Array, "Ny Nx"]
    relative_vorticity: Float[Array, "Ny Nx"]

    def invariants(self) -> dict[str, Array]:
        """Kinetic energy, enstrophy and circulation."""
        return {
            "kinetic_energy": self.kinetic_energy,
            "enstrophy": self.enstrophy,
            "circulation": self.circulation,
        }


class SphericalQG(Model):
    r"""Barotropic quasi-geostrophic flow on a sphere.

    Advects absolute vorticity ``q + f`` by the non-divergent flow
    recovered from the streamfunction::

        dq/dt = -adv_sphere(q + f, u, v) + nu lap(q) - kappa q + tau curl

    with ``(u, v) = (-1/R dpsi/dlat, 1/(R cos(phi)) dpsi/dlon)`` and
    ``psi`` from ``lap_sphere(psi) = q``.

    The planetary vorticity gradient is not a constant here. Advecting
    the *absolute* vorticity ``q + f(phi)`` with ``f = 2 Omega sin(phi)``
    reproduces ``beta = 2 Omega cos(phi)/R`` implicitly, so it varies
    from its maximum at the equator to zero at the poles rather than
    being frozen at a reference latitude.

    Inversion is iterative. The spherical Laplacian is not diagonal in
    any transform a lat-lon grid affords — the ``cos(phi)`` metric
    couples latitudes — so the DST route the Cartesian model uses does
    not apply, and the elliptic problem is solved by conjugate
    gradients against the masked spherical Laplacian.

    Args:
        params: Differentiable parameters.
        consts: Frozen physical constants.
        grid: Spherical Arakawa C-grid.
        ops: gridax's operations on the grid and the mask.
        mask: Optional land/ocean mask (``None`` = all-ocean).
        f_field: Precomputed Coriolis field at T-points.
        wind_forcing: Normalised wind-stress-curl pattern.
        scheme: Scheme of the advective face values.
        cg_tol: Convergence tolerance for the PV inversion, used for
            both the relative and the absolute criterion. The default
            is chosen for float32: a tighter absolute tolerance never
            trips, and CG runs to its step cap.
        cg_max_steps: Iteration cap for the PV inversion.
    """

    params: SphericalQGParams
    consts: SphericalQGPhysConsts = eqx.field(static=True)
    grid: SphericalCGrid2D
    ops: GridOps
    mask: Mask2D | None
    f_field: Float[Array, "Ny Nx"]
    wind_forcing: Float[Array, "Ny Nx"]
    scheme: Scheme = eqx.field(static=True, default=UPWIND)
    cg_tol: float = eqx.field(static=True, default=1e-6)
    cg_max_steps: int = eqx.field(static=True, default=500)

    def _masked(self, field: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """``field`` zeroed on land at T-points."""
        return field if self.mask is None else apply_mask(field, self.mask, at="T")

    def laplacian(self, psi: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """The spherical Laplacian of a field at T-points."""
        return self.ops.laplacian(psi, at="T")

    def _expand(self, unknowns: Float[Array, "Ny_i Nx_i"]) -> Float[Array, "Ny Nx"]:
        """Place interior unknowns into a full field with its ghost ring.

        Longitude wraps. The polar ghost rows are filled
        *antisymmetrically*, ``psi_ghost = -psi_adjacent``: this grid
        is cell-centred, so the wall lies on the face between the
        ghost centre and the first interior centre, and it is their
        average that the boundary condition constrains. Zeroing the
        ghost centre instead would put ``psi = 0`` half a cell outside
        the wall, shifting the stencil and leaving the recovered flow
        free to cross it. It removes the constant null space just the
        same, so CG is still happy.
        """
        full = embed(unknowns, self.grid)
        return fill_ghosts(full, self.grid, _STREAMFUNCTION, stagger="T")

    def invert_pv(self, q: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        """Recover the streamfunction from vorticity: ``lap(psi) = q``.

        Three things have to be arranged before conjugate gradients
        will touch this problem, and each was found by it refusing to:

        **The unknowns are the interior only,** with ``psi = 0`` on the
        polar ghost rows and longitude wrapping. A Laplacian over the
        whole array has zero rows on the ghost ring and a constant null
        space; the Dirichlet condition removes both, and it is the
        physically right one — no flow through the polar walls, with the
        gauge pinned.

        **The operator is area-weighted.** The spherical Laplacian is
        self-adjoint in the area-weighted inner product, not the
        Euclidean one CG assumes: measured directly, the bare operator
        is 15% asymmetric, while ``area * lap`` is symmetric to machine
        precision and negative definite. CG on the bare operator does
        not converge to anything.

        **The right-hand side is normalised.** With an Earth-radius
        grid the weighted residual sits around ``1e-13``, and CG's
        internal inner products then underflow float32. Solving the
        unit-norm system and rescaling afterwards is exact, since the
        operator is linear.

        Args:
            q: Relative vorticity at T-points.

        Returns:
            Streamfunction at T-points, zero on the polar boundary.
        """
        weights = interior(area_weights(self.grid), self.grid)

        def matvec(psi: Float[Array, "Ny_i Nx_i"]) -> Float[Array, "Ny_i Nx_i"]:
            return weights * interior(self.laplacian(self._expand(psi)), self.grid)

        rhs = weights * interior(q, self.grid)
        scale = jnp.linalg.norm(rhs)
        # A zero right-hand side has psi = 0; dividing by its norm would
        # be 0/0, and CG would be handed NaNs.
        safe_scale = jnp.where(scale == 0.0, 1.0, scale)
        operator = gaussx.as_linear_operator(
            matvec,
            in_structure=jax.ShapeDtypeStruct(rhs.shape, rhs.dtype),
            negative_definite=True,
        )
        solution, _ = gaussx.linear_solve_with_info(
            operator,
            rhs / safe_scale,
            solver=gaussx.CGSolver(
                rtol=self.cg_tol, atol=self.cg_tol, max_steps=self.cg_max_steps
            ),
        )
        return self._expand(solution * scale)

    def velocities(
        self, psi: Float[Array, "Ny Nx"]
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Non-divergent velocity from the streamfunction.

        ``u = -(1/R) dpsi/dlat`` at U-points and
        ``v = (1/(R cos(phi))) dpsi/dlon`` at V-points, from ``psi``
        averaged to the corners. It is zero on the walls, so no flow
        crosses them, and the discrete divergence vanishes.

        Args:
            psi: Streamfunction at T-points, with its ghosts filled.

        Returns:
            ``(u, v)`` on their staggered points.
        """
        u, v = self.ops.grad_perp(psi, at="T")
        return (
            fill_ghosts(u, self.grid, _WRAP, stagger="U"),
            fill_ghosts(v, self.grid, _WRAP, stagger="V"),
        )

    def _advection(
        self,
        q: Float[Array, "Ny Nx"],
        u: Float[Array, "Ny Nx"],
        v: Float[Array, "Ny Nx"],
    ) -> Float[Array, "Ny Nx"]:
        """``-div(q u)``, with no flux through the walls."""
        return self.ops.advect(q, u, v, scheme=self.scheme, bc=_FLAT)

    def _laplacian_of_vorticity(self, q: Any) -> Any:
        """The Laplacian of the vorticity, with no flux through the walls."""
        q = fill_ghosts(q, self.grid, _FLAT, stagger="T")
        return self.ops.laplacian(q, at="T")

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> SphericalQGState:
        """Compute the vorticity tendency."""
        q = values(state.q)
        nu = self.params.lateral_viscosity
        kappa = self.params.bottom_drag
        tau0 = self.params.wind_amplitude

        psi = self.invert_pv(q)
        u, v = self.velocities(psi)

        # Advect absolute vorticity: this is where beta comes from.
        advection = self._advection(q + self.f_field, u, v)

        # Masked at T-points, like every other operator here: an
        # unmasked source would inject vorticity over land, and
        # nothing projects it back out of the saved state.
        wind = tau0 * self._masked(self.wind_forcing)

        def tendency(lap: Array, vorticity: Array) -> Array:
            dq_dt = advection + wind + nu * lap - kappa * vorticity
            # Same reason as the shallow-water model: the polar rows carry
            # a Dirichlet condition that the integrator does not reapply to
            # the returned state, so the tendency there has to be zero for
            # the condition to survive a step.
            return dq_dt.at[0, :].set(0.0).at[-1, :].set(0.0)

        lap = self._laplacian_of_vorticity(state.q)
        return SphericalQGState(q=each(tendency, lap, state.q))

    def apply_boundary_conditions(self, state: PyTree) -> SphericalQGState:
        """Periodic in longitude, zero vorticity on the polar edges."""
        q = embed(interior(state.q, self.grid), self.grid)
        return SphericalQGState(q=fill_ghosts(q, self.grid, _WRAP, stagger="T"))

    def diagnose(self, state: PyTree) -> SphericalQGDiagnostics:
        """Area-weighted energy, enstrophy and circulation."""
        q = values(state.q)
        psi = self.invert_pv(q)
        u, v = self.velocities(psi)

        u_on_T = self.ops.interpolate(u, src="U", dst="T")
        v_on_T = self.ops.interpolate(v, src="V", dst="T")
        ke = 0.5 * area_sum(u_on_T**2 + v_on_T**2, self.grid)
        enstrophy = 0.5 * area_sum(q**2, self.grid)
        circulation = area_sum(q, self.grid)

        return SphericalQGDiagnostics(
            kinetic_energy=ke,
            enstrophy=enstrophy,
            circulation=circulation,
            streamfunction=psi,
            relative_vorticity=q,
        )

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 128,
        ny: int = 64,
        rossby: float,
        ekman: float = 0.0,
        ekman_lateral: float = 0.0,
        wind_hat: float = 0.0,
        lat_range: tuple[float, float] = (-80.0, 80.0),
        lon_range: tuple[float, float] = (0.0, 360.0),
        **create_kw: Any,
    ) -> tuple[SphericalQG, Scales]:
        r"""Build the model from dimensionless numbers instead of SI coefficients.

        Non-dimensional form
        --------------------
        Scale set: **planetary** (:meth:`sesmax_core.Scales.planetary`), with
        ``a = Omega = 1``, so ``T = 1/Omega = 1``, ``f0 = 2 Omega = 2``
        and ``U = 2 Omega a Ro = 2 Ro``. The vorticity the state
        carries is then ``O(U/a) = O(2 Ro)``.

        ==================  ==================================  ====================
        Input               Definition                          ``create()`` kwarg
        ==================  ==================================  ====================
        ``rossby``          :math:`Ro = U/(2\Omega a)`           sets ``U``
        ``ekman``           :math:`\kappa/(2\Omega)`             ``bottom_drag``
        ``ekman_lateral``   :math:`\nu/(2\Omega a^2)`            ``lateral_viscosity``
        ``wind_hat``        :math:`a\tau_0/(2\Omega U)`          ``wind_amplitude``
        ==================  ==================================  ====================

        There is no Burger number here and no ``g``: the barotropic QG
        model is rigid-lid, so it has no gravity wave and no
        deformation radius to resolve. The stratification knobs belong
        to :class:`SphericalSWM`.

        Neither is there a ``beta_hat``. On a sphere the planetary
        vorticity gradient is fixed by the geometry —
        ``beta = 2 Omega cos(phi)/a``, which is ``cos(phi)`` in these
        units — so it is not a free dimensionless number the way it is
        on a beta plane.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            rossby: Rossby number; sets the velocity scale.
            ekman: Linear bottom-drag Ekman number.
            ekman_lateral: Lateral-viscosity Ekman number.
            wind_hat: Dimensionless wind-stress-curl amplitude.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            **create_kw: Forwarded to :meth:`create` (``wind_profile``,
                ``scheme``, ``mask``, ``cg_tol``, ``cg_max_steps``).

        Returns:
            ``(model, scales)`` with ``scales.kind == "planetary"``.

        Raises:
            ValueError: If a dimensionless input is out of range.

        Example:
            >>> model, scales = SphericalQG.from_nondimensional(
            ...     nx=64,
            ...     ny=32,
            ...     rossby=0.05,
            ...     ekman=0.01,
            ... )
            >>> scales.kind
            'planetary'
        """
        context = "SphericalQG.from_nondimensional"
        require_positive(context, rossby=rossby)
        require_non_negative(
            context,
            ekman=ekman,
            ekman_lateral=ekman_lateral,
            wind_hat=wind_hat,
        )

        # Unit scales: a = Omega = 1, so f0 = 2 and U = 2 Ro. The wind
        # enters as a vorticity tendency, so it carries f0 * U/a rather
        # than the f0 * U of a momentum forcing.
        f0 = 2.0
        vorticity_scale = f0 * rossby  # = U / a
        model = SphericalQG.create(
            nx=nx,
            ny=ny,
            lon_range=lon_range,
            lat_range=lat_range,
            radius=1.0,
            omega=1.0,
            lateral_viscosity=f0 * ekman_lateral,
            bottom_drag=f0 * ekman,
            wind_amplitude=wind_hat * f0 * vorticity_scale,
            **create_kw,
        )
        scales = Scales.planetary(a=1.0, Omega=1.0, H=1.0, rossby=rossby)
        return model, scales

    @staticmethod
    def create(
        nx: int = 128,
        ny: int = 64,
        lon_range: tuple[float, float] = (0.0, 360.0),
        lat_range: tuple[float, float] = (-80.0, 80.0),
        radius: float = SPHERE.semi_major,
        omega: float = SPHERE.rotation_rate,
        lateral_viscosity: float = 0.0,
        bottom_drag: float = 0.0,
        wind_amplitude: float = 0.0,
        wind_profile: str = "zonal",
        scheme: Scheme = UPWIND,
        mask: Mask2D | None = None,
        cg_tol: float = 1e-6,
        cg_max_steps: int = 500,
        wind_forcing: Float[Array, "Ny Nx"] | None = None,
    ) -> SphericalQG:
        """Convenience factory.

        Args:
            nx: Interior cells in longitude.
            ny: Interior cells in latitude.
            lon_range: ``(lon_min, lon_max)`` in degrees.
            lat_range: ``(lat_min, lat_max)`` in degrees.
            radius: Planet radius (m).
            omega: Planet angular velocity (rad/s).
            lateral_viscosity: Harmonic viscosity (m^2/s).
            bottom_drag: Linear bottom drag (1/s).
            wind_amplitude: Wind-stress-curl amplitude (1/s^2).
            wind_profile: ``"zonal"`` gives a ``-sin(2 lat)`` curl;
                ``"none"`` gives no pattern.
            scheme: Scheme of the advective face values.
            mask: Optional land/ocean mask.
            cg_tol: Convergence tolerance for the PV inversion, used for
                both the relative and the absolute criterion. The default
                is chosen for float32: a tighter absolute tolerance never
                trips, and CG runs to its step cap.
            cg_max_steps: Iteration cap for the PV inversion.
            wind_forcing: Precomputed wind-stress-curl pattern on the
                model grid, overriding ``wind_profile``. For a scenario
                that supplies its own forcing field.

        Returns:
            A ``SphericalQG`` instance.
        """
        context = "SphericalQG.create"
        require_global_longitude(context, lon_range)
        require_open_poles(context, lat_range)
        grid = SphericalCGrid2D.from_interior(nx, ny, lon_range, lat_range, R=radius)

        params = SphericalQGParams(
            lateral_viscosity=as_parameter(lateral_viscosity),
            bottom_drag=as_parameter(bottom_drag),
            wind_amplitude=as_parameter(wind_amplitude),
        )
        consts = SphericalQGPhysConsts(omega=omega, radius=radius)

        lat_T = latitudes(grid)
        f_field = 2.0 * omega * jnp.sin(lat_T)
        if wind_forcing is not None:
            wind_forcing = jnp.asarray(wind_forcing)
        elif wind_profile == "none":
            wind_forcing = jnp.zeros_like(lat_T)
        else:
            wind_forcing = -jnp.sin(2.0 * lat_T)

        return SphericalQG(
            params=params,
            consts=consts,
            grid=grid,
            ops=GridOps(grid, mask),
            mask=mask,
            f_field=f_field,
            wind_forcing=wind_forcing,
            scheme=scheme,
            cg_tol=cg_tol,
            cg_max_steps=cg_max_steps,
        )
