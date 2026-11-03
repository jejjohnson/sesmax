"""2D incompressible Navier-Stokes in vorticity-streamfunction form."""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from finitevolx import (
    Advection2D as FVXAdvection2D,
    CartesianGrid2D,
    Difference2D,
    Interpolation2D,
    Mask2D,
    enforce_periodic,
    streamfunction_from_vorticity,
)
from jaxtyping import Array, Float, PyTree
from spectraldiffx import solve_helmholtz_2d

from sesmax_core._src._validation import require_positive
from sesmax_core._src.model import Model
from sesmax_core._src.scales import Scales
from sesmax_core._src.types import Diagnostics, Params, State, as_parameter


class NSVorticityState(State):
    """State for the vorticity-streamfunction NS formulation.

    Args:
        omega: Vorticity field on T-points, shape ``(Ny, Nx)``.
        flux: Volume flux through the channel per unit depth,
            ``Q = psi(Ly) - psi(0)``, a scalar. Only ``problem="channel"``
            evolves it; the other problems carry it at zero.
    """

    omega: Array
    flux: Array = eqx.field(default_factory=lambda: jnp.zeros(()))

    # ``omega`` is this family's spelling of vorticity; the flux is a
    # difference of streamfunction values.
    scale_kinds: ClassVar[dict[str, str]] = {
        "omega": "vorticity",
        "flux": "streamfunction",
    }
    # Vorticity is carried at T-points, not the C-grid corner.
    mask_locations: ClassVar[dict[str, str]] = {"omega": "h"}


class NSParams(Params):
    """Differentiable parameters for incompressible Navier-Stokes.

    Args:
        nu: Kinematic viscosity.
    """

    nu: Array


class NSDiagnostics(Diagnostics):
    """Diagnostics for incompressible Navier-Stokes.

    Args:
        psi: Streamfunction.
        u: x-velocity.
        v: y-velocity.
        kinetic_energy: Domain-averaged kinetic energy.
        enstrophy: Domain-averaged enstrophy.
    """

    psi: Float[Array, "Ny Nx"]
    u: Float[Array, "Ny Nx"]
    v: Float[Array, "Ny Nx"]
    kinetic_energy: Array
    enstrophy: Array


# -----------------------------------------------------------------------
# Ghia et al. (1982) benchmark data for lid-driven cavity at Re = 100
# u-velocity along vertical centreline (y, u) and
# v-velocity along horizontal centreline (x, v).
# -----------------------------------------------------------------------
GHIA_RE100_Y = jnp.array(
    [
        0.0000,
        0.0547,
        0.0625,
        0.0703,
        0.1016,
        0.1719,
        0.2813,
        0.4531,
        0.5000,
        0.6172,
        0.7344,
        0.8516,
        0.9531,
        0.9609,
        0.9688,
        0.9766,
        1.0000,
    ]
)
GHIA_RE100_U = jnp.array(
    [
        0.00000,
        -0.03717,
        -0.04192,
        -0.04775,
        -0.06434,
        -0.10150,
        -0.15662,
        -0.21090,
        -0.20581,
        -0.13641,
        0.00332,
        0.23151,
        0.68717,
        0.73722,
        0.78871,
        0.84123,
        1.00000,
    ]
)
GHIA_RE100_X = jnp.array(
    [
        0.0000,
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5000,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
        1.0000,
    ]
)
GHIA_RE100_V = jnp.array(
    [
        0.00000,
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
        0.00000,
    ]
)


#: The transform of the streamfunction solve for each problem.
_POISSON_BC: dict[str, str] = {"cavity": "dst", "periodic": "fft", "channel": "mixed"}


class IncompressibleNS2D(Model):
    r"""2D incompressible Navier-Stokes (vorticity-streamfunction).

    Solves the vorticity transport equation::

        d(omega)/dt + u * d(omega)/dx + v * d(omega)/dy = nu * laplacian(omega)

    where the velocity is recovered at each step via the Poisson
    inversion :math:`\nabla^2 \psi = -\omega` and
    :math:`u = \partial\psi/\partial y`,
    :math:`v = -\partial\psi/\partial x`.

    Supports three problems:

    - **Lid-driven cavity** (``problem="cavity"``): no-slip walls on all
      four sides, the top one sliding at ``u_lid``; psi = 0 on the walls
      (``poisson_bc="dst"``).
    - **Doubly periodic box** (``problem="periodic"``): no walls; the
      ghost ring wraps in x and y (``poisson_bc="fft"``). Taylor-Green
      vortices are an exact solution here.
    - **Re-entrant channel** (``problem="channel"``): periodic in x,
      stationary no-slip walls at y = 0 and y = Ly
      (``poisson_bc="mixed"``), driven by a uniform pressure gradient
      ``G = -(1/rho) dp/dx``. Psi is 0 on the bottom wall and equals the
      state's flux ``Q`` on the top wall, and ``Q`` follows the
      x-averaged momentum budget
      ``dQ/dt = G Ly - nu (mean omega_top - mean omega_bottom)``.
      The steady state is Poiseuille flow,
      ``u = (G / 2 nu) y (Ly - y)`` with ``Q = G Ly**3 / (12 nu)``.

    The cavity and channel set the wall vorticity with Thom's formula
    on the wall nodes of the streamfunction solve (rows and columns 0
    and -2; -1 is a ghost).

    Args:
        params: Differentiable parameters (viscosity ``nu``).
        grid: 2D Arakawa C-grid.
        diff: Difference operators.
        interp: Interpolation operators.
        advection: Advection operator.
        mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
        problem: ``"cavity"``, ``"periodic"`` or ``"channel"``.
        poisson_bc: Spectral solver BC type for Poisson inversion
            (``"dst"``, ``"fft"`` or ``"mixed"``; set by the problem).
        u_lid: Lid velocity for cavity flow (default 1.0); the other
            problems ignore it.
        body_force: Constant vorticity source, added in every problem.
        pressure_gradient: Uniform driving ``G = -(1/rho) dp/dx`` of the
            channel; the other problems ignore it.
        method: Advection reconstruction method.
    """

    params: NSParams
    grid: CartesianGrid2D = eqx.field(static=True)
    diff: Difference2D
    interp: Interpolation2D
    advection: FVXAdvection2D
    mask: Mask2D | None
    problem: str = eqx.field(static=True, default="cavity")
    poisson_bc: str = eqx.field(static=True, default="dst")
    u_lid: float = eqx.field(static=True, default=1.0)
    body_force: float = eqx.field(static=True, default=0.0)
    method: str = eqx.field(static=True, default="upwind1")
    pressure_gradient: float = eqx.field(static=True, default=0.0)

    def _solve_psi(
        self, omega: Array, flux: Array | float = 0.0
    ) -> Float[Array, "Ny Nx"]:
        """Recover the streamfunction from vorticity, nabla^2 psi = -omega.

        The cavity solves with psi = 0 on the walls. The periodic box
        and the channel fill the ghost ring as well, so the velocity
        stencils at the domain edges read periodic neighbours.
        """
        if self.problem == "periodic":
            psi = streamfunction_from_vorticity(
                -omega, self.grid.dx, self.grid.dy, bc="fft"
            )
            return enforce_periodic(psi)
        if self.problem == "channel":
            return self._solve_channel_psi(omega, flux)
        return streamfunction_from_vorticity(
            -omega, self.grid.dx, self.grid.dy, bc=self.poisson_bc
        )

    def _solve_channel_psi(
        self, omega: Array, flux: Array | float
    ) -> Float[Array, "Ny Nx"]:
        """Channel streamfunction: periodic in x, psi = 0 and Q on the walls.

        The wall nodes are rows 0 and -2; the unknowns are rows
        ``1:-2`` and the periodic columns ``1:-1``.
        """
        n_cols = self.grid.Nx - 2
        rhs = -omega[1:-2, 1:-1]
        bottom = jnp.zeros(n_cols, dtype=omega.dtype)
        top = jnp.full(n_cols, flux, dtype=omega.dtype)
        interior = solve_helmholtz_2d(
            rhs,
            self.grid.dx,
            self.grid.dy,
            bc_x="periodic",
            bc_y="dirichlet",
            bc_y_values=(bottom, top),
        )
        psi = jnp.zeros_like(omega)
        psi = psi.at[1:-2, 1:-1].set(interior)
        # Wall nodes, and the ghost row above the top wall at the wall value.
        psi = psi.at[-2:, :].set(jnp.asarray(flux, dtype=omega.dtype))
        # Periodic ghost columns.
        psi = psi.at[:, 0].set(psi[:, -2])
        return psi.at[:, -1].set(psi[:, 1])

    def _wrap(self, field: Array) -> Array:
        """Fill the periodic ghosts of a field the stencils left at zero.

        finitevolX's interpolations write only the interior, so on a
        periodic axis the edge stencils would average with a zero ghost.
        T-, U- and V-point arrays share the same-index rule, so one copy
        serves all three.
        """
        if self.problem == "periodic":
            return enforce_periodic(field)
        if self.problem == "channel":
            field = field.at[:, 0].set(field[:, -2])
            return field.at[:, -1].set(field[:, 1])
        return field

    def _flux_tendency(self, omega: Array) -> Array:
        """dQ/dt = G Ly - nu (mean omega_top - mean omega_bottom).

        The x-averaged momentum budget integrated across the channel:
        on a stationary no-slip wall du/dy = -omega and uv = 0.
        """
        if self.problem != "channel":
            return jnp.zeros((), dtype=omega.dtype)
        wall_top = jnp.mean(omega[-2, 1:-1])
        wall_bottom = jnp.mean(omega[0, 1:-1])
        forcing = self.pressure_gradient * self.grid.Ly
        return forcing - self.params.nu * (wall_top - wall_bottom)

    def _velocity_from_psi(
        self, psi: Float[Array, "Ny Nx"]
    ) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
        """Compute velocity from streamfunction."""
        u = self.diff.diff_y_T_to_V(psi)  # u = dpsi/dy at V-points
        v = -self.diff.diff_x_T_to_U(psi)  # v = -dpsi/dx at U-points
        return u, v

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> NSVorticityState:
        """Compute vorticity tendency."""
        omega = state.omega
        # 1. Poisson inversion: nabla^2 psi = -omega
        psi = self._solve_psi(omega, state.flux)
        # 2. Velocity from streamfunction
        u_V, v_U = self._velocity_from_psi(psi)
        # Interpolate to T-points for scalar advection
        u_T = self._wrap(self.interp.V_to_T(self._wrap(u_V)))
        v_T = self._wrap(self.interp.U_to_T(self._wrap(v_U)))
        # Interpolate to face points for transport
        u_on_U = self._wrap(self.interp.T_to_U(u_T))
        v_on_V = self._wrap(self.interp.T_to_V(v_T))
        # 3. Advection of vorticity; the periodic box and the channel
        # read the periodic ghost columns, so their edge faces are open.
        wall = "closed" if self.problem == "cavity" else "open"
        adv = self.advection(omega, u_on_U, v_on_V, method=self.method, wall=wall)
        # 4. Diffusion
        diff = self.params.nu * self.diff.laplacian(omega)
        # 5. Body force
        domega_dt = adv + diff + self.body_force
        return NSVorticityState(omega=domega_dt, flux=self._flux_tendency(omega))

    def apply_boundary_conditions(self, state: PyTree) -> NSVorticityState:
        """Apply vorticity boundary conditions.

        For cavity flow (``problem="cavity"``): Thom's formula at all walls.
        For the periodic box (``problem="periodic"``): a periodic ghost
        ring. For the channel (``problem="channel"``): periodic columns,
        and Thom's formula on the walls with psi = 0 below and Q above.
        """
        omega = state.omega
        if self.problem == "periodic":
            omega = enforce_periodic(omega)
        elif self.problem == "channel":
            # Periodic columns first, so the solve sees a periodic source.
            omega = omega.at[:, 0].set(omega[:, -2])
            omega = omega.at[:, -1].set(omega[:, 1])
            psi = self._solve_psi(omega, state.flux)
            dy2 = self.grid.dy**2
            # Walls on rows 0 (psi = 0) and -2 (psi = Q); row -1 is a ghost.
            # Bottom: omega = 2 (psi_wall - psi[1]) / dy^2
            omega = omega.at[0, :].set(-2.0 * psi[1, :] / dy2)
            # Top: omega = 2 (psi_wall - psi[-3]) / dy^2
            omega = omega.at[-2, :].set(2.0 * (state.flux - psi[-3, :]) / dy2)
            omega = omega.at[-1, :].set(omega[-2, :])
        else:
            # Cavity: all walls are no-slip
            psi = self._solve_psi(omega)
            dx2 = self.grid.dx**2
            dy2 = self.grid.dy**2
            # The DST solve puts the walls (psi = 0) on rows and columns
            # 0 and -2; row and column -1 are ghosts outside the domain.
            # Bottom wall (y=0): omega = -2*psi[1]/dy^2
            omega = omega.at[0, :].set(-2.0 * psi[1, :] / dy2)
            # Top wall (y=Ly): omega = -2*psi[-3]/dy^2 - 2*u_lid/dy
            omega = omega.at[-2, :].set(
                -2.0 * psi[-3, :] / dy2 - 2.0 * self.u_lid / self.grid.dy
            )
            # Left wall (x=0): omega = -2*psi[:,1]/dx^2
            omega = omega.at[:, 0].set(-2.0 * psi[:, 1] / dx2)
            # Right wall (x=Lx): omega = -2*psi[:,-3]/dx^2
            omega = omega.at[:, -2].set(-2.0 * psi[:, -3] / dx2)
        return NSVorticityState(omega=omega, flux=state.flux)

    def diagnose(self, state: PyTree) -> NSDiagnostics:
        """Compute velocity, streamfunction, KE, and enstrophy."""
        psi = self._solve_psi(state.omega, state.flux)
        u_V, v_U = self._velocity_from_psi(psi)
        u_T = self._wrap(self.interp.V_to_T(self._wrap(u_V)))
        v_T = self._wrap(self.interp.U_to_T(self._wrap(v_U)))
        interior_u = u_T[1:-1, 1:-1]
        interior_v = v_T[1:-1, 1:-1]
        interior_w = state.omega[1:-1, 1:-1]
        ke = 0.5 * jnp.mean(interior_u**2 + interior_v**2)
        enstrophy = 0.5 * jnp.mean(interior_w**2)
        return NSDiagnostics(
            psi=psi,
            u=u_T,
            v=v_T,
            kinetic_energy=ke,
            enstrophy=enstrophy,
        )

    @staticmethod
    def from_nondimensional(
        *,
        nx: int = 64,
        ny: int = 64,
        aspect: float = 1.0,
        reynolds: float,
        **create_kw: Any,
    ) -> tuple[IncompressibleNS2D, Scales]:
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
            **create_kw: Forwarded to :meth:`create` (``problem``,
                ``u_lid``, ``body_force``, ``method``, ``mask``).

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
        context = "IncompressibleNS2D.from_nondimensional"
        require_positive(context, aspect=aspect)
        require_positive(context, reynolds=reynolds)
        model = IncompressibleNS2D.create(
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
        Lx: float = 1.0,
        Ly: float = 1.0,
        nu: float = 0.01,
        problem: str = "cavity",
        u_lid: float = 1.0,
        body_force: float = 0.0,
        method: str = "upwind1",
        mask: Mask2D | None = None,
        pressure_gradient: float = 0.0,
    ) -> IncompressibleNS2D:
        """Convenience factory.

        Args:
            nx: Number of interior cells in x.
            ny: Number of interior cells in y.
            Lx: Domain length in x.
            Ly: Domain length in y.
            nu: Kinematic viscosity.
            problem: ``"cavity"`` (closed box, moving lid),
                ``"periodic"`` (doubly periodic) or ``"channel"``
                (periodic in x, walls in y).
            u_lid: Lid velocity for cavity flow.
            body_force: Constant vorticity source, in every problem.
            method: Advection reconstruction method.
            mask: Optional Arakawa C-grid mask (``None`` = all-ocean).
                The periodic box and the channel take no mask.
            pressure_gradient: Uniform driving ``G = -(1/rho) dp/dx`` of
                the channel.

        Returns:
            An ``IncompressibleNS2D`` model instance.

        Raises:
            ValueError: If ``problem`` is unknown, or a mask is given for
                the periodic box or the channel.
        """
        if problem not in _POISSON_BC:
            raise ValueError(
                f"problem must be one of {sorted(_POISSON_BC)}, got {problem!r}"
            )
        if mask is not None and problem != "cavity":
            raise ValueError(f"problem={problem!r} does not take a mask")
        grid = CartesianGrid2D.from_interior(nx, ny, Lx, Ly)
        params = NSParams(nu=as_parameter(nu))
        diff = Difference2D(grid=grid, mask=mask)
        interp = Interpolation2D(grid=grid, mask=mask)
        advection = FVXAdvection2D(grid=grid, mask=mask)
        poisson_bc = _POISSON_BC[problem]
        return IncompressibleNS2D(
            params=params,
            grid=grid,
            diff=diff,
            interp=interp,
            advection=advection,
            mask=mask,
            problem=problem,
            poisson_bc=poisson_bc,
            u_lid=u_lid,
            body_force=body_force,
            method=method,
            pressure_gradient=pressure_gradient,
        )
