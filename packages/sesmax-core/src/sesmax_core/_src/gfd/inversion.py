"""The streamfunction, pressure and QG PV inversions on the C-grid.

Three elliptic inversions of a layered-fluid model, each a composition of
the pieces built from the grid and the mask (gridax) and the generic solves
(gaussx, spectraldiffx):

    (∇² - λ) ψ = ζ          streamfunction_from_vorticity   ψ at X points
    ∇² p = ∇ · u            pressure_from_divergence        p at T points
    (∇² - λ_k) ψ_k = q_k    pv_inversion                    ψ_k at X points

| ``method``      | gridax supplies                          | the solve       |
| --------------- | ---------------------------------------- | --------------- |
| ``spectral``    | the wall of each (staggering, bc)        | spectraldiffx   |
| ``cg``          | the masked operator, the spectral        | gaussx CG       |
|                 | preconditioner                           |                 |
| ``capacitance`` | ``build_capacitance_solver`` (the wall)  | spectraldiffx   |
| ``multigrid``   | ``build_multigrid_solver`` (the levels)  | gaussx CG with  |
|                 |                                          | one V-cycle     |
| known values    | ``SolveDomain`` (the coast ring)         | gaussx lifting  |

Moved from gridax-jax's ``solvers/elliptic.py`` at gridax ``01d8320``
(sesmax#126, geostack#291 rule 3), with the arithmetic unchanged: every
method returned arrays bit for bit equal to gridax-jax's when it moved
(sesmax#129). The CG solve is gaussx's
``linear_solve_with_info`` in place of gridax-jax's deprecated ``solve_cg``,
which it reproduces bit for bit; ``method="multigrid"`` is new.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import gaussx
import jax
import jax.numpy as jnp
import numpy as np
from gridax import BoundaryConditionSet, Dirichlet1D, Mask2D, Neumann1D, Periodic1D
from gridax.solvers import (
    CapacitanceSolver,
    MultigridSolver,
    SolveDomain,
    coast_ring,
    make_spectral_preconditioner,
    masked_face_dirichlet_laplacian,
    masked_laplacian,
    masked_neumann_laplacian,
    project_zero_mean,
    solve_helmholtz_dct1,
    solve_helmholtz_dct2,
    solve_helmholtz_dst1,
    solve_helmholtz_dst2,
    solve_helmholtz_fft,
    wet_components,
)
from jaxtyping import Array, Bool, Float
from scipy import ndimage


__all__ = [
    "pressure_from_divergence",
    "pv_inversion",
    "streamfunction_from_vorticity",
]


_MaskLike = Float[Array, "Ny Nx"] | Mask2D
_PrecondLike = Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]]
_BCLike = str | BoundaryConditionSet
_Method = Literal["spectral", "cg", "capacitance", "multigrid"]
_METHODS = ("spectral", "cg", "capacitance", "multigrid")
_METHOD_ERROR = "method must be 'spectral', 'cg', 'capacitance' or 'multigrid'"

# CG settings of gridax-jax's ``solve_cg``, which the wrappers used.
_CG_RTOL = 1e-6
_CG_ATOL_FLOAT64 = 1e-6
_CG_MAX_STEPS = 500


# ---------------------------------------------------------------------------
# The solves
# ---------------------------------------------------------------------------


def _cg(
    matvec: Callable[[Float[Array, "Ny Nx"]], Float[Array, "Ny Nx"]],
    b: Float[Array, "Ny Nx"],
    preconditioner: _PrecondLike,
) -> Float[Array, "Ny Nx"]:
    """Preconditioned CG on a negative-definite grid operator, raising if unconverged.

        x = A⁻¹ b,   M⁻¹ ≈ A⁻¹

    gaussx solves (-A) x = -b, so the preconditioner it receives is -M⁻¹.
    The tolerances are gridax-jax ``solve_cg``'s: ``rtol = 1e-6`` and, in
    float64, ``atol = 1e-6``; below float64 gaussx's default, √ε relative
    to maxₙ |b_n|. Raises (through `equinox.error_if`, so also under
    ``jit``) when CG has not converged in 500 steps or its explicit residual
    exceeds the tolerance.

    Args:
        matvec: The operator A on an ``[Ny, Nx]`` field.
        b: The right-hand side, ``[Ny, Nx]``.
        preconditioner: M⁻¹ ≈ A⁻¹ (negative definite, like A).

    Returns:
        The solution x, ``[Ny, Nx]``.
    """
    operator = gaussx.as_linear_operator(
        matvec,
        in_structure=jax.ShapeDtypeStruct(b.shape, b.dtype),
        negative_definite=True,
    )
    atol = _CG_ATOL_FLOAT64 if jnp.finfo(b.dtype).bits >= 64 else None
    solver = gaussx.CGSolver(rtol=_CG_RTOL, atol=atol, max_steps=_CG_MAX_STEPS)

    def positive(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        # (-A)⁻¹ ≈ -M⁻¹
        return -preconditioner(r)

    x, info = gaussx.linear_solve_with_info(
        operator, b, solver=solver, preconditioner=positive
    )
    return eqx.error_if(
        x,
        jnp.logical_not(info.converged),
        "CG did not converge: max_steps reached, or the residual |b - A x| "
        "exceeds atol + rtol |b| (an indefinite operator, e.g. lambda_ < 0, or "
        "a preconditioner of the wrong sign).",
    )


def _resolve_mask_arr(
    mask: _MaskLike | None,
    dtype: jnp.dtype,
    staggering: str = "X",
) -> Float[Array, "Ny Nx"] | None:
    """The float mask of the unknown's staggering, in the right-hand side's dtype.

    A `gridax.Mask2D` gives ``xy_corner_strict`` for ``"X"`` (ψ) and ``h``
    for ``"T"`` (pressure); an array mask is used as given. The mask takes
    the right-hand side's dtype: a strongly typed float32 mask times a
    weakly typed float64 field is float32, which would run the whole solve
    in single precision under x64.

    Args:
        mask: The mask, or ``None``.
        dtype: The right-hand side's dtype.
        staggering: ``"X"`` or ``"T"``.

    Returns:
        The mask as an ``[Ny, Nx]`` array of ``dtype``, or ``None``.
    """
    if mask is None:
        return None
    if isinstance(mask, Mask2D):
        stag_mask = mask.xy_corner_strict if staggering == "X" else mask.h
        return jnp.asarray(stag_mask, dtype=dtype)
    return jnp.asarray(mask, dtype=dtype)


# Spectral solve per (staggering, bc): the interior slice of the [Ny, Nx]
# array that holds the unknowns, and the transform whose implicit boundary
# matches the C-grid wall there.  On a ``from_interior`` grid the T cells
# are [1:-1] with the walls on the faces at 1/2 and N-3/2; X node i sits at
# i + 1/2, so the walls are the X nodes 0 and N-2 (N-1 is outside).
#
#   X, dst : DST-I  on [1:-2, 1:-2]  psi = 0 on the wall nodes 0 and N-2
#   X, dct : DCT-I  on [0:-1, 0:-1]  d(psi)/dn = 0 reflected about nodes 0, N-2
#            (the wall nodes are unknowns: the rhs is read there too)
#   X, fft : FFT    on [1:-1, 1:-1]  periodic, period N-2
#   T, dct : DCT-II on [1:-1, 1:-1]  dp/dn = 0 on the faces (half a cell out)
#   T, dst : DST-II on [1:-1, 1:-1]  p = 0 on the faces
#   T, fft : FFT    on [1:-1, 1:-1]  periodic, period N-2
_SPECTRAL_STAGGERED: dict[tuple[str, str], tuple[slice, Callable]] = {
    ("X", "dst"): (slice(1, -2), solve_helmholtz_dst1),
    ("X", "dct"): (slice(0, -1), solve_helmholtz_dct1),
    ("X", "fft"): (slice(1, -1), solve_helmholtz_fft),
    ("T", "dct"): (slice(1, -1), solve_helmholtz_dct2),
    ("T", "dst"): (slice(1, -1), solve_helmholtz_dst2),
    ("T", "fft"): (slice(1, -1), solve_helmholtz_fft),
}


def _solve_spectral(
    rhs: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    bc: str,
    staggering: str,
) -> Float[Array, "... Ny Nx"]:
    """Rectangular spectral solve with the wall on the C-grid wall.

    Slices the unknowns of ``staggering`` out of the full array (see
    ``_SPECTRAL_STAGGERED``), solves, and embeds the result in zeros: the
    ghost ring (and, for ψ, the wall nodes) is left at zero. No Python
    branch on ``lambda_``, so it is safe under ``vmap`` with a traced
    Helmholtz parameter.
    """
    entry = _SPECTRAL_STAGGERED.get((staggering, bc))
    if entry is None:
        raise ValueError(f"bc must be 'fft', 'dst', or 'dct'; got {bc!r}")
    sl, solver = entry
    # psi[..., j, i] = solver(rhs[..., sl, sl])[j - j0, i - i0] on the slice,
    # 0 elsewhere
    psi = solver(rhs[..., sl, sl], dx, dy, lambda_)
    return jnp.zeros_like(rhs).at[..., sl, sl].set(psi.astype(rhs.dtype))


def _check_cg_bc(bc: str, staggering: str, method: str = "cg") -> None:
    """Raise unless a mask-based CG solve can honour ``bc`` for this staggering.

    CG takes the coast condition from ``bc``: ``"dst"`` is Dirichlet on the
    coast nodes (`gridax.solvers.masked_laplacian`), ``"dct"`` is zero flux
    through the coast faces (T points only). A periodic ``"fft"`` has no
    meaning on a masked domain.
    """
    if bc == "dst" or (bc == "dct" and staggering == "T"):
        return
    if bc == "dct":
        raise ValueError(
            f"method={method!r} with bc='dct' (zero flux through the coast faces) "
            "is only defined for the T-point pressure solve; the streamfunction "
            "solves need bc='dst' (psi prescribed on the coast)."
        )
    raise ValueError(
        f"method={method!r} takes its coast condition from bc: 'dst' (Dirichlet) "
        f"or, for pressure, 'dct' (Neumann); got bc={bc!r}.  A periodic 'fft' "
        "cannot be imposed on a masked domain."
    )


def _cg_masked_solve(
    rhs: Float[Array, "Ny Nx"],
    solve_mask: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    bc: str,
    preconditioner: _PrecondLike | None,
    staggering: str,
) -> Float[Array, "Ny Nx"]:
    """Preconditioned CG on ``solve_mask`` with the coast condition of ``bc``.

    The wall is where the spectral solve of the same staggering puts it:

    * ψ (X), ``"dst"``: ψ = 0 on the dry nodes next to the wet ones
      (`gridax.solvers.masked_laplacian`), the C-grid wall nodes.
    * p (T), ``"dst"``: p = 0 on the wet-dry *faces*, half a cell out
      (the DST-II wall).
    * p (T), ``"dct"``: zero flux through every wet-dry face (the DCT-II
      wall). At a concrete λ = 0 this is singular (a constant per connected
      wet component), so the right-hand side is projected to zero mean on
      each component and the zero-mean solution is returned; a traced mask
      cannot be labelled, so it is treated as one component.

    The default preconditioner is the spectral solve of the same ``bc``.
    Safe under ``vmap`` with a traced ``lambda_``.
    """
    if staggering == "X":
        operator = masked_laplacian
    elif bc == "dst":
        operator = masked_face_dirichlet_laplacian
    else:
        operator = masked_neumann_laplacian

    def _matvec(x: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return operator(x, solve_mask, dx, dy, lambda_=lambda_)

    if preconditioner is None:
        preconditioner = make_spectral_preconditioner(dx, dy, lambda_=lambda_, bc=bc)

    b = rhs * solve_mask  # [Ny, Nx]
    project = None
    if bc == "dct" and _is_zero(lambda_):
        try:
            labels, counts = wet_components(np.asarray(solve_mask))
        except jax.errors.TracerArrayConversionError:
            # Traced mask: one component, labels = 1 on the wet cells and
            # counts = [1, n_wet].
            labels = (solve_mask > 0.5).astype(jnp.int32)
            counts = jnp.stack([jnp.ones((), b.dtype), jnp.sum(solve_mask)])

        def project(u: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            # u <- u - mean(u over its wet component): the consistent rhs and
            # the zero-mean member of the solution family p + c.
            return project_zero_mean(u, solve_mask, labels, counts)

        b = project(b)
    x = _cg(_matvec, b, preconditioner)
    x = x * solve_mask
    if project is not None:
        x = project(x)
    return x


def _solve_cg_method(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    mask: _MaskLike | None,
    preconditioner: _PrecondLike | None,
    staggering: str = "X",
    bc: str = "dst",
    method: str = "cg",
) -> Float[Array, "Ny Nx"]:
    """Solve with preconditioned CG on a masked domain."""
    _check_cg_bc(bc, staggering, method)
    mask_arr = _resolve_mask_arr(mask, rhs.dtype, staggering)
    if mask_arr is None:
        raise ValueError(f"method={method!r} requires a mask (array or Mask2D)")
    return _cg_masked_solve(
        rhs, mask_arr, dx, dy, lambda_, bc, preconditioner, staggering
    )


def _v_cycle_preconditioner(
    multigrid_solver: MultigridSolver | None, bc: str, staggering: str
) -> _PrecondLike:
    """One V-cycle of a prebuilt hierarchy from a zero guess, r ↦ V(0, r) ≈ A⁻¹ r.

    The hierarchy must be built on the solve's mask, ``lambda_`` and wall:
    ``build_multigrid_solver(mask, dx, dy, lambda_)`` (Dirichlet) for ψ, or
    ``bc="neumann"`` on ``mask.h`` for the pressure with ``bc="dct"``. A
    T-point ``bc="dst"`` puts its wall on the faces, which the hierarchy's
    Dirichlet wall (on the dry centres) is not, so it raises.
    """
    if multigrid_solver is None:
        raise ValueError(
            "method='multigrid' requires a pre-built MultigridSolver "
            "(see gridax.solvers.build_multigrid_solver)"
        )
    if staggering == "T" and bc == "dst":
        raise ValueError(
            "method='multigrid' with bc='dst' is not available for the T-point "
            "pressure solve: its wall is on the cell faces, while the multigrid "
            "hierarchy's Dirichlet wall is on the dry cell centres.  Use "
            "method='cg', or bc='dct' with a bc='neumann' hierarchy."
        )
    wanted = "neumann" if bc == "dct" else "dirichlet"
    if multigrid_solver.bc != wanted:
        raise ValueError(
            f"bc={bc!r} needs a multigrid hierarchy built with bc={wanted!r}; "
            f"got one built with bc={multigrid_solver.bc!r}."
        )

    def v_cycle(r: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
        return multigrid_solver.v_cycle(jnp.zeros_like(r), r)

    return v_cycle


def _check_capacitance_bc(bc: str) -> None:
    """Raise unless ``bc`` is the Dirichlet coast the capacitance solver imposes."""
    if bc != "dst":
        raise ValueError(
            "method='capacitance' imposes psi = 0 (Dirichlet) on the coast, so "
            f"it needs bc='dst'; got bc={bc!r}.  For a Neumann (bc='dct') "
            "pressure solve on a masked domain use method='cg'."
        )


def _solve_capacitance_method(
    rhs: Float[Array, "Ny Nx"],
    capacitance_solver: CapacitanceSolver | None,
) -> Float[Array, "Ny Nx"]:
    """Solve with a pre-built capacitance matrix solver."""
    if capacitance_solver is None:
        raise ValueError(
            "method='capacitance' requires a pre-built CapacitanceSolver "
            "(see gridax.solvers.build_capacitance_solver)"
        )
    # The solver's precomputed arrays are float64 under x64: keep the rhs dtype.
    return capacitance_solver(rhs).astype(rhs.dtype)


def _solve_dispatch(
    rhs: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    bc: str,
    method: str,
    mask: _MaskLike | None,
    capacitance_solver: CapacitanceSolver | None,
    multigrid_solver: MultigridSolver | None,
    preconditioner: _PrecondLike | None,
    staggering: str,
) -> Float[Array, "Ny Nx"]:
    """Dispatch an elliptic solve to the selected method."""
    if method == "spectral":
        return _solve_spectral(rhs, dx, dy, lambda_, bc, staggering)
    if method == "cg":
        return _solve_cg_method(
            rhs, dx, dy, lambda_, mask, preconditioner, staggering, bc
        )
    if method == "capacitance":
        _check_capacitance_bc(bc)
        return _solve_capacitance_method(rhs, capacitance_solver)
    if method == "multigrid":
        _check_cg_bc(bc, staggering, method)
        v_cycle = _v_cycle_preconditioner(multigrid_solver, bc, staggering)
        return _solve_cg_method(
            rhs, dx, dy, lambda_, mask, v_cycle, staggering, bc, method
        )
    raise ValueError(f"{_METHOD_ERROR}; got {method!r}")


# ---------------------------------------------------------------------------
# Boundary-condition sets and known values
# ---------------------------------------------------------------------------


def _is_zero(value: object) -> bool:
    """True only for a concrete real zero; tracers count as non-zero.

    Accepts Python / NumPy numbers and concrete NumPy or JAX arrays whose
    entries are all zero, e.g. ``jnp.array(0.0)``. A tracer (a value under
    ``jit`` / ``vmap``) cannot be inspected, so it is treated as non-zero.
    """
    if isinstance(value, numbers.Real):
        return bool(value == 0)
    if not isinstance(value, (np.ndarray, np.generic, jax.Array)):
        return False
    try:
        arr = np.asarray(value)  # raises for a tracer
    except jax.errors.TracerArrayConversionError:
        return False
    return arr.dtype.kind in "iuf" and not arr.any()


def _resolve_bc(
    bc: _BCLike,
    method: str,
    mask: _MaskLike | None,
    known_values: Array | None,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    staggering: str,
) -> tuple[str, _MaskLike | None, Array | None]:
    """Translate ``bc`` into ``(bc_str, mask, known_values)``.

    A string passes through unchanged. A `gridax.BoundaryConditionSet`
    selects the spectral transform from its face types (all Dirichlet →
    ``"dst"``, all zero Neumann → ``"dct"``, all periodic → ``"fft"``),
    supplies ``bc.mask`` to the mask-based methods (``method="spectral"``
    with a ``bc.mask`` raises), and turns non-zero Dirichlet face values into
    known values on each face's wall (coast) nodes unless ``known_values``
    is given (which then wins).
    """
    if not isinstance(bc, BoundaryConditionSet):
        return bc, mask, known_values

    faces = (bc.south, bc.north, bc.west, bc.east)
    if all(isinstance(f, Dirichlet1D) for f in faces):
        bc_str = "dst"
    elif all(isinstance(f, Neumann1D) for f in faces):
        if not all(_is_zero(f.value) for f in faces):
            raise ValueError(
                "Inhomogeneous Neumann faces are not supported by the elliptic "
                "solvers; use Neumann1D faces with value=0."
            )
        bc_str = "dct"
    elif all(isinstance(f, Periodic1D) for f in faces):
        bc_str = "fft"
    else:
        names = [type(f).__name__ for f in faces]
        raise ValueError(
            "An elliptic solve needs all four faces of the BoundaryConditionSet "
            "to be Dirichlet1D, Neumann1D or Periodic1D (no mixing); got "
            f"south/north/west/east = {names}"
        )

    # The spectral path is rectangular: a bc.mask cannot be honoured there.
    if bc.mask is not None:
        if method == "spectral":
            raise ValueError(
                "method='spectral' solves the rectangle and cannot honour "
                "bc.mask; use method='cg' or 'capacitance' for a masked domain, "
                "or drop the mask from the BoundaryConditionSet."
            )
        if mask is not None:
            raise ValueError("Pass the mask either as bc.mask or as mask=, not both.")
        mask = bc.mask

    if (
        bc_str == "dst"
        and known_values is None
        and not all(_is_zero(f.value) for f in faces)
    ):
        if staggering == "T":
            # Dirichlet1D prescribes the value on the wall *face* (ghost =
            # 2 v - interior), but the lifting prescribes values at cell
            # centres: on T points that is the ghost-cell centre, half a cell
            # outside the face (a silent first-order wall error).  Refuse
            # rather than move the wall.
            raise ValueError(
                "Non-zero Dirichlet1D faces are not supported for the T-point "
                "pressure solve: the value belongs on the cell face, but known "
                "values can only be pinned at cell centres (the ghost cells, "
                "half a cell outside the face)."
            )
        mask_arr = _resolve_mask_arr(mask, dtype, staggering)
        wet = _x_basin(shape[-2:]) if mask_arr is None else mask_arr > 0.5
        _check_faces_cover_outer_coast(wet)
        known_values = _dirichlet_face_values(bc, wet, dtype)
    return bc_str, mask, known_values


def _x_basin(shape: tuple[int, ...]) -> Array:
    """The ψ (X-point) wet mask of a ``from_interior`` rectangle.

    It equals ``Mask2D.xy_corner_strict`` of that rectangle. X node i sits
    at i + 1/2 in T-index units: nodes 0 and N-2 are on the
    walls, N-1 is outside the domain, the wet nodes are [1:-2, 1:-2].
    """
    ny, nx = shape[-2], shape[-1]
    # basin[j, i] = True for 1 <= j <= Ny-3, 1 <= i <= Nx-3
    return jnp.zeros((ny, nx), dtype=bool).at[1:-2, 1:-2].set(True)


def _check_faces_cover_outer_coast(wet: Array) -> None:
    """Raise if the outer coast of ``wet`` is not on the four walls.

    A `gridax.BoundaryConditionSet` gives one value per face, which is only
    well defined when every coast node of the outer land (the dry region
    connected to the array edge) lies on one of the four wall lines around
    the wet bounding box: a rectangular basin, islands allowed (their coasts
    stay at zero). On a general coastline most coast nodes belong to no face
    and would silently get zero. Checked on concrete masks only
    (construction-time NumPy / SciPy); a traced mask is not checked.
    """
    try:
        wet_np = np.asarray(wet, dtype=bool)  # [Ny, Nx]
    except jax.errors.TracerArrayConversionError:
        return  # traced mask: cannot be labelled
    coast = np.asarray(coast_ring(jnp.asarray(wet_np)))  # [Ny, Nx]
    # 4-connected dry components; those touching the array edge are the outer land.
    labels, _ = ndimage.label(~wet_np)
    edge = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    outer_coast = coast & np.isin(labels, edge[edge > 0])
    rows = np.flatnonzero(wet_np.any(axis=1))
    cols = np.flatnonzero(wet_np.any(axis=0))
    j, i = np.indices(wet_np.shape)
    # wall lines: the row / column just outside the wet bounding box
    on_walls = (
        (j == rows[0] - 1)
        | (j == rows[-1] + 1)
        | (i == cols[0] - 1)
        | (i == cols[-1] + 1)
    )
    n_off = int(np.sum(outer_coast & ~on_walls))
    if n_off:
        raise ValueError(
            "Non-zero Dirichlet1D face values need a basin whose outer coast "
            "lies on the four walls (a rectangle; islands are allowed and stay "
            f"at zero), but {n_off} outer coast nodes are on no face.  For a "
            "general coastline pass known_values on gridax.solvers.coast_ring(mask)."
        )


def _dirichlet_face_values(
    bc: BoundaryConditionSet, wet: Array, dtype: jnp.dtype
) -> Float[Array, "Ny Nx"]:
    """Known-value field holding each Dirichlet face value on that face's wall nodes.

    The walls are the coast cells (dry, 4-adjacent to ``wet``) on the row
    just south / north of the wet bounding box and the column just west /
    east of it. For ψ on a ``from_interior`` rectangle these are X rows 0
    (south) and N-2 (north) and X columns 0 (west) and N-2 (east), the
    C-grid wall nodes. Faces are written in the BoundaryConditionSet order
    (south, north, west, east), so west / east win where a wall node is on
    two faces. Island coasts, which are not on a face, stay at zero.
    """
    ny, nx = wet.shape[-2], wet.shape[-1]
    # j[j, i] = j, i[j, i] = i  (broadcast to [Ny, Nx])
    j = jnp.arange(ny)[:, None]
    i = jnp.arange(nx)[None, :]
    wet_rows = jnp.any(wet, axis=1)  # [Ny]
    wet_cols = jnp.any(wet, axis=0)  # [Nx]
    # south wall row = first wet row - 1; north wall row = last wet row + 1
    j_south = jnp.argmax(wet_rows) - 1
    j_north = ny - jnp.argmax(wet_rows[::-1])
    # west wall column = first wet column - 1; east = last wet column + 1
    i_west = jnp.argmax(wet_cols) - 1
    i_east = nx - jnp.argmax(wet_cols[::-1])

    values = jnp.zeros((ny, nx), dtype=dtype)
    values = jnp.where(j == j_south, jnp.asarray(bc.south.value, dtype), values)
    values = jnp.where(j == j_north, jnp.asarray(bc.north.value, dtype), values)
    values = jnp.where(i == i_west, jnp.asarray(bc.west.value, dtype), values)
    values = jnp.where(i == i_east, jnp.asarray(bc.east.value, dtype), values)
    # Only the coast cells carry boundary values.
    return jnp.where(coast_ring(wet), values, jnp.zeros((), dtype))


def _check_method(method: str) -> None:
    """Raise the standard error for an unknown solver method."""
    if method not in _METHODS:
        raise ValueError(f"{_METHOD_ERROR}; got {method!r}")


def _known_value_domain(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    method: str,
    bc: str,
    mask: _MaskLike | None,
    known_mask: Array | None,
    staggering: str,
) -> SolveDomain:
    """The `gridax.solvers.SolveDomain` of a solve with known values.

    Mask-based methods use the caller's ``mask`` (required); the known
    values sit on its coast ring (the dry cells adjacent to the wet cells).
    The spectral path has no mask: it solves the ψ basin of a
    ``from_interior`` rectangle, wet X nodes ``[1:-2, 1:-2]``, whose coast
    is the wall nodes (rows / columns 0 and N-2). This is the problem
    ``method="cg"`` solves with the rectangle's ``Mask2D``, so both methods
    agree to the solver tolerance.
    """
    _check_method(method)
    if staggering == "T":
        # Every T-point Dirichlet wall here (spectral DST-II, cg with
        # bc="dst") is on the cell faces; the lifting pins values at cell
        # centres (the ghost cells, half a cell outside the faces), which
        # would be a different, first-order wall.
        raise ValueError(
            "known_values are not supported for the T-point pressure solve: "
            "its Dirichlet wall sits on the cell faces, while known values are "
            "pinned at cell centres (the ghost cells, half a cell outside)."
        )
    if method == "spectral":
        if bc != "dst":
            raise ValueError(
                "known_values prescribe Dirichlet data, so method='spectral' "
                f"requires bc='dst'; got bc={bc!r}.  Use method='cg' or "
                "'capacitance' with a mask for other boundary types."
            )
        if mask is not None:
            raise ValueError(
                "method='spectral' with known_values solves the rectangular "
                "basin only and does not accept a mask; use method='cg' or "
                "'capacitance' for a custom mask."
            )
        if known_mask is not None:
            raise ValueError(
                "method='spectral' does not support interior known values "
                "(known_mask); use method='cg'."
            )
        return SolveDomain(_x_basin(shape))

    if bc != "dst":
        raise ValueError(
            "known_values prescribe Dirichlet data on the coast, so they need "
            f"bc='dst'; got bc={bc!r}."
        )
    mask_arr = _resolve_mask_arr(mask, dtype, staggering)
    if mask_arr is None:
        raise ValueError(f"known_values with method={method!r} requires a mask")
    if method == "capacitance" and known_mask is not None:
        raise ValueError(
            "method='capacitance' does not support known_mask: the capacitance "
            "solver's unknowns are fixed by the mask it was built on.  Use "
            "method='cg'."
        )
    return SolveDomain(mask_arr, known_mask)


def _lift_rhs(
    domain: SolveDomain,
    rhs: Float[Array, "Ny Nx"],
    known_values: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
) -> tuple[Float[Array, "Ny Nx"], Float[Array, "Ny Nx"]]:
    """Lift the known values out of the right-hand side.

        g̃ = g on K,   f̃ = f - (∇² - λ) g̃   on the solve cells E

    K is ``domain.all_known`` (the coast ring and the ``known_mask`` cells)
    and E is ``domain.effective_mask``. The lift lives on dry coast cells,
    so the five-point operator is applied without a mask. ``lambda_`` may
    be traced, so this vmaps over layers with their own λ.

    Returns:
        ``(f̃, g̃)``: the corrected right-hand side, zero off E, and the lift.
    """
    shape = rhs.shape
    unmasked = jnp.ones(shape, dtype=rhs.dtype)

    def _matvec(v: Float[Array, " n"]) -> Float[Array, " n"]:
        return masked_laplacian(v.reshape(shape), unmasked, dx, dy, lambda_).ravel()

    operator = gaussx.as_linear_operator(
        _matvec, shape=(rhs.size, rhs.size), dtype=rhs.dtype
    )
    # g̃ = g on K, f - A g̃ off K
    lifted, value_lift = gaussx.lift_known_values(
        operator, rhs.ravel(), domain.all_known.ravel(), known_values.ravel()
    )
    # f̃[j, i] = f[j, i] - (∇² - λ) g̃ [j, i] on E, 0 elsewhere
    rhs_corrected = jnp.where(domain.effective_mask, lifted.reshape(shape), 0.0)
    return rhs_corrected, value_lift.reshape(shape)


def _reconstruct(
    domain: SolveDomain,
    psi_hom: Float[Array, "Ny Nx"],
    value_lift: Float[Array, "Ny Nx"],
) -> Float[Array, "Ny Nx"]:
    """ψ = g̃ + ψ_hom on E, so the known cells carry the known values."""
    return value_lift + jnp.where(domain.effective_mask, psi_hom, 0.0)


def _solve_with_known_values(
    rhs: Float[Array, "Ny Nx"],
    known_values: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Array,
    bc: str,
    method: str,
    mask: _MaskLike | None,
    known_mask: Array | None,
    capacitance_solver: CapacitanceSolver | None,
    multigrid_solver: MultigridSolver | None,
    preconditioner: _PrecondLike | None,
    staggering: str,
) -> Float[Array, "Ny Nx"]:
    """Inhomogeneous solve: lift, solve homogeneously, reconstruct.

    ψ = g̃ + ψ_hom, where ψ_hom solves (∇² - λ) ψ_hom = f - (∇² - λ) g̃ on
    the solve domain, with ψ_hom = 0 on the coast ring and the
    ``known_mask`` cells.
    """
    domain = _known_value_domain(
        rhs.shape, rhs.dtype, method, bc, mask, known_mask, staggering
    )
    rhs_corrected, value_lift = _lift_rhs(domain, rhs, known_values, dx, dy, lambda_)
    if method == "spectral":
        # The ψ basin's solve domain is the rectangle [1:-2, 1:-2] (DST-I).
        psi_hom = _solve_spectral(rhs_corrected, dx, dy, lambda_, "dst", "X")
    else:
        # CG solves on the effective domain directly.  A capacitance solver
        # built on the same wet mask has every wet cell as an unknown and
        # psi = 0 on the coast, which is exactly the effective domain when
        # known_mask is None (the only case it accepts).
        eff_mask = domain.effective_mask.astype(rhs.dtype)
        psi_hom = _solve_dispatch(
            rhs_corrected,
            dx,
            dy,
            lambda_,
            bc,
            method,
            eff_mask,
            capacitance_solver,
            multigrid_solver,
            preconditioner,
            staggering,
        )
    return _reconstruct(domain, psi_hom, value_lift)


# ---------------------------------------------------------------------------
# The inversions
# ---------------------------------------------------------------------------


def streamfunction_from_vorticity(
    zeta: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    *,
    bc: _BCLike = "dst",
    lambda_: float = 0.0,
    method: _Method = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    multigrid_solver: MultigridSolver | None = None,
    preconditioner: _PrecondLike | None = None,
    known_values: Float[Array, "Ny Nx"] | None = None,
    known_mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "Ny Nx"]:
    r"""Invert the vorticity-streamfunction relation for ψ at X points.

        (∇² - λ) ψ = ζ

    with ∇² the five-point Laplacian; λ = 0 is the Poisson problem, λ > 0
    the Helmholtz problem of a QG inversion. The methods:

    * ``"spectral"`` (default): a direct DST / DCT / FFT solve on the
      rectangle, chosen by ``bc``. The unknowns are sliced out of the full
      array so that the wall is the C-grid wall (``"dst"``: DST-I on
      ``[1:-2, 1:-2]``, ψ = 0 on the wall nodes 0 and N-2; ``"dct"``:
      DCT-I on ``[0:-1, 0:-1]``; ``"fft"``: ``[1:-1, 1:-1]``), the same
      wall ``method="cg"`` imposes with a ``Mask2D``. The other entries of
      the result are zero.
    * ``"cg"``: preconditioned conjugate gradients (gaussx) on a masked
      domain. Needs ``mask``; the preconditioner is the spectral solve of
      ``bc`` unless ``preconditioner`` is given.
    * ``"capacitance"``: the spectral solve with a capacitance-matrix
      correction for the coast. Needs a ``capacitance_solver`` from
      `gridax.solvers.build_capacitance_solver`.
    * ``"multigrid"``: CG preconditioned by one V-cycle of a multigrid
      hierarchy. Needs ``mask`` and a ``multigrid_solver`` from
      `gridax.solvers.build_multigrid_solver` built on the same mask and λ;
      it agrees with ``"cg"`` to the CG tolerance in fewer steps.

    Args:
        zeta: Relative vorticity ζ (the right-hand side), ``[Ny, Nx]``.
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        bc: Boundary condition on the wall / coast. ``"dst"`` (Dirichlet,
            ψ = 0 on the wall nodes) is the standard choice and the only one
            the mask-based methods accept for ψ; ``"dct"`` and ``"fft"`` are
            for ``method="spectral"`` only (the others raise rather than
            ignore them). With ``"dct"`` (∂ψ/∂n = 0 on the walls) the wall
            nodes are unknowns, so ``zeta`` is read there too (X rows /
            columns 0 and N-2): fill the wall values of ``zeta`` first. A
            `gridax.BoundaryConditionSet` is also accepted: its face types
            select the transform (all ``Dirichlet1D`` → ``"dst"``, all
            zero-valued ``Neumann1D`` → ``"dct"``, all ``Periodic1D`` →
            ``"fft"``; mixing raises), its ``mask`` is used by the
            mask-based methods, and non-zero ``Dirichlet1D`` values become
            ``known_values`` on each face's wall nodes (explicit
            ``known_values`` take precedence).
        lambda_: The Helmholtz parameter λ ≥ 0.
        method: ``"spectral"``, ``"cg"``, ``"capacitance"`` or
            ``"multigrid"``.
        mask: Domain mask, needed by ``"cg"`` and ``"multigrid"``. A
            `gridax.Mask2D` gives its ψ mask ``xy_corner_strict``.
        capacitance_solver: The pre-built capacitance solver of
            ``method="capacitance"``.
        multigrid_solver: The pre-built multigrid hierarchy of
            ``method="multigrid"``.
        preconditioner: A custom preconditioner r ↦ M⁻¹ r ≈ (∇² - λ)⁻¹ r
            for ``method="cg"``.
        known_values: Prescribed (inhomogeneous Dirichlet) values of ψ, used
            on the coast ring (the dry cells next to a wet cell, where
            ``method="cg"`` puts its ψ = 0 wall) and at the ``known_mask``
            cells; values elsewhere are ignored. The solution equals them
            exactly there and every other wet cell is solved for. The
            mask-based methods need ``mask``; for ``"capacitance"`` build
            the solver on the same wet mask. ``method="spectral"`` needs
            ``bc="dst"`` and no mask: it solves the ψ basin of the rectangle
            (wet X nodes ``[1:-2, 1:-2]``, values on the wall nodes 0 and
            N-2), matching ``method="cg"`` with the rectangle's ``Mask2D``.
        known_mask: Extra interior wet cells with known values (sparse
            observations, say), pinned with the coast ring. ``"cg"`` and
            ``"multigrid"`` only.

    Returns:
        The streamfunction ψ, ``[Ny, Nx]``.

    Raises:
        ValueError: For an unknown ``method`` or ``bc``, a missing ``mask``
            or pre-built solver, or a combination the method cannot honour.

    Examples:
        ψ = sin(πx) sin(πy) on the unit square has ∇²ψ = -2π² ψ:

        >>> import jax.numpy as jnp
        >>> from gridax import CartesianGrid2D
        >>> from sesmax_core.gfd import streamfunction_from_vorticity
        >>> grid = CartesianGrid2D.from_interior(32, 32, 1.0, 1.0)
        >>> # X node (j, i) at (j dy, i dx): the walls are the nodes 0 and N-2
        >>> x = jnp.arange(grid.Nx) * grid.dx
        >>> y = jnp.arange(grid.Ny) * grid.dy
        >>> exact = jnp.sin(jnp.pi * y)[:, None] * jnp.sin(jnp.pi * x)[None, :]
        >>> psi = streamfunction_from_vorticity(
        ...     -2 * jnp.pi**2 * exact, grid.dx, grid.dy
        ... )
        >>> error = jnp.max(jnp.abs(psi - exact)[1:-2, 1:-2])
        >>> bool(error < 2e-3)
        True
    """
    bc, mask, known_values = _resolve_bc(
        bc, method, mask, known_values, zeta.shape, zeta.dtype, "X"
    )
    if known_values is not None:
        return _solve_with_known_values(
            zeta,
            known_values,
            dx,
            dy,
            lambda_,
            bc,
            method,
            mask,
            known_mask,
            capacitance_solver,
            multigrid_solver,
            preconditioner,
            "X",
        )
    return _solve_dispatch(
        zeta,
        dx,
        dy,
        lambda_,
        bc,
        method,
        mask,
        capacitance_solver,
        multigrid_solver,
        preconditioner,
        "X",
    )


def pressure_from_divergence(
    div_u: Float[Array, "Ny Nx"],
    dx: float,
    dy: float,
    *,
    bc: _BCLike = "dct",
    method: _Method = "spectral",
    mask: _MaskLike | None = None,
    multigrid_solver: MultigridSolver | None = None,
    preconditioner: _PrecondLike | None = None,
) -> Float[Array, "Ny Nx"]:
    r"""Solve for the pressure p at T points of a pressure projection.

        ∇² p = ∇ · u

    the step of a pressure-projection (Chorin) scheme that removes the
    divergence of a provisional velocity. Every method puts the wall on the
    cell faces: the spectral path solves on the T interior ``[1:-1, 1:-1]``
    (``"dct"``: DCT-II, ∂p/∂n = 0; ``"dst"``: DST-II, p = 0; ``"fft"``:
    periodic) and returns a zero ghost ring; ``method="cg"`` solves on a
    mask with the same face conditions on every wet-dry face. The
    capacitance method and known values pin values at cell centres, not on
    faces, so they are not offered here.

    Args:
        div_u: The divergence ∇ · u (the right-hand side), ``[Ny, Nx]``.
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        bc: Boundary condition on the wall / coast. ``"dct"`` (Neumann,
            ∂p/∂n = 0 on the faces) is the standard choice with solid walls;
            with ``method="cg"`` it is zero flux through every wet-dry face,
            and the solution is the zero-mean one on each connected wet
            region (the problem fixes p only up to a constant per region).
            ``"dst"`` is p = 0 on the faces, for ``"spectral"`` and ``"cg"``.
            ``"fft"`` (periodic) is ``method="spectral"`` only. A
            `gridax.BoundaryConditionSet` is accepted as in
            `streamfunction_from_vorticity`, except that non-zero
            ``Dirichlet1D`` faces raise.
        method: ``"spectral"``, ``"cg"`` or ``"multigrid"``.
            ``"capacitance"`` raises: its solver holds the unknown at zero
            on the dry cell centres (the ψ wall), not on the faces.
        mask: Domain mask, needed by ``"cg"`` and ``"multigrid"``. A
            `gridax.Mask2D` gives its T-point mask ``h``.
        multigrid_solver: The pre-built hierarchy of ``method="multigrid"``,
            ``build_multigrid_solver(mask.h, dx, dy, bc="neumann")`` for
            ``bc="dct"``.
        preconditioner: A custom preconditioner for ``method="cg"``.

    Returns:
        The pressure p, ``[Ny, Nx]``.

    Raises:
        ValueError: For an unknown or unavailable ``method`` or ``bc``, or a
            missing ``mask`` or pre-built solver.

    Examples:
        p = cos(πx) cos(πy) on the unit square satisfies ∂p/∂n = 0 and
        ∇²p = -2π² p:

        >>> import jax.numpy as jnp
        >>> from gridax import CartesianGrid2D
        >>> from sesmax_core.gfd import pressure_from_divergence
        >>> grid = CartesianGrid2D.from_interior(32, 32, 1.0, 1.0)
        >>> # T cell (j, i) at ((j - 1/2) dy, (i - 1/2) dx)
        >>> x = (jnp.arange(grid.Nx) - 0.5) * grid.dx
        >>> y = (jnp.arange(grid.Ny) - 0.5) * grid.dy
        >>> exact = jnp.cos(jnp.pi * y)[:, None] * jnp.cos(jnp.pi * x)[None, :]
        >>> p = pressure_from_divergence(-2 * jnp.pi**2 * exact, grid.dx, grid.dy)
        >>> error = jnp.max(jnp.abs(p - exact)[1:-1, 1:-1])
        >>> bool(error < 2e-3)
        True
    """
    if method == "capacitance":
        raise ValueError(
            "method='capacitance' is not available for the T-point pressure "
            "solve: the capacitance solver holds the unknown at zero on the dry "
            "cell centres next to the wet ones (the psi wall), not on the cell "
            "faces where the pressure wall is.  Use method='cg' with a mask."
        )
    # a non-zero Dirichlet set raises for T points, so no known values return
    bc, mask, _ = _resolve_bc(bc, method, mask, None, div_u.shape, div_u.dtype, "T")
    return _solve_dispatch(
        div_u,
        dx,
        dy,
        0.0,
        bc,
        method,
        mask,
        None,
        multigrid_solver,
        preconditioner,
        "T",
    )


def pv_inversion(
    pv: Float[Array, "... Ny Nx"],
    dx: float,
    dy: float,
    lambda_: float | Float[Array, " nl"],
    *,
    bc: _BCLike = "dst",
    method: _Method = "spectral",
    mask: _MaskLike | None = None,
    capacitance_solver: CapacitanceSolver | None = None,
    multigrid_solver: MultigridSolver | None = None,
    preconditioner: _PrecondLike | None = None,
    known_values: Float[Array, "... Ny Nx"] | None = None,
    known_mask: Bool[Array, "Ny Nx"] | None = None,
) -> Float[Array, "... Ny Nx"]:
    r"""QG potential-vorticity inversion for ψ_k at X points, layer by layer.

        (∇² - λ_k) ψ_k = q_k

    with λ_k = 1 / R_{d,k}² the deformation parameter of layer (or vertical
    mode) k. Leading axes of ``pv`` are batched. With an array ``lambda_``
    of shape ``(nl,)`` the axis ``-3`` of ``pv`` is the layer axis and each
    layer gets its own λ_k (the 1 / R_d² of each vertical mode from
    `sesmax_core.vertical.decompose_vertical_modes`, say). The methods are
    those of `streamfunction_from_vorticity`; ``"capacitance"`` and
    ``"multigrid"`` need a scalar ``lambda_``, since their pre-built
    solvers are built for one λ.

    Args:
        pv: The potential vorticity q, ``[..., Ny, Nx]``.
        dx: Grid spacing in x.
        dy: Grid spacing in y.
        lambda_: λ, a scalar for every layer or one per layer, ``(nl,)``.
        bc: Boundary condition on the wall / coast, as in
            `streamfunction_from_vorticity` (``"dst"`` for the mask-based
            methods); a `gridax.BoundaryConditionSet` is accepted as there.
        method: ``"spectral"``, ``"cg"``, ``"capacitance"`` or
            ``"multigrid"``.
        mask: Domain mask, needed by ``"cg"`` and ``"multigrid"``.
        capacitance_solver: The pre-built solver of ``method="capacitance"``.
        multigrid_solver: The pre-built hierarchy of ``method="multigrid"``.
        preconditioner: A custom preconditioner for ``method="cg"``.
        known_values: Prescribed Dirichlet values (see
            `streamfunction_from_vorticity`), broadcast against ``pv``: an
            ``[Ny, Nx]`` field applies to every layer, ``[nl, Ny, Nx]``
            gives one per layer. Each layer's lift uses its own λ_k.
        known_mask: Extra interior wet cells with known values, shared by
            all layers. ``"cg"`` and ``"multigrid"`` only.

    Returns:
        The streamfunction ψ, the shape of ``pv``.

    Raises:
        ValueError: For an unknown ``method`` or ``bc``, a ``pv`` whose
            layer axis does not match an array ``lambda_``, or a method that
            needs a scalar ``lambda_``.

    Examples:
        Two layers with their own deformation parameter, one mode each:

        >>> import jax.numpy as jnp
        >>> from gridax import CartesianGrid2D
        >>> from sesmax_core.gfd import pv_inversion
        >>> grid = CartesianGrid2D.from_interior(32, 32, 1.0, 1.0)
        >>> # X node (j, i) at (j dy, i dx)
        >>> x = jnp.arange(grid.Nx) * grid.dx
        >>> y = jnp.arange(grid.Ny) * grid.dy
        >>> mode = jnp.sin(jnp.pi * y)[:, None] * jnp.sin(jnp.pi * x)[None, :]
        >>> lam = jnp.array([0.0, 50.0])
        >>> # q_k = (∇² - λ_k) ψ_k ≈ -(2π² + λ_k) ψ_k for ψ_k = mode
        >>> q = -(2 * jnp.pi**2 + lam)[:, None, None] * mode
        >>> psi = pv_inversion(q, grid.dx, grid.dy, lam)
        >>> psi.shape
        (2, 34, 34)
        >>> bool(jnp.max(jnp.abs(psi - mode)[:, 1:-2, 1:-2]) < 2e-3)
        True
    """
    bc, mask, known_values = _resolve_bc(
        bc, method, mask, known_values, pv.shape, pv.dtype, "X"
    )
    lam = jnp.asarray(lambda_)
    # A Python-number lambda stays concrete (the singular-Neumann projection
    # keys on a concrete zero); a traced scalar is passed through as is, so
    # jit / vmap over a scalar lambda_ work.
    lam_scalar = lambda_ if isinstance(lambda_, numbers.Real) else lam
    kv = None if known_values is None else jnp.broadcast_to(known_values, pv.shape)

    if lam.ndim == 0 and kv is not None:

        def _solve_one_known(
            rhs: Float[Array, "Ny Nx"], kv_i: Float[Array, "Ny Nx"]
        ) -> Float[Array, "Ny Nx"]:
            return _solve_with_known_values(
                rhs,
                kv_i,
                dx,
                dy,
                lam_scalar,
                bc,
                method,
                mask,
                known_mask,
                capacitance_solver,
                multigrid_solver,
                preconditioner,
                "X",
            )

        if pv.ndim == 2:
            return _solve_one_known(pv, kv)
        shape = pv.shape
        flat = pv.reshape(-1, shape[-2], shape[-1])
        kv_flat = kv.reshape(-1, shape[-2], shape[-1])
        return eqx.filter_vmap(_solve_one_known)(flat, kv_flat).reshape(shape)

    if lam.ndim == 0:

        def _solve_one(rhs: Float[Array, "Ny Nx"]) -> Float[Array, "Ny Nx"]:
            return _solve_dispatch(
                rhs,
                dx,
                dy,
                lam_scalar,
                bc,
                method,
                mask,
                capacitance_solver,
                multigrid_solver,
                preconditioner,
                "X",
            )

        # Scalar lambda: vmap over all leading dims if present
        if pv.ndim == 2:
            return _solve_one(pv)
        shape = pv.shape
        flat = pv.reshape(-1, shape[-2], shape[-1])
        return eqx.filter_vmap(_solve_one)(flat).reshape(shape)

    # Array lambda: leading dim must match lam.shape[0]
    if pv.ndim < 3:
        raise ValueError(
            f"pv must have at least 3 dims when lambda_ is an array, "
            f"got shape {pv.shape}"
        )
    nl = lam.shape[0]
    if pv.shape[-3] != nl:
        raise ValueError(
            f"pv.shape[-3]={pv.shape[-3]} does not match lambda_ length {nl}"
        )

    # Solve each layer with its own lambda.  lam_i is a JAX tracer inside
    # vmap, so the per-layer solves below call the tracer-safe building
    # blocks (the masked CG, or _solve_spectral, which has no Python branch
    # on lambda) with the same X staggering as _solve_dispatch.
    if method in ("capacitance", "multigrid"):
        raise ValueError(
            f"method={method!r} does not support array-valued lambda_: its "
            "pre-built solver holds one lambda.  Solve each layer separately or "
            "use method='spectral' or 'cg' for multi-layer problems."
        )

    # Known values: shared solve domain; each layer lifts with its own lambda.
    domain = (
        None
        if kv is None
        else _known_value_domain(pv.shape, pv.dtype, method, bc, mask, known_mask, "X")
    )

    if method == "cg":
        _check_cg_bc(bc, "X")
        mask_arr = _resolve_mask_arr(mask, pv.dtype, "X")
        if mask_arr is None:
            raise ValueError("method='cg' requires a mask")
        # With known values the solve runs on the effective domain.
        solve_mask = (
            mask_arr if domain is None else domain.effective_mask.astype(pv.dtype)
        )

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            return _cg_masked_solve(
                rhs, solve_mask, dx, dy, lam_i, bc, preconditioner, "X"
            )

    elif method == "spectral":
        # With known values the ψ basin's solve domain is the DST-I rectangle.
        layer_bc = bc if domain is None else "dst"
        if ("X", layer_bc) not in _SPECTRAL_STAGGERED:
            raise ValueError(f"bc must be 'fft', 'dst', or 'dct'; got {bc!r}")

        def _solve_layer(
            rhs: Float[Array, "Ny Nx"], lam_i: float
        ) -> Float[Array, "Ny Nx"]:
            return _solve_spectral(rhs, dx, dy, lam_i, layer_bc, "X")

    else:
        raise ValueError(f"{_METHOD_ERROR}; got {method!r}")

    # Flatten any leading batch dims: (..., nl, Ny, Nx) -> (batch, nl, Ny, Nx)
    shape = pv.shape
    ny, nx = shape[-2], shape[-1]
    pv_4d = pv.reshape(-1, nl, ny, nx)

    if domain is None or kv is None:
        # vmap over layer axis (pairing each layer with its lambda)
        _solve_layers = eqx.filter_vmap(_solve_layer, in_axes=(0, 0))

        # vmap over the (flattened) batch axis
        out_4d = eqx.filter_vmap(lambda batch: _solve_layers(batch, lam))(pv_4d)
        return out_4d.reshape(shape)

    known_domain = domain

    def _solve_layer_known(
        rhs: Float[Array, "Ny Nx"], kv_i: Float[Array, "Ny Nx"], lam_i: float
    ) -> Float[Array, "Ny Nx"]:
        # psi = lift + psi_hom, psi_hom solving the lambda_i-corrected RHS
        rhs_c, value_lift = _lift_rhs(known_domain, rhs, kv_i, dx, dy, lam_i)
        return _reconstruct(known_domain, _solve_layer(rhs_c, lam_i), value_lift)

    kv_4d = jnp.reshape(kv, (-1, nl, ny, nx))
    _solve_layers_known = eqx.filter_vmap(_solve_layer_known, in_axes=(0, 0, 0))
    out_4d = eqx.filter_vmap(lambda batch, k: _solve_layers_known(batch, k, lam))(
        pv_4d, kv_4d
    )
    return out_4d.reshape(shape)
