"""Reference-free evaluation metrics for ocean model states.

These are *field-level diagnostics* computed on a model's own Cartesian
grid — distinct from ``model.diagnose()`` (per-model conserved invariants)
and from the per-field min/mean/max stats the runner already logs. They
give a run a quantitative evaluation surface for free-running simulations:
how divergent the flow is, how much enstrophy / kinetic energy it carries,
and how far it sits from geostrophic balance.

All metrics are **reference-free** — they need only the state itself, not a
ground-truth field. Reference-based skill scores (RMSE, PSD score, …) need
observations / a truth trajectory and belong with the data-assimilation
work in a later phase.

Design
------
- The generic metrics (:func:`rms_divergence`, :func:`total_enstrophy`,
  :func:`kinetic_energy`) take raw velocity fields plus the
  ``gridax.GridOps`` / grid the model already holds, so they apply to any
  Arakawa C-grid fluid model (SWM, QG, Navier-Stokes).
- :func:`geostrophic_imbalance` is model-aware: it reuses the model's *own*
  pressure-gradient and Coriolis operators, so it measures departure from
  exactly the balance the model is built around.
- :func:`compute_eval_metrics` is a defensive dispatcher: it returns the
  applicable metrics for fluid models and an empty dict for everything else
  (Lorenz, diffusion, …), so the runner can call it unconditionally.

Reductions use the grid interior (dropping the one-cell ghost halo) to match
the convention in the models' ``diagnose`` methods and avoid contaminating
the metric with boundary / ghost-point artifacts.

Moved from somax 9cb19b3: ``somax/_src/eval/metrics.py``.
"""

from __future__ import annotations

import contextlib
from typing import Any

import jax.numpy as jnp
import paramax
from gridax import CartesianCGrid2D, CField, GridOps, SphericalCGrid2D
from jaxtyping import Array, Float

from sesmax_core import State
from sesmax_core.fields import each, placed, values


# One-cell ghost halo dropped from every reduction (matches ``diagnose``).
_INTERIOR = (slice(1, -1), slice(1, -1))


def _interior(field: Float[Array, "Ny Nx"] | CField, *, interior: bool) -> Array:
    """Optionally trim the one-cell ghost halo before reducing."""
    arr = values(field)
    return arr[_INTERIOR] if interior else arr


def rms_divergence(
    u: Float[Array, "Ny Nx"] | CField,
    v: Float[Array, "Ny Nx"] | CField,
    ops: GridOps,
    *,
    interior: bool = True,
) -> Float[Array, ""]:
    r"""Root-mean-square horizontal divergence ``sqrt(<(∇·u)²>)``.

    A diagnostic of how non-divergent the flow is. For incompressible /
    geostrophic flow it should stay near zero; a growing value flags
    spurious compressibility or numerical noise.

    Args:
        u: x-velocity field on its C-grid points, shape ``(Ny, Nx)``.
        v: y-velocity field on its C-grid points, shape ``(Ny, Nx)``.
        ops: The model's ``gridax.GridOps``; its :meth:`divergence` lowers
            ``(u, v)`` to ``∇·u`` at tracer points.
        interior: Drop the one-cell ghost halo before reducing (default).

    Returns:
        Scalar RMS divergence (same units as ``∇·u``, i.e. 1/s for m/s
        velocities on a metre grid).
    """
    div = _interior(ops.divergence(u, v), interior=interior)
    return jnp.sqrt(jnp.mean(div**2))


def total_enstrophy(
    u: Float[Array, "Ny Nx"] | CField,
    v: Float[Array, "Ny Nx"] | CField,
    ops: GridOps,
    grid: CartesianCGrid2D,
    *,
    interior: bool = True,
) -> Float[Array, ""]:
    r"""Domain-integrated enstrophy ``0.5 ∫ ζ² dA`` with ``ζ = ∂v/∂x - ∂u/∂y``.

    Enstrophy is a robust health metric for 2D / quasi-2D turbulence: in the
    inviscid limit it is bounded, so runaway growth signals instability.

    Args:
        u: x-velocity field, shape ``(Ny, Nx)``.
        v: y-velocity field, shape ``(Ny, Nx)``.
        ops: The model's ``gridax.GridOps``; its :meth:`curl` returns ζ.
        grid: The model's ``CartesianCGrid2D`` (for the ``dx·dy`` cell area).
        interior: Drop the one-cell ghost halo before reducing (default).

    Returns:
        Scalar total enstrophy.
    """
    zeta = _interior(ops.curl(u, v), interior=interior)
    cell_area = grid.spacing("x") * grid.spacing("y")
    return 0.5 * jnp.sum(zeta**2) * cell_area


def kinetic_energy(
    u: Float[Array, "Ny Nx"] | CField,
    v: Float[Array, "Ny Nx"] | CField,
    grid: CartesianCGrid2D,
    *,
    interior: bool = True,
) -> Float[Array, ""]:
    r"""Domain-integrated kinetic energy ``0.5 ∫ (u² + v²) dA``.

    Mirrors the velocity term of the models' ``diagnose`` energy (summing the
    staggered components directly), so it tracks consistently alongside the
    model's own energy diagnostic.

    Args:
        u: x-velocity field, shape ``(Ny, Nx)``.
        v: y-velocity field, shape ``(Ny, Nx)``.
        grid: The model's ``CartesianCGrid2D`` (for the ``dx·dy`` cell area).
        interior: Drop the one-cell ghost halo before reducing (default).

    Returns:
        Scalar kinetic energy (per unit depth).
    """
    ui = _interior(u, interior=interior)
    vi = _interior(v, interior=interior)
    cell_area = grid.spacing("x") * grid.spacing("y")
    return 0.5 * jnp.sum(ui**2 + vi**2) * cell_area


def geostrophic_imbalance(
    model: Any,
    state: State,
    *,
    interior: bool = True,
    eps: float = 1e-30,
) -> Float[Array, ""]:
    r"""Dimensionless ageostrophic fraction of a shallow-water-type state.

    Geostrophic balance is ``f x u = -g ∇η`` — equivalently, the
    pressure-gradient and Coriolis accelerations cancel. This metric forms
    that residual using the model's *own* operators::

        r_u = -g ∂η/∂x + f·v        (at U-points)
        r_v = -g ∂η/∂y - f·u        (at V-points)

    and returns ``rms(r) / rms(coriolis)`` — 0 for a perfectly geostrophic
    flow, O(1) when ageostrophic accelerations rival the Coriolis term.
    Reusing ``model.ops`` keeps the residual exactly
    consistent with the balance the model integrates around (staggering,
    masks and the β-plane ``f`` field all included).

    Args:
        model: A shallow-water-type model exposing ``ops``, ``f_field`` and
            ``consts.gravity``.
        state: A state with ``h`` (height perturbation η), ``u`` and ``v``.
        interior: Drop the one-cell ghost halo before reducing (default).
        eps: Floor added to the denominator to keep a motionless state
            (zero Coriolis term) finite.

    Returns:
        Scalar ageostrophic fraction in ``[0, ∞)``.
    """
    g = model.consts.gravity
    h, u, v = state.h, state.u, state.v

    # Leading-order momentum balance terms, on their native C-grid points.
    dh_dx = model.ops.difference(h, axis="x", src="T", dst="U")
    dh_dy = model.ops.difference(h, axis="y", src="T", dst="V")
    cor_u, cor_v = model.ops.coriolis(u, v, f=placed(model.f_field, u, "T"))

    def residual_of(gradient: Array, coriolis: Array) -> Array:
        return -g * gradient + coriolis

    res_u = _interior(each(residual_of, dh_dx, cor_u), interior=interior)
    res_v = _interior(each(residual_of, dh_dy, cor_v), interior=interior)
    cor_u_i = _interior(cor_u, interior=interior)
    cor_v_i = _interior(cor_v, interior=interior)

    residual = jnp.sqrt(jnp.mean(res_u**2) + jnp.mean(res_v**2))
    scale = jnp.sqrt(jnp.mean(cor_u_i**2) + jnp.mean(cor_v_i**2))
    return residual / (scale + eps)


def qg_balance_residual(
    model: Any,
    state: State,
    *,
    interior: bool = True,
    eps: float = 1e-30,
) -> Float[Array, ""]:
    r"""Dimensionless PV-inversion residual for a **barotropic** QG model.

    The QG balance analog of :func:`geostrophic_imbalance` for vorticity /
    streamfunction models: instead of a velocity-divergence residual (which QG
    models do not define), it measures how well the state's PV closes its own
    inversion. For barotropic QG the PV *is* the relative vorticity,
    ``q = nabla^2 psi``, so with ``psi = L^{-1} q`` and
    ``q_hat = laplacian(psi)`` it returns

    .. math::

        \frac{\lVert \hat q - q \rVert}{\lVert q \rVert + \epsilon},

    reduced over the grid interior. For a freshly inverted state this is
    ~machine-eps; a large value flags a state inconsistent with the model's
    elliptic operator.

    Restricted to barotropic QG (a 2-D PV field). Baroclinic / reparameterized
    QG invert a *modal Helmholtz* operator ``q = nabla^2 psi - f0^2 A psi``;
    the bare Laplacian omits the stretching term, so this residual would be
    O(1) even for a perfectly balanced state. Reconstructing the full
    stretching operator here would duplicate model internals, so the check is
    intentionally scoped to the barotropic case (use :meth:`model.diagnose`
    invariants for the layered models). Returns ``0.0`` for a trivially zero
    PV field.

    Args:
        model: The constructed barotropic QG model.
        state: A state carrying a 2-D PV field ``q``.
        interior: Reduce over the inversion's unknowns only (the interior,
            less the X-point wall row / column for ``"dst"`` / ``"dct"``).
            Defaults True.
        eps: Small constant guarding the normalisation.

    Returns:
        Scalar dimensionless residual.
    """
    q = values(state.q)
    psi = model._invert_pv(q)
    q_hat = model.ops.laplacian(psi, at="T")
    trim = pv_unknowns(model) if interior else (...,)
    resid = (q_hat - q)[trim]
    ref = q[trim]
    return jnp.linalg.norm(resid) / (jnp.linalg.norm(ref) + eps)


def pv_unknowns(model: Any) -> tuple[slice, slice]:
    """The cells the inversion of a barotropic QG model solves for.

    The spectral inversions put psi on X points: ``"dst"`` / ``"dct"`` solve on
    ``[1:-2, 1:-2]`` with the wall on node ``N-2``. ``"fft"``, and the iterative
    one on the sphere, solve on ``[1:-1, 1:-1]``. Only there does
    ``laplacian(invert(q)) == q`` hold.
    """
    if getattr(model, "poisson_bc", None) in ("dst", "dct"):
        return (slice(1, -2), slice(1, -2))
    return _INTERIOR


def _is_fluid_state(state: State) -> bool:
    """A 2D C-grid velocity state we can compute field metrics for."""
    if not (hasattr(state, "u") and hasattr(state, "v")):
        return False
    return state.u.ndim == 2 and state.v.ndim == 2


def _supports_qg_balance(model: Any, state: State) -> bool:
    """Whether ``qg_balance_residual`` applies — barotropic QG only.

    A 2-D PV field signals the barotropic ``q = nabla^2 psi`` relation the
    residual assumes; baroclinic / reparameterized QG (3-D PV, modal-Helmholtz
    inversion with a stretching term) are excluded.
    """
    q = getattr(state, "q", None)
    return q is not None and q.ndim == 2 and hasattr(model, "_invert_pv")


def _supports_geostrophy(model: Any, state: State) -> bool:
    """Whether the model exposes the pieces ``geostrophic_imbalance`` needs."""
    consts = getattr(model, "consts", None)
    return (
        hasattr(state, "h")
        and hasattr(model, "f_field")
        and consts is not None
        and hasattr(consts, "gravity")
    )


def compute_eval_metrics(model: Any, state: State) -> dict[str, float]:
    """Compute every applicable reference-free metric for ``(model, state)``.

    A defensive dispatcher meant to be called unconditionally by the runner:
    it inspects the model / state and returns only the metrics that make
    sense. Non-fluid models (Lorenz, diffusion, …) and multilayer (3D) states
    yield an empty dict rather than an error.

    Scope: this targets **Cartesian velocity-state Arakawa C-grid models** —
    those whose state carries 2D ``u`` / ``v`` (SWM, Burgers). Spherical
    models get their ``invariant_*`` entries, which their own ``diagnose``
    area-weights correctly, but not the field metrics, which assume a
    uniform ``dx * dy`` cell area. **Vorticity / streamfunction
    models** (``barotropic_qg``, the vorticity Navier-Stokes) are intentionally
    *not* covered: they evolve ``q`` / ``omega`` and never define a discrete
    velocity divergence (non-divergence is only an analytic property, so a
    divergence metric would be operator-dependent with no canonical zero —
    misleading rather than diagnostic). Those models already report
    ``kinetic_energy`` and ``enstrophy`` through their own ``diagnose`` output,
    so they are not metric-less.

    Args:
        model: A constructed model.
        state: The state to evaluate (typically the final integrated state).

    Returns:
        Flat ``{metric_name: float}`` dict. For velocity-state models a subset
        of ``rms_divergence`` / ``total_enstrophy`` / ``kinetic_energy`` /
        ``geostrophic_imbalance``; for QG (vorticity/streamfunction) models a
        ``qg_balance_residual``. Plus any conserved quantities the model's
        ``diagnose(state).invariants()`` advertises, prefixed ``invariant_``.
        Empty when no metric applies.
    """
    out: dict[str, float] = {}

    # Conserved-invariant snapshot — works for any model that advertises them
    # via diagnose().invariants() (SWM mass/energy/enstrophy/Casimir, QG
    # energy/enstrophy), independent of the velocity-state field metrics below.
    try:
        invariants = paramax.unwrap(model).diagnose(state).invariants()
    except Exception:
        invariants = {}
    for name, value in invariants.items():
        with contextlib.suppress(TypeError, ValueError):
            out[f"invariant_{name}"] = float(jnp.asarray(value))

    if not (hasattr(model, "ops") and hasattr(model, "grid")):
        return out

    # Spherical models stop here. The velocity-state metrics below use
    # one ``dx * dy`` as a uniform cell area, which a spherical
    # grid does not have — its area is latitude-dependent — so they
    # would raise, and the runner would swallow that as a warning and
    # silently report no metrics at all. The invariants above are
    # already area-weighted correctly by the model's own ``diagnose``,
    # so a spherical run still gets the numbers that mean something.
    if isinstance(model.grid, SphericalCGrid2D):
        return out

    # QG (vorticity/streamfunction) models: a PV-inversion balance residual
    # stands in for the velocity-divergence metric they cannot define.
    if _supports_qg_balance(model, state):
        out["qg_balance_residual"] = float(qg_balance_residual(model, state))
        return out

    if not _is_fluid_state(state):
        return out

    u, v, ops, grid = state.u, state.v, model.ops, model.grid
    out["rms_divergence"] = float(rms_divergence(u, v, ops))
    out["total_enstrophy"] = float(total_enstrophy(u, v, ops, grid))
    out["kinetic_energy"] = float(kinetic_energy(u, v, grid))
    if _supports_geostrophy(model, state):
        out["geostrophic_imbalance"] = float(geostrophic_imbalance(model, state))
    return out
