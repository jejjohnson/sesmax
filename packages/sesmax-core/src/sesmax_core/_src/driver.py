"""The function-level PDE driver: diffeqsolve on a right-hand side, with corrections.

`solve_pde` integrates dy/dt = P(f(t, B(y))) for a right-hand side f, a
state correction B (fill ghost cells from the interior) and a tendency
correction P (zero the tendency on walls or dry cells), both applied inside
every Runge-Kutta stage. `Model.integrate` is its object-level form. Moved
from gridax-jax's ``solve_ocean_pde`` (``timestepping/_solve.py`` at gridax
``c69768c``), whose ``bc_fn`` is ``tendency_bc`` here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
from jaxtyping import PyTree


__all__ = ["solve_pde"]


class _CorrectedVectorField(eqx.Module):
    """P(f(t, B(y), args)) as a pytree, so transforms see the leaves of f.

    A closure would hide the leaves of ``rhs`` (a model's parameters, say)
    from ``eqx.filter_grad``.
    """

    rhs: Callable[[Any, PyTree, PyTree], PyTree]
    state_bc: Callable[[PyTree], PyTree] | None
    tendency_bc: Callable[[PyTree], PyTree] | None

    def __call__(self, t: Any, y: PyTree, args: PyTree) -> PyTree:
        if self.state_bc is not None:
            # y ← B(y)
            y = self.state_bc(y)
        # dy/dt = f(t, y)
        tendency = self.rhs(t, y, args)
        if self.tendency_bc is not None:
            # dy/dt ← P(dy/dt)
            tendency = self.tendency_bc(tendency)
        return tendency


def _fixed_step_count(t0: Any, t1: Any, dt0: Any) -> int | None:
    """⌈(t₁ - t₀) / Δt₀⌉ for concrete times, or ``None`` when they are traced.

    Args:
        t0: Start time.
        t1: End time.
        dt0: The fixed step.

    Returns:
        The number of fixed steps, at least 0, or ``None``.
    """
    try:
        steps = math.ceil((float(t1) - float(t0)) / float(dt0))
    except (TypeError, jax.errors.ConcretizationTypeError):
        return None
    return max(steps, 0)


def solve_pde(
    rhs_fn: Callable[[Any, PyTree, PyTree], PyTree],
    solver: dfx.AbstractSolver,
    y0: PyTree,
    t0: float,
    t1: float,
    dt0: float,
    *,
    saveat: dfx.SaveAt | None = None,
    state_bc: Callable[[PyTree], PyTree] | None = None,
    tendency_bc: Callable[[PyTree], PyTree] | None = None,
    args: PyTree = None,
    **kwargs: Any,
) -> dfx.Solution:
    """Integrate dy/dt = P(f(t, B(y))) with `diffrax.diffeqsolve`.

        dy/dt = P(f(t, B(y), args)),   y(t₀) = y₀

    Both corrections act at every stage evaluation, not once per step. For
    a fixed-step run (no ``stepsize_controller``) with concrete times and
    no ``max_steps``, ``max_steps`` defaults to ⌈(t₁ - t₀) / Δt₀⌉ + 16,
    because diffrax's own default of 4,096 steps would stop a long run
    early; otherwise diffrax's default applies.

    Args:
        rhs_fn: The right-hand side f(t, y, args) ↦ dy/dt.
        solver: The time-stepping scheme (``integrix.march.RK3SSP()``, …).
        y0: The initial state y₀, any pytree.
        t0: The start time t₀.
        t1: The end time t₁.
        dt0: The fixed (or first) step Δt₀.
        saveat: What to save; ``None`` saves the final state only.
        state_bc: The state correction B, y ↦ B(y), applied before each
            evaluation of f (fill ghost cells from the interior).
        tendency_bc: The tendency correction P, dy/dt ↦ P(dy/dt), applied
            after each evaluation (zero the tendency on walls).
        args: Passed to ``rhs_fn``.
        **kwargs: Forwarded to `diffrax.diffeqsolve` (``stepsize_controller``,
            ``adjoint``, ``max_steps``, ``throw``, ``event``, …).

    Returns:
        The diffrax solution.

    Examples:
        Exponential decay over 10,000 fixed steps, more than diffrax's
        default step limit:

        >>> import math
        >>> import jax.numpy as jnp
        >>> from integrix.march import RK3SSP
        >>> from sesmax_core import solve_pde
        >>> solution = solve_pde(
        ...     lambda t, y, args: -y, RK3SSP(), jnp.array(1.0), 0.0, 10.0, 1e-3
        ... )
        >>> bool(jnp.allclose(solution.ys[0], math.exp(-10.0), rtol=1e-4))
        True

        A tendency correction that zeroes the first entry holds it fixed:

        >>> solution = solve_pde(
        ...     lambda t, y, args: -y,
        ...     RK3SSP(),
        ...     jnp.ones(3),
        ...     0.0,
        ...     1.0,
        ...     0.1,
        ...     tendency_bc=lambda dydt: dydt.at[0].set(0.0),
        ... )
        >>> [round(v, 3) for v in solution.ys[0].tolist()]
        [1.0, 0.368, 0.368]
    """
    vector_field = _CorrectedVectorField(rhs_fn, state_bc, tendency_bc)
    if "max_steps" not in kwargs and "stepsize_controller" not in kwargs:
        # fixed steps: exactly ⌈(t₁ - t₀) / Δt₀⌉ of them
        steps = _fixed_step_count(t0, t1, dt0)
        if steps is not None:
            kwargs["max_steps"] = steps + 16
    return dfx.diffeqsolve(
        dfx.ODETerm(vector_field),
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        saveat=saveat if saveat is not None else dfx.SaveAt(t1=True),
        **kwargs,
    )
