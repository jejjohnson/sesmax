"""Public surface for sesmax-core's solver helpers, which live in integrix.

:func:`imex_solver` / :func:`imex_stepsize_controller` build a memory-safe,
matrix-free IMEX integrator for stiff term-based models (those built with
``imex=True``). They avoid the dense-Jacobian OOM that a bare
``diffrax.KenCarp3()`` hits at large grids (#55) by solving the implicit
stage with a Krylov (GMRES) Newton root-finder — O(N) memory rather than
O(N^2).

:func:`stable_step` is the largest stable step for an explicit solve, from the
actual spacings, speeds and diffusivities. It returns a 0-d JAX array (``inf``
when nothing moves or diffuses) and traces under ``jit``.

The three are defined in :mod:`integrix.march` (a base dependency) and
re-exported here unchanged, so ``sesmax_core.solvers.imex_solver is
integrix.march.imex_solver``. The module is kept out of ``sesmax_core``'s
top-level ``__init__`` (except :func:`stable_step`) to mirror the
module-per-surface layout (cf. :mod:`sesmax_core.guards`).

Example::

    from sesmax_core.solvers import imex_solver, imex_stepsize_controller

    # Burgers2DTermModel comes from sesmax_core.testing
    model = Burgers2DTermModel.create(nx=256, ny=256, nu=0.05, imex=True)
    sol = model.integrate(
        state0,
        t0=0.0,
        t1=1.0,
        dt=1e-3,
        solver=imex_solver(),
        stepsize_controller=imex_stepsize_controller(),
    )
"""

from __future__ import annotations

from integrix.march import (
    imex_solver,
    imex_stepsize_controller,
    stable_step,
)


__all__ = [
    "imex_solver",
    "imex_stepsize_controller",
    "stable_step",
]
