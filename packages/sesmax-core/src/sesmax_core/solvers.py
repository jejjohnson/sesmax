"""Public surface for sesmax-core's solver helpers.

:func:`imex_solver` / :func:`imex_stepsize_controller` build a memory-safe,
matrix-free IMEX integrator for stiff term-based models (those built with
``imex=True``). They avoid the dense-Jacobian OOM that a bare
``diffrax.KenCarp3()`` hits at large grids (#55) by solving the implicit
stage with a Krylov (GMRES) Newton root-finder — O(N) memory rather than
O(N^2).

Pure JAX / diffrax / lineax / optimistix (all base dependencies), so
importing this is cheap; it is kept out of ``sesmax_core``'s top-level
``__init__`` to mirror the module-per-surface layout (cf.
:mod:`sesmax_core.guards`).

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

from sesmax_core._src.solvers import imex_solver, imex_stepsize_controller


__all__ = [
    "imex_solver",
    "imex_stepsize_controller",
]
