"""Checks of a model before a run, beside those of its resolution.

Moved from somax 9cb19b3: ``somax/_src/cli/_assertions.py``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import paramax
from gridax import apply_mask

from sesmax_core.component import AssertionFailedError
from sesmax_ocean._src.eval import pv_unknowns


def check_pv_inversion(spec: object | None, model: Any, *, tol: float = 1e-3) -> None:
    r"""Does the inversion of a barotropic QG model undo its Laplacian?

    Inverts a random potential vorticity :math:`q`, the same one every time, and
    compares :math:`\nabla^2 \psi` with it. A broken solver, a wrong boundary
    stencil or a field on the wrong points gives a difference of order one.

    Not for the layered models, whose inversion has a stretching term that the
    Laplacian alone does not undo.

    Args:
        spec: The spec of the run, unused.
        model: The model.
        tol: The largest :math:`\lVert \nabla^2 \psi - q \rVert / \lVert q \rVert`.

    Raises:
        AssertionFailedError: If the difference is larger, or the model is no
            barotropic QG model.

    Examples:
        >>> from sesmax_ocean.checks import check_pv_inversion
        >>> from sesmax_ocean.qg import BarotropicQG
        >>> check_pv_inversion(None, BarotropicQG.create(nx=16, ny=16))
    """
    del spec
    model = paramax.unwrap(model)
    invert = getattr(model, "invert_pv", None) or getattr(model, "_invert_pv", None)
    if invert is None or hasattr(model, "strat"):
        raise AssertionFailedError(
            f"pv_inversion: {type(model).__name__!r} is no barotropic QG model, with "
            "q = laplacian(psi)"
        )
    grid = model.grid
    q = jnp.zeros((grid.Ny, grid.Nx))
    inside = pv_unknowns(model)
    q = q.at[inside].set(jax.random.normal(jax.random.key(0), q[inside].shape))
    if model.mask is not None:
        q = apply_mask(q, model.mask, at="T")
    undone = model.ops.laplacian(invert(q), at="T")
    residual = float(jnp.linalg.norm((undone - q)[inside]) / jnp.linalg.norm(q))
    if not residual <= tol:
        raise AssertionFailedError(
            f"pv_inversion check FAILED: relative residual {residual:.2e} > tol "
            f"{tol:.0e}\n"
            "  ||laplacian(psi) - q|| / ||q|| where the inversion solves for psi"
        )


def check_static_stability(spec: object | None, model: Any) -> None:
    """Does the density of a layered model increase with depth?

    Args:
        spec: The spec of the run, unused.
        model: The model.

    Raises:
        AssertionFailedError: If the reduced gravity of an interface between two
            layers is not positive, or the model has no layers.

    Examples:
        >>> from sesmax_ocean.checks import check_static_stability
        >>> from sesmax_ocean.qg import BaroclinicQG
        >>> check_static_stability(None, BaroclinicQG.create(nx=16, ny=16))
    """
    del spec
    strat = getattr(paramax.unwrap(model), "strat", None)
    if strat is None:
        raise AssertionFailedError(
            f"static_stability: {type(model).__name__!r} has no layers"
        )
    # The first is the surface, with the full gravity.
    g_prime = np.asarray(strat.g_prime)[1:]
    if bad := [k + 1 for k, g in enumerate(g_prime) if not g > 0.0]:
        raise AssertionFailedError(
            f"static_stability check FAILED: the reduced gravity of interface(s) {bad} "
            f"is not positive (g' = {g_prime.tolist()})\n"
            "  order the layers from light to dense, top to bottom"
        )
