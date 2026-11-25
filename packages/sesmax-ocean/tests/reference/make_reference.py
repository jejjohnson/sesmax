"""Write the arrays that the equality tests compare sesmax-ocean with.

The reference is somax at commit 9cb19b3, on the finitevolX it pins, in
float64. Run this in somax's environment, from a checkout of that commit:

    git -C ../somax archive 9cb19b3 | tar -x -C /tmp/somax-9cb19b3
    cd /tmp/somax-9cb19b3 && uv sync --no-dev
    uv run --no-sync python <this file> <family> [<family> ...]

Each family is one ``<family>.npz`` beside this file, whose keys are
``case|kind|field``: the input, and what somax computes from it.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


NX, NY = 16, 12
Arrays = dict[str, np.ndarray]


def island() -> np.ndarray:
    """Wet cells of a closed box with an island, with the ghost ring."""
    wet = np.zeros((NY + 2, NX + 2), dtype=bool)
    wet[1:-1, 1:-1] = True
    wet[5:8, 7:10] = False
    return wet


def masks() -> dict[str, object]:
    """The two domains every case runs on."""
    from finitevolx import Mask2D

    return {"nomask": None, "island": Mask2D.from_center(jnp.asarray(island()))}


def wet(mask: object) -> tuple[object, object, object]:
    """The wet points at T, U and V; a velocity is zero on the dry faces."""
    if mask is None:
        return 1.0, 1.0, 1.0
    return np.asarray(mask.h), np.asarray(mask.u), np.asarray(mask.v)


def record(out: Arrays, case: str, kind: str, tree: object) -> None:
    """Store every array field of a state or of diagnostics."""
    for field in dataclasses.fields(tree):
        out[f"{case}|{kind}|{field.name}"] = np.asarray(getattr(tree, field.name))


def eval_metrics() -> Arrays:
    """The reference-free metrics, on the linear shallow-water model."""
    from somax._src.models.swm.linear_2d import LinearShallowWater2D, LinearSW2DState
    from somax.eval import compute_eval_metrics

    rng = np.random.default_rng(0)
    out: Arrays = {}
    for name, mask in masks().items():
        model = LinearShallowWater2D.create(
            nx=NX, ny=NY, Lx=1e6, Ly=8e5, beta=1e-11, bc="wall", mask=mask
        )
        wet_t, wet_u, wet_v = wet(mask)
        state = LinearSW2DState(
            h=jnp.asarray(rng.normal(size=(NY + 2, NX + 2)) * wet_t),
            u=jnp.asarray(0.1 * rng.normal(size=(NY + 2, NX + 2)) * wet_u),
            v=jnp.asarray(0.1 * rng.normal(size=(NY + 2, NX + 2)) * wet_v),
        )
        state = model.apply_boundary_conditions(state)
        record(out, name, "init", state)
        for metric, value in compute_eval_metrics(model, state).items():
            out[f"{name}|metric|{metric}"] = np.asarray(value)
    return out


FAMILIES: dict[str, Callable[[], Arrays]] = {"eval": eval_metrics}


def main(families: list[str]) -> None:
    """Write the given families, or all of them."""
    # Not at import: pytest imports this module to collect doctests.
    jax.config.update("jax_enable_x64", True)
    for family in families or list(FAMILIES):
        path = Path(__file__).parent / f"{family}.npz"
        np.savez_compressed(path, **FAMILIES[family]())
        print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
