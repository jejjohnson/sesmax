"""Write the arrays that the equality tests compare sesmax-ocean with.

The reference is somax at commit 9cb19b3, on the finitevolX it pins, in
float64. Run this in somax's environment, from a checkout of that commit:

    git -C ../somax archive 9cb19b3 | tar -x -C /tmp/somax-9cb19b3
    cd /tmp/somax-9cb19b3 && uv sync --no-dev
    uv pip install pyyaml  # for the family "configs"
    uv run --no-sync python <this file> <family> [<family> ...]

Each family is one ``<family>.npz`` beside this file, whose keys are
``case|kind|field``: the input, and what somax computes from it.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "support"))

from somax_reference import (
    CASES,
    CHUNK,
    CONFIGS,
    HALF_CELL,
    LAT_RANGE,
    N_STEPS,
    SHALLOW_WATER,
    SIZE,
    SPHERICAL,
    island,
)


SHAPE = (SIZE["ny"] + 2, SIZE["nx"] + 2)
Arrays = dict[str, np.ndarray]


def masks() -> dict[str, object]:
    """The two domains every case runs on."""
    from finitevolx import Mask2D

    return {"nomask": None, "island": Mask2D.from_mask(jnp.asarray(island()))}


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
        model = LinearShallowWater2D.create(**SIZE, **CASES["eval"][0], mask=mask)
        wet_t, wet_u, wet_v = wet(mask)
        state = LinearSW2DState(
            h=jnp.asarray(rng.normal(size=SHAPE) * wet_t),
            u=jnp.asarray(0.1 * rng.normal(size=SHAPE) * wet_u),
            v=jnp.asarray(0.1 * rng.normal(size=SHAPE) * wet_v),
        )
        state = model.apply_boundary_conditions(state)
        record(out, name, "init", state)
        for metric, value in compute_eval_metrics(model, state).items():
            out[f"{name}|metric|{metric}"] = np.asarray(value)
    return out


def integrate(out: Arrays, case: str, model: object, state: object, dt: float) -> None:
    """The tendency, the end state and the diagnostics from one initial state."""
    import diffrax as dfx

    record(out, case, "given", state)
    state = model.apply_boundary_conditions(state)
    solution = model.integrate(state, t0=0.0, t1=N_STEPS * dt, dt=dt, solver=dfx.Heun())
    end = jax.tree.map(lambda x: x[-1], solution.ys)
    record(out, case, "init", state)
    record(out, case, "tendency", model.vector_field(0.0, state))
    record(out, case, "end", end)
    record(out, case, "diagnostics", model.diagnose(state))


def quasi_geostrophic(family: str) -> Arrays:
    """One of the QG models, from random PV."""
    from somax._src.models.qg import baroclinic, barotropic

    model_class, state_class, layers = {
        "barotropic_qg": (barotropic.BarotropicQG, barotropic.BarotropicQGState, ()),
        "baroclinic_qg": (baroclinic.BaroclinicQG, baroclinic.BaroclinicQGState, (3,)),
    }[family]
    kwargs, dt = CASES[family]
    rng = np.random.default_rng(0)
    out: Arrays = {}
    for name, mask in masks().items():
        wet_t, _, _ = wet(mask)
        q = jnp.asarray(1e-5 * rng.normal(size=(*layers, *SHAPE)) * wet_t)
        model = model_class.create(**SIZE, **kwargs, mask=mask)
        integrate(out, name, model, state_class(q=q), dt)
    return out


def shallow_water(family: str) -> Arrays:
    """One of the models with a thickness and a velocity, from random fields."""
    from somax._src.models.qg.reparameterized import ReparameterizedQG
    from somax._src.models.swm import multilayer, nonlinear_2d

    model_class, state_class, depth = {
        "nonlinear_sw": (
            nonlinear_2d.NonlinearShallowWater2D,
            nonlinear_2d.NonlinearSW2DState,
            np.asarray(100.0),
        ),
        "multilayer_sw": (
            multilayer.MultilayerShallowWater2D,
            multilayer.MultilayerSW2DState,
            np.array([400.0, 1100.0, 2600.0])[:, None, None],
        ),
        "reparameterized_qg": (
            ReparameterizedQG,
            multilayer.MultilayerSW2DState,
            np.array([400.0, 1100.0, 2600.0])[:, None, None],
        ),
    }[family]
    kwargs, dt = CASES[family]
    shape = (*depth.shape[:1], *SHAPE)
    out: Arrays = {}
    for case, (bc, scheme, has_island) in SHALLOW_WATER[family].items():
        rng = np.random.default_rng(0)
        mask = masks()["island"] if has_island else None
        _, wet_u, wet_v = wet(mask)
        state = state_class(
            # A thickness on land too: the PV divides by it.
            h=jnp.asarray(depth + rng.normal(size=shape)),
            u=jnp.asarray(0.1 * rng.normal(size=shape) * wet_u),
            v=jnp.asarray(0.1 * rng.normal(size=shape) * wet_v),
        )
        model = model_class.create(**SIZE, **kwargs, bc=bc, method=scheme, mask=mask)
        integrate(out, case, model, state, dt)
    return out


def spherical(family: str) -> Arrays:
    """One of the models on the sphere, from random fields."""
    from somax._src.models.spherical import (
        SphericalQG,
        SphericalQGState,
        SphericalSWM,
        SphericalSWMState,
    )

    kwargs, dt = CASES[family]
    lat_range = tuple(lat + HALF_CELL for lat in LAT_RANGE)
    out: Arrays = {}
    for case, (scheme, has_island) in SPHERICAL[family].items():
        rng = np.random.default_rng(0)
        mask = masks()["island"] if has_island else None
        wet_t, wet_u, wet_v = wet(mask)
        if family == "spherical_qg":
            model_class = SphericalQG
            state = SphericalQGState(
                q=jnp.asarray(1e-5 * rng.normal(size=SHAPE) * wet_t)
            )
        else:
            model_class = SphericalSWM
            state = SphericalSWMState(
                h=jnp.asarray(1000.0 + rng.normal(size=SHAPE)),
                u=jnp.asarray(0.1 * rng.normal(size=SHAPE) * wet_u),
                v=jnp.asarray(0.1 * rng.normal(size=SHAPE) * wet_v),
            )
        model = model_class.create(
            **SIZE, **kwargs, lat_range=lat_range, method=scheme, mask=mask
        )
        integrate(out, case, model, state, dt)
    return out


def configs() -> Arrays:
    """somax's run configs: where they start, and their first chunk."""
    import yaml
    from somax._src.cli._factories import build

    out: Arrays = {}
    for name, n in CONFIGS.items():
        spec = yaml.safe_load(Path(f"configs/simulation/{name}.yaml").read_text())
        scenario, model_spec = spec["scenario"], spec["model"]
        scenario["grid"] |= {"nx": n, "ny": n}
        model, state = build(
            scenario.pop("name"),
            model_spec.pop("name"),
            nondim=scenario.pop("nondim", None),
            scenario_params=scenario,
            model_params=model_spec,
        )
        state = model.apply_boundary_conditions(state)
        dt = spec["timestepping"]["dt"]
        solution = model.integrate(state, t0=0.0, t1=CHUNK * dt, dt=dt)
        record(out, name, "init", state)
        record(out, name, "end", jax.tree.map(lambda x: x[-1], solution.ys))
    return out


FAMILIES: dict[str, Callable[[], Arrays]] = {
    "configs": configs,
    "eval": eval_metrics,
    "barotropic_qg": partial(quasi_geostrophic, "barotropic_qg"),
    "baroclinic_qg": partial(quasi_geostrophic, "baroclinic_qg"),
    "nonlinear_sw": partial(shallow_water, "nonlinear_sw"),
    "multilayer_sw": partial(shallow_water, "multilayer_sw"),
    "reparameterized_qg": partial(shallow_water, "reparameterized_qg"),
    "spherical_sw": partial(spherical, "spherical_sw"),
    "spherical_qg": partial(spherical, "spherical_qg"),
}


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
