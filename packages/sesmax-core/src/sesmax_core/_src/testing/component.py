"""The testing component: models the run loop's own tests run."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from sesmax_core._src.component import BuiltRun, Component
from sesmax_core._src.fields import values
from sesmax_core._src.model import Model
from sesmax_core._src.testing.gfd_testcases import geostrophic_adjustment_2d
from sesmax_core._src.testing.lorenz96 import L96State, Lorenz96
from sesmax_core._src.testing.swm.linear_2d import LinearShallowWater2D
from sesmax_core._src.types import State


def lorenz96_run(spec: Mapping[str, Any]) -> BuiltRun:
    """A Lorenz-96 ring, near its equilibrium.

    ``scenario.grid`` takes ``nx``, ``scenario.initial_condition.params``
    ``perturbation`` and ``model.params`` ``F``. A non-empty ``scenario.nondim``
    marks the run as nondimensional; Lorenz-96 has no units to lose.
    """
    scenario, model = spec["scenario"], spec["model"]
    F = float(model["params"].get("F", 8.0))
    initial = scenario["initial_condition"].get("params", {})
    x = jnp.full(int(scenario["grid"].get("nx", 40)), F)
    x = x.at[0].add(float(initial.get("perturbation", 0.01)))
    return BuiltRun(
        model=Lorenz96.create(F=F),
        state0=L96State(x=x),
        coordinates="nondimensional" if scenario.get("nondim") else "si",
    )


def linear_swm_run(spec: Mapping[str, Any]) -> BuiltRun:
    """The geostrophic adjustment of a step in height, in linear shallow water.

    ``scenario.grid`` takes ``nx``, ``ny``, ``Lx`` and ``Ly``, ``scenario.consts``
    ``f0``, ``scenario.initial_condition.params`` ``amplitude`` and
    ``model.params`` ``H0``.
    """
    scenario, model = spec["scenario"], spec["model"]
    grid = scenario["grid"]
    initial = scenario["initial_condition"].get("params", {})
    built, state0 = geostrophic_adjustment_2d(
        nx=int(grid.get("nx", 32)),
        ny=int(grid.get("ny", 32)),
        Lx=float(grid.get("Lx", 1e6)),
        Ly=float(grid.get("Ly", 1e6)),
        f0=float(scenario["consts"].get("f0", 1e-4)),
        H0=float(model["params"].get("H0", 100.0)),
        eta_max=float(initial.get("amplitude", 1.0)),
    )
    return BuiltRun(model=built, state0=state0)


def check_gravity_wave_cfl(spec: Any, model: Model, *, limit: float = 1.0) -> None:
    """``c Δt / Δx`` of the fastest wave, below a limit.

    Raises:
        ValueError: If it is not.
    """
    if not isinstance(model, LinearShallowWater2D):
        raise TypeError(f"not a shallow-water model: {type(model).__name__}")
    speed = (model.consts.gravity * model.consts.H0) ** 0.5
    spacing = min(float(model.grid.spacing(axis)) for axis in "xy")
    courant = speed * spec.timestepping.dt / spacing
    if not courant < limit:
        raise ValueError(f"the Courant number {courant:.2f} is not below {limit}")


def _evaluate(model: Model, state: State) -> dict[str, float]:
    """The largest value of each field."""
    del model
    return {
        f"largest_{name}": float(jnp.max(jnp.abs(values(getattr(state, name)))))
        for name in (field.name for field in dataclasses.fields(state))
    }


def _describe_numerics(model: Model) -> dict[str, Any]:
    """The class, and the boundary condition if there is one."""
    out = {"model_class": type(model).__name__}
    if hasattr(model, "bc_type"):
        out["bc"] = model.bc_type
    return out


#: The test problems, as a component a run spec can name.
COMPONENT = Component(
    name="testing",
    models={"lorenz96": lorenz96_run, "linear_swm": linear_swm_run},
    checks={"gravity_wave_cfl": check_gravity_wave_cfl},
    evaluate=_evaluate,
    field_units={"linear_swm": {"h": "m", "u": "m/s", "v": "m/s"}},
    describe_numerics=_describe_numerics,
)
