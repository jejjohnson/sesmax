"""Run a ported model beside the arrays somax computed.

Apart from `somax_reference`, which the reference script imports in somax's
environment, because this module imports sesmax.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from gridax import CField
from somax_numerics import with_somax_numerics
from somax_reference import (
    CASES,
    N_STEPS,
    RTOL,
    SIZE,
    assert_equal_to_somax,
    float64,
    island,
    load,
)

from sesmax_core import Model, State
from sesmax_core.fields import values


__all__ = [
    "CASES",
    "SIZE",
    "assert_reproduces_somax",
    "float64",
    "initial_state",
    "island",
    "load",
]


def initial_state(
    state_class: type[State],
    family: str,
    case: str,
    points: dict[str, str],
    dims: tuple[str, ...] | None,
    *,
    kind: str = "init",
) -> State:
    """The state the reference starts from.

    Args:
        state_class: The class of the state.
        family: A family of `CASES`.
        case: One of its cases.
        points: The point of each field.
        dims: The dims of located fields, or ``None`` for arrays.
        kind: ``"init"``, or ``"given"`` for the state before its boundary
            conditions.

    Returns:
        The state, of arrays or of `gridax.CField`s.
    """
    fields = {}
    for name, at in points.items():
        array = jnp.asarray(load(family)[case][kind][name])
        fields[name] = array if dims is None else CField(array, dims, loc=at)
    return state_class(**fields)


@eqx.filter_jit
def _run(model: Model, given: State, state: State, dt: float) -> dict[str, Any]:
    """What the reference holds, computed by the ported model."""
    solution = model.integrate(state, t0=0.0, t1=N_STEPS * dt, dt=dt, solver=dfx.Heun())
    return {
        "init": model.apply_boundary_conditions(given),
        "tendency": model.vector_field(0.0, state),
        "end": jax.tree.map(lambda x: x[-1], solution.ys),
        "diagnostics": model.diagnose(state),
    }


def assert_reproduces_somax(
    model: Model,
    given: State,
    state: State,
    family: str,
    case: str,
    *,
    rtol: float = RTOL,
    factors: Mapping[str, float] = MappingProxyType({}),
) -> None:
    """The boundary conditions, the tendency, the end state, the diagnostics agree.

    Args:
        model: The ported model.
        given: The state of `initial_state` before its boundary conditions.
        state: The state of `initial_state`.
        family: A family of `CASES`.
        case: One of its cases.
        rtol: The relative difference allowed over the interior.
        factors: What to multiply somax's value of a diagnostic by, where the two
            differ by a known factor.
    """
    expected = load(family)[case]
    got = _run(with_somax_numerics(model), given, state, CASES[family][1])
    for field in dataclasses.fields(state):
        located = isinstance(getattr(state, field.name), CField)
        assert isinstance(getattr(got["tendency"], field.name), CField) == located
    for kind, tree in got.items():
        for field in dataclasses.fields(tree):
            assert_equal_to_somax(
                values(getattr(tree, field.name)),
                expected[kind][field.name] * factors.get(field.name, 1.0),
                f"{family}/{case}/{kind}/{field.name}",
                rtol=rtol,
            )
