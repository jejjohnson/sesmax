"""`Model.integrate` compiles once across calls.

A right-hand side runs as Python only while it is traced, so a call that finds
the compiled solve of the last one does not run it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from sesmax_core import (
    Model,
    ScaledModel,
    State,
    StateAffine,
    TermModel,
    implicit,
)
from sesmax_core._src.terms import Term
from sesmax_core.solvers import imex_solver, imex_stepsize_controller


TRACES: list[str] = []


class DecayState(State):
    x: Array


class Decay(Model):
    rate: Array

    def vector_field(self, t: float, state: Any, args: Any = None) -> DecayState:
        TRACES.append("decay")
        return DecayState(x=-self.rate * state.x)

    def apply_boundary_conditions(self, state: Any) -> Any:
        return state


class DecayTerm(Term):
    rate: Array

    def __call__(self, t: float, state: Any, args: Any = None) -> DecayState:
        TRACES.append("term")
        return DecayState(x=-self.rate * state.x)


def _decay() -> Model:
    return Decay(rate=jnp.asarray(1.0))


def _terms() -> Model:
    return TermModel(terms=DecayTerm(rate=jnp.asarray(1.0)))


def _imex() -> Model:
    explicit_part = DecayTerm(rate=jnp.asarray(1.0))
    return TermModel(terms=explicit_part + implicit(DecayTerm(rate=jnp.asarray(2.0))))


def _scaled(inner: Callable[[], Model]) -> Callable[[], Model]:
    def build() -> Model:
        identity = StateAffine.identity(DecayState(x=jnp.ones(3)))
        return ScaledModel(inner=inner(), transform=identity, time_scale=2.0)

    return build


IMEX = {
    "solver": imex_solver(),
    "stepsize_controller": imex_stepsize_controller(),
    "max_steps": 1000,
}
CASES: dict[str, tuple[Callable[[], Model], dict[str, Any]]] = {
    "model": (_decay, {}),
    "term model": (_terms, {}),
    "imex": (_imex, IMEX),
    "scaled": (_scaled(_decay), {}),
    "scaled term model": (_scaled(_terms), {}),
    "scaled imex": (_scaled(_imex), IMEX),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_second_call_does_not_trace(case: str) -> None:
    """At other times, and on a model built again."""
    build, settings = CASES[case]
    state = DecayState(x=jnp.ones(3))
    build().integrate(state, t0=0.0, t1=0.1, dt=0.01, **settings)
    TRACES.clear()
    for k in range(1, 4):
        t0 = 0.1 * k
        build().integrate(state, t0=t0, t1=t0 + 0.1, dt=0.01, **settings)
    assert TRACES == []


def test_another_parameter_does_not_trace() -> None:
    state = DecayState(x=jnp.ones(3))
    model = _decay()
    first = model.step(state, 0.1)
    TRACES.clear()
    faster = eqx.tree_at(lambda m: m.rate, model, jnp.asarray(2.0))
    second = faster.step(state, 0.1, t0=0.3)
    assert TRACES == []
    assert bool((second.x < first.x).all())


@pytest.mark.parametrize("case", ["model", "term model", "scaled"])
def test_the_answer_is_diffrax_s_own(case: str) -> None:
    """The same solve, written out with Python times and a closure."""
    build, settings = CASES[case]
    model = build()
    state = DecayState(x=jnp.ones(3))

    def closure(t: Any, y: Any, args: Any) -> Any:
        return model.vector_field(t, y, args)

    expected = dfx.diffeqsolve(
        dfx.ODETerm(closure), dfx.Tsit5(), t0=0.0, t1=0.3, dt0=0.01, y0=state
    ).ys
    got = model.integrate(state, t0=0.0, t1=0.3, dt=0.01, **settings).ys
    assert bool(jnp.allclose(got.x, expected.x, rtol=1e-6))


def test_the_times_may_be_traced() -> None:
    model = _decay()
    state = DecayState(x=jnp.ones(3))

    def end(t1: Array) -> Array:
        return model.integrate(state, t0=0.0, t1=t1, dt=0.01).ys.x[-1].sum()

    times = jnp.array([0.1, 0.2])
    values = jax.vmap(end)(times)
    assert bool(jnp.allclose(values, 3.0 * jnp.exp(-times), rtol=1e-4))


def test_no_first_step_for_an_adaptive_controller() -> None:
    """``dt=None`` leaves the first step to the controller."""
    state = DecayState(x=jnp.ones(3))
    solution = _decay().integrate(
        state,
        t0=0.0,
        t1=1.0,
        dt=None,
        stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
    )
    assert bool(jnp.allclose(solution.ys.x, jnp.exp(-1.0), rtol=1e-3))
