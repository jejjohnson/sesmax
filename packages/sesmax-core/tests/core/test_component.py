"""A component registers its models without the package that runs them."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from sesmax_core.component import BuiltRun, Component
from sesmax_core.testing import L96State, Lorenz96


def _build(spec) -> BuiltRun:
    model = Lorenz96.create(F=spec["F"])
    return BuiltRun(model=model, state0=L96State(x=jnp.arange(spec["n"], dtype=float)))


COMPONENT = Component(name="toys", models={"lorenz96": _build})


def test_a_builder_gives_a_model_and_its_state() -> None:
    run = COMPONENT.models["lorenz96"]({"F": 8.0, "n": 6})
    assert run.state0.x.shape == (6,)
    assert run.coordinates == "si"
    assert run.model.step(run.state0, 0.01).x.shape == (6,)


def test_the_optional_parts_are_empty() -> None:
    assert dict(COMPONENT.checks) == {}
    assert dict(COMPONENT.field_units) == {}
    assert COMPONENT.evaluate is None
    assert COMPONENT.describe_numerics is None


def test_a_component_does_not_change() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        COMPONENT.name = "other"  # type: ignore[misc]
