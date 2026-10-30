"""Tests for ``ModelOp`` on a model defined here.

somax tests ``SomaxModelOp`` through ``Burgers2DOp`` and the run loop's
registry (``tests/test_operators.py`` at 9cb19b3); those cases move with
``sesmax_core.testing`` and the runner. These check the wrapper itself.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, PyTree
from pipekit import Operator

from sesmax_core import Model, Params, State
from sesmax_core.operators import ModelOp


class Scalar(State):
    x: Array


class DecayParams(Params):
    rate: Array


class Decay(Model):
    params: DecayParams

    def vector_field(
        self, t: float, state: PyTree, args: PyTree | None = None
    ) -> PyTree:
        return Scalar(x=-self.params.rate * state.x)

    def apply_boundary_conditions(self, state: PyTree) -> PyTree:
        return state


@pytest.fixture
def op() -> ModelOp:
    return ModelOp(Decay(params=DecayParams(rate=jnp.asarray(0.5))), dt=0.1)


def test_is_a_pipekit_operator(op):
    assert isinstance(op, Operator)


def test_apply_advances_one_dt(op):
    s0 = Scalar(x=jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(op(s0).x, op.model.step(s0, 0.1).x)


def test_sequential_composition_is_two_steps(op):
    s0 = Scalar(x=jnp.array([1.0, 2.0]))
    np.testing.assert_allclose((op | op)(s0).x, op.step(op.step(s0, 0.1), 0.1).x)


def test_forward_model_protocol(op):
    assert op.dt == 0.1
    assert op.state_signature is None
    assert op.forbid_in_yaml is True
