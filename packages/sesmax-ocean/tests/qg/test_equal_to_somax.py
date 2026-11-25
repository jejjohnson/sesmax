"""The QG models give somax's numbers, on arrays and on located fields.

What somax computed is in ``tests/reference``: from one random PV, the
tendency, the state after 20 Heun steps and the diagnostics, with and without
an island.
"""

from __future__ import annotations

import dataclasses

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest
from gridax import CField, Mask2D, StaggerMismatchError
from somax_reference import (
    CASES,
    N_STEPS,
    SIZE,
    assert_equal_to_somax,
    float64,
    island,
    load,
)

from sesmax_core.fields import values
from sesmax_ocean.qg import (
    BaroclinicQG,
    BaroclinicQGState,
    BarotropicQG,
    BarotropicQGState,
)


MODELS = {
    "barotropic_qg": (BarotropicQG, BarotropicQGState, ("y", "x")),
    "baroclinic_qg": (BaroclinicQG, BaroclinicQGState, ("layer", "y", "x")),
}


def _model_and_state(family: str, case: str, located: bool):
    model_class, state_class, dims = MODELS[family]
    mask = Mask2D.from_center(island()) if case == "island" else None
    model = model_class.create(**SIZE, **CASES[family][0], mask=mask)
    q = jnp.asarray(load(family)[case]["init"]["q"])
    return model, state_class(q=CField(q, dims, loc="T") if located else q)


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize("case", ["nomask", "island"])
@pytest.mark.parametrize("family", list(MODELS))
def test_equal_to_somax(family: str, case: str, located: bool) -> None:
    with float64():
        model, state = _model_and_state(family, case, located)
        expected = load(family)[case]
        dt = CASES[family][1]

        tendency = model.vector_field(0.0, state)
        solution = model.integrate(
            state, t0=0.0, t1=N_STEPS * dt, dt=dt, solver=dfx.Heun()
        )
        end = jax.tree.map(lambda x: x[-1], solution.ys)
        assert isinstance(tendency.q, CField) == located
        got = {
            "init": model.apply_boundary_conditions(state),
            "tendency": tendency,
            "end": end,
            "diagnostics": model.diagnose(state),
        }
        for kind, tree in got.items():
            for field in dataclasses.fields(tree):
                assert_equal_to_somax(
                    values(getattr(tree, field.name)),
                    expected[kind][field.name],
                    f"{family}/{case}/{kind}/{field.name}",
                )


@pytest.mark.parametrize("family", list(MODELS))
def test_a_pv_that_is_not_at_t_is_rejected(family: str) -> None:
    model_class, state_class, dims = MODELS[family]
    model = model_class.create(**SIZE)
    q = jnp.zeros((*((3,) if len(dims) == 3 else ()), model.grid.Ny, model.grid.Nx))
    with pytest.raises(StaggerMismatchError):
        model.vector_field(0.0, state_class(q=CField(q, dims, loc="X")))
