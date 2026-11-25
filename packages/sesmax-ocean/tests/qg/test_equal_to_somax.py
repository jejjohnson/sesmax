"""The QG models give somax's numbers, on arrays and on located fields.

What somax computed is in ``tests/reference``: from one random PV, the
tendency, the state after 20 Heun steps and the diagnostics, with and without
an island.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from equality import (
    CASES,
    SIZE,
    assert_reproduces_somax,
    float64,
    initial_state,
    island,
)
from gridax import CField, Mask2D, StaggerMismatchError

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


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize("case", ["nomask", "island"])
@pytest.mark.parametrize("family", list(MODELS))
def test_equal_to_somax(family: str, case: str, located: bool) -> None:
    model_class, state_class, dims = MODELS[family]
    with float64():
        mask = Mask2D.from_center(island()) if case == "island" else None
        model = model_class.create(**SIZE, **CASES[family][0], mask=mask)
        given, state = (
            initial_state(
                state_class,
                family,
                case,
                {"q": "T"},
                dims if located else None,
                kind=kind,
            )
            for kind in ("given", "init")
        )
        assert_reproduces_somax(model, given, state, family, case)


@pytest.mark.parametrize("family", list(MODELS))
def test_a_pv_that_is_not_at_t_is_rejected(family: str) -> None:
    model_class, state_class, dims = MODELS[family]
    model = model_class.create(**SIZE)
    q = jnp.zeros((*((3,) if len(dims) == 3 else ()), model.grid.Ny, model.grid.Nx))
    with pytest.raises(StaggerMismatchError):
        model.vector_field(0.0, state_class(q=CField(q, dims, loc="X")))
