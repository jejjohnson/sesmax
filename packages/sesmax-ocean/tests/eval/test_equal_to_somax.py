"""The metrics give somax's numbers, on arrays and on located fields."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from gridax import CField, Mask2D
from somax_reference import assert_equal_to_somax, float64, island, load

from sesmax_core.testing import LinearShallowWater2D, LinearSW2DState
from sesmax_ocean.eval import compute_eval_metrics


CASES = load("eval")


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_metrics_equal_somax(case: str, located: bool) -> None:
    with float64():
        mask = Mask2D.from_center(island()) if case == "island" else None
        model = LinearShallowWater2D.create(
            nx=16, ny=12, Lx=1e6, Ly=8e5, beta=1e-11, bc="wall", mask=mask
        )
        fields = {name: jnp.asarray(x) for name, x in CASES[case]["init"].items()}
        if located:
            points = {"h": "T", "u": "U", "v": "V"}
            fields = {
                name: CField(x, ("y", "x"), loc=points[name])
                for name, x in fields.items()
            }
        metrics = compute_eval_metrics(model, LinearSW2DState(**fields))
        expected = CASES[case]["metric"]
        assert set(metrics) == set(expected)
        for name, value in expected.items():
            assert_equal_to_somax(metrics[name], value, f"{case}/{name}")
