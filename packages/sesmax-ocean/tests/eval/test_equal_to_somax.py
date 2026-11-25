"""The metrics give somax's numbers, on arrays and on located fields."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from gridax import CField, Mask2D
from somax_reference import CASES, SIZE, assert_equal_to_somax, float64, island, load

from sesmax_core.testing import LinearShallowWater2D, LinearSW2DState
from sesmax_ocean.eval import compute_eval_metrics


REFERENCE = load("eval")


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize("case", sorted(REFERENCE))
def test_metrics_equal_somax(case: str, located: bool) -> None:
    with float64():
        mask = Mask2D.from_center(island()) if case == "island" else None
        model = LinearShallowWater2D.create(**SIZE, **CASES["eval"][0], mask=mask)
        fields = {name: jnp.asarray(x) for name, x in REFERENCE[case]["init"].items()}
        if located:
            points = {"h": "T", "u": "U", "v": "V"}
            fields = {
                name: CField(x, ("y", "x"), loc=points[name])
                for name, x in fields.items()
            }
        metrics = compute_eval_metrics(model, LinearSW2DState(**fields))
        expected = REFERENCE[case]["metric"]
        assert set(metrics) == set(expected)
        for name, value in expected.items():
            assert_equal_to_somax(metrics[name], value, f"{case}/{name}")
