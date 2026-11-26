"""The run configs start where somax's do, and run the same first chunk.

What somax computed is in ``tests/reference/configs.npz``, on a coarser grid than
the configs have.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import pytest
import yaml
from equality import float64, load
from somax_numerics import with_somax_numerics
from somax_reference import CHUNK, CONFIGS, assert_equal_to_somax

from sesmax_core import Model, State
from sesmax_core.fields import values
from sesmax_ocean.scenarios import build_ocean_run


DIRECTORY = Path(__file__).resolve().parents[2] / "configs"


def _spec(name: str) -> dict[str, Any]:
    """A config, on the grid of the reference."""
    spec = yaml.safe_load((DIRECTORY / f"{name}.yaml").read_text("utf-8"))
    spec["scenario"]["grid"] |= {"nx": CONFIGS[name], "ny": CONFIGS[name]}
    return spec


@eqx.filter_jit
def _chunk(model: Model, state: State, dt: float) -> State:
    """The steps of `integrate`, which fills the ghosts of the end; somax did not."""
    solution = dfx.diffeqsolve(
        model.build_terms(), dfx.Tsit5(), 0.0, CHUNK * dt, dt, state
    )
    return jax.tree.map(lambda x: x[-1], solution.ys)


def test_every_config_is_compared() -> None:
    assert {path.stem for path in DIRECTORY.glob("*.yaml")} == set(CONFIGS)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_a_config_is_somax_s(name: str) -> None:
    expected = load("configs")[name]
    spec = _spec(name)
    with float64():
        run = build_ocean_run(spec)
        model = with_somax_numerics(run.model)
        end = _chunk(model, run.state0, spec["timestepping"]["dt"])
        for kind, state in (("init", run.state0), ("end", end)):
            for field in dataclasses.fields(state):
                assert_equal_to_somax(
                    values(getattr(state, field.name)),
                    expected[kind][field.name],
                    f"{name}/{kind}/{field.name}",
                )
