"""The checks before and after a run.

Ported from somax 9cb19b3 (``tests/test_cli_assertions.py``), on grids of gridax in
place of stubs and on the testing component.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gridax import CartesianCGrid1D, CartesianCGrid2D, SphericalCGrid2D
from pipekit_cycle import Cycle, ForwardModel
from run_specs import shallow_water

from sesmax_core.component import AssertionFailedError, Component
from sesmax_core.operators import ModelOp
from sesmax_experiment._src.sim.checks import (
    POSTFLIGHT,
    PREFLIGHT,
    postflight,
    preflight,
)
from sesmax_experiment.sim import (
    OutputSpec,
    check_bounded_metric,
    check_cfl,
    model_op,
    simulate,
    spinup,
)


NONE = Component(name="none", models={})


def _box(dx: float, dy: float | None = None) -> Any:
    """A model of which the check needs the grid."""
    dy = dx if dy is None else dy
    grid = CartesianCGrid2D.from_interior(8, 8, 8 * dx, 8 * dy)
    return SimpleNamespace(grid=grid)


class TestCFL:
    def test_a_small_step_passes(self) -> None:
        check_cfl(shallow_water(), _box(15625.0), wave_speed_m_per_s=221.0)

    def test_the_limit_passes(self) -> None:
        check_cfl(shallow_water(), _box(4420.0), wave_speed_m_per_s=221.0)

    def test_a_large_step_fails_and_says_which_would_not(self) -> None:
        with pytest.raises(AssertionFailedError, match="cfl check FAILED") as error:
            check_cfl(shallow_water(dt=300.0), _box(15625.0), wave_speed_m_per_s=221.0)
        for part in (
            "CFL = 4.243",
            "wave_speed = 221",
            "dt         = 300",
            "dx_min     = 15625",
            "maximum stable dt at this CFL: 35.35",
        ):
            assert part in str(error.value)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_the_numbers_are_positive(self, bad: float) -> None:
        spec, model = shallow_water(), _box(1e4)
        with pytest.raises(AssertionFailedError, match="wave_speed_m_per_s must be"):
            check_cfl(spec, model, wave_speed_m_per_s=bad)
        with pytest.raises(AssertionFailedError, match="max_cfl must be > 0"):
            check_cfl(spec, model, wave_speed_m_per_s=1.0, max_cfl=bad)

    def test_the_narrowest_cell_counts(self) -> None:
        with pytest.raises(AssertionFailedError, match="dx_min     = 1000"):
            check_cfl(shallow_water(), _box(1e4, 1e3), wave_speed_m_per_s=100.0)

    def test_a_line_has_one_axis(self) -> None:
        model = SimpleNamespace(grid=CartesianCGrid1D.from_interior(8, 8e3))
        with pytest.raises(AssertionFailedError, match="dx_min     = 1000"):
            check_cfl(shallow_water(), model, wave_speed_m_per_s=100.0)

    def test_a_sphere_has_its_narrowest_by_the_pole(self) -> None:
        grid = SphericalCGrid2D.from_interior(36, 16, (0.0, 360.0), (-80.0, 80.0))
        widest = 111.2e3 * 10.0
        # The last row of cells is at 75 degrees; its ghost, at 85, is not a cell.
        narrowest = widest * np.cos(np.deg2rad(75.0))
        model, dt = SimpleNamespace(grid=grid), 10.0
        limit = 0.5 * narrowest / dt
        check_cfl(shallow_water(), model, wave_speed_m_per_s=0.99 * limit)
        with pytest.raises(AssertionFailedError, match="cfl check FAILED"):
            check_cfl(shallow_water(), model, wave_speed_m_per_s=1.01 * limit)

    def test_a_model_without_a_grid(self) -> None:
        with pytest.raises(AssertionFailedError, match=r"has no \.grid attribute"):
            check_cfl(shallow_water(), object(), wave_speed_m_per_s=1.0)


class TestBoundedMetric:
    def test_in_the_bounds(self) -> None:
        check_bounded_metric(None, {"ke": 0.5}, name="ke", min=0.0, max=1.0)

    @pytest.mark.parametrize(
        ("value", "bounds", "message"),
        [
            (-0.1, {"min": 0.0, "max": 1.0}, "below min"),
            (1.5, {"min": 0.0, "max": 1.0}, "above max"),
            (-1.0, {"min": 0.0}, "below min"),
            (1.0, {"max": 0.0}, "above max"),
            (float("nan"), {}, "non-finite"),
            (float("inf"), {"max": 1.0}, "non-finite"),
            ("not a number", {"min": 0.0}, "not a numeric scalar"),
        ],
    )
    def test_out_of_them(self, value: Any, bounds: dict, message: str) -> None:
        with pytest.raises(AssertionFailedError, match=message):
            check_bounded_metric(None, {"ke": value}, name="ke", **bounds)

    def test_a_metric_the_run_does_not_have(self) -> None:
        with pytest.raises(AssertionFailedError, match="'nonexistent' not present"):
            check_bounded_metric(None, {"a": 1.0}, name="nonexistent", min=0.0)


class TestWhichRun:
    def test_none_is_asked_for(self) -> None:
        preflight(shallow_water(), _box(1e4), NONE)
        postflight(shallow_water(), {"x": 1.0})

    def test_a_name_nobody_has(self) -> None:
        spec = shallow_water(assertions={"unknown_check": {}})
        with pytest.raises(AssertionFailedError, match="unknown assertion"):
            preflight(spec, _box(1e4), NONE)

    def test_each_at_its_time(self) -> None:
        spec = shallow_water(
            assertions={
                "cfl": {"wave_speed_m_per_s": 100.0},
                "bounded_metric": {"name": "x", "min": 0.0},
            }
        )
        preflight(spec, _box(1e4), NONE)  # no metrics yet
        postflight(spec, {"x": 1.0})
        with pytest.raises(AssertionFailedError, match="below min"):
            postflight(spec, {"x": -1.0})
        spec.assertions["cfl"]["wave_speed_m_per_s"] = 1e9
        postflight(spec, {"x": 1.0})
        with pytest.raises(AssertionFailedError, match="cfl check FAILED"):
            preflight(spec, _box(1e4), NONE)

    @pytest.mark.parametrize("name", [*PREFLIGHT, *POSTFLIGHT])
    def test_a_component_does_not_take_a_name(self, name: str) -> None:
        component = Component(name="c", models={}, checks={name: lambda *a: None})
        with pytest.raises(ValueError, match="checks named as the run's"):
            preflight(shallow_water(), _box(1e4), component)


class TestInARun:
    def test_before_it(self, tmp_path: Path) -> None:
        cfl = {"cfl": {"wave_speed_m_per_s": 31.4, "max_cfl": 0.5}}
        with pytest.raises(AssertionFailedError, match="cfl check FAILED"):
            simulate(shallow_water(dt=1200.0, t1=2400.0, assertions=cfl), tmp_path)
        assert not any(tmp_path.iterdir())
        assert simulate(shallow_water(assertions=cfl), tmp_path).status == "finished"

    def test_after_it(self, tmp_path: Path) -> None:
        bad = {"bounded_metric": {"name": "largest_h", "max": 1e-3}}
        with pytest.raises(AssertionFailedError, match="above max"):
            simulate(shallow_water(assertions=bad), tmp_path / "bad")
        left = {p.name for p in (tmp_path / "bad").iterdir()}
        assert left == {"snapshots.partial.zarr", "run.log", "metrics.jsonl"}
        good = {"bounded_metric": {"name": "largest_h", "max": 10.0}}
        simulate(shallow_water(assertions=good), tmp_path / "good")

    def test_after_one_that_writes_no_metrics(self, tmp_path: Path) -> None:
        spec = shallow_water(
            assertions={"bounded_metric": {"name": "largest_h", "max": 1e-3}},
            output=OutputSpec(write_metrics=False),
        )
        with pytest.raises(AssertionFailedError, match="above max"):
            simulate(spec, tmp_path)
        # A spinup has none to check, with the same spec.
        assert spinup(spec, tmp_path / "spinup").status == "finished"


class TestModelOp:
    def test_the_model_of_a_spec(self) -> None:
        op, state = model_op(shallow_water(dt=30.0))
        assert isinstance(op, ModelOp) and isinstance(op, ForwardModel)
        assert type(op.model).__name__ == "LinearShallowWater2D"
        assert op.dt == 30.0
        np.testing.assert_array_equal(op(state).h, op.model.step(state, 30.0).h)

    def test_it_drives_a_cycle(self) -> None:
        op, state = model_op(shallow_water(), dt=20.0)
        end, _ = Cycle(step_op=op, n_steps=2)(state, None)
        np.testing.assert_array_equal(end.h, op.step(op.step(state, 20.0), 20.0).h)
