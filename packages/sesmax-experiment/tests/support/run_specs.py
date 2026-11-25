"""The run specs the tests of the runner share."""

from __future__ import annotations

from typing import Any

from sesmax_experiment.sim import ModelSpec, RunSpec, ScenarioSpec, TimesteppingSpec


def shallow_water(
    *,
    t1: float = 600.0,
    dt: float = 10.0,
    save_interval: float | None = None,
    nx: int = 16,
    **spec: Any,
) -> RunSpec:
    """The testing component's shallow-water step, a wave of about 31 m/s."""
    return RunSpec(
        scenario=ScenarioSpec(
            name="adjustment",
            grid={"nx": nx, "ny": nx, "Lx": 1.0e6, "Ly": 1.0e6},
            consts={"f0": 1.0e-4},
            initial_condition={"type": "step", "params": {"amplitude": 1.0}},
        ),
        model=ModelSpec(name="linear_swm", params={"H0": 100.0}),
        timestepping=TimesteppingSpec(
            t0=0.0, t1=t1, dt=dt, save_interval=save_interval or t1
        ),
        **spec,
    )


def lorenz96(*, nondim: bool = False, **spec: Any) -> RunSpec:
    """A Lorenz-96 ring."""
    return RunSpec(
        scenario=ScenarioSpec(
            name="ring",
            grid={"nx": 12},
            nondim={"rossby": 0.1} if nondim else {},
        ),
        model=ModelSpec(name="lorenz96", params={"F": 8.0}),
        timestepping=TimesteppingSpec(t0=0.0, t1=0.4, dt=0.01, save_interval=0.2),
        **spec,
    )
