"""The checks a spec asks for, before a run and after it.

``spec.assertions`` maps a check's name to its parameters. Those here are for any
component; a component adds its own, which run before a run.

Moved from somax 9cb19b3: ``somax/_src/cli/_assertions.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sesmax_core import Model, stable_step
from sesmax_core.component import AssertionFailedError, Component
from sesmax_experiment._src.sim.spec import RunSpec


def _narrowest_cell(model: Model) -> float:
    """The narrowest cell of a model's grid, off the ghost ring."""
    grid = getattr(model, "grid", None)
    if grid is None:
        raise AssertionFailedError(
            f"cfl: model {type(model).__name__!r} has no .grid attribute; "
            "cannot infer dx"
        )
    widths = []
    for axis in grid.axes:
        width = np.asarray(grid.spacing(axis))
        if width.ndim:
            width = width[(slice(1, -1),) * width.ndim]
        widths.append(float(width.min()))
    return min(widths)


def check_cfl(
    spec: RunSpec,
    model: Model,
    *,
    wave_speed_m_per_s: float,
    max_cfl: float = 0.5,
) -> None:
    """The step of a run, against the stable one for a wave speed.

    The speed is the caller's, because the fastest signal depends on the equations:
    the external gravity wave, an internal one, or the currents.

    Args:
        spec: The spec of the run.
        model: The model, for its grid.
        wave_speed_m_per_s: The fastest speed, in the units of the run.
        max_cfl: The largest Courant number on the narrowest cell.

    Raises:
        AssertionFailedError: If the step is too large, a number is not positive, or
            the model has no grid.
    """
    speed, dt = float(wave_speed_m_per_s), float(spec.timestepping.dt)
    if not speed > 0.0:
        raise AssertionFailedError(f"cfl: wave_speed_m_per_s must be > 0 (got {speed})")
    if not max_cfl > 0.0:
        raise AssertionFailedError(f"cfl: max_cfl must be > 0 (got {max_cfl})")
    dx = _narrowest_cell(model)
    stable = float(stable_step((dx,), speed=(speed,), cfl=max_cfl))
    # At the limit passes, whatever the rounding of the bound.
    if dt > stable * (1.0 + 1e-6):
        raise AssertionFailedError(
            f"cfl check FAILED: CFL = {speed * dt / dx:.3f} > max_cfl = {max_cfl}\n"
            f"  wave_speed = {speed:.2f} m/s\n"
            f"  dt         = {dt:.4f} s\n"
            f"  dx_min     = {dx:.2f} m\n"
            f"  → maximum stable dt at this CFL: {stable:.4f} s"
        )


def check_bounded_metric(
    spec: RunSpec,
    metrics: Mapping[str, Any],
    *,
    name: str,
    min: float | None = None,
    max: float | None = None,
) -> None:
    """A metric of a run, between two bounds.

    Args:
        spec: The spec of the run, unused.
        metrics: The metrics.
        name: The metric.
        min: The smallest value that passes, or ``None``.
        max: The largest, or ``None``.

    Raises:
        AssertionFailedError: If the metric is missing, not a finite number, or out
            of the bounds.
    """
    del spec
    if name not in metrics:
        raise AssertionFailedError(
            f"bounded_metric: metric {name!r} not present in run output. "
            f"Available metrics: {sorted(metrics)}"
        )
    raw = metrics[name]
    try:
        value = float(np.asarray(raw))
    except (TypeError, ValueError) as exc:
        raise AssertionFailedError(
            f"bounded_metric: metric {name!r} is not a numeric scalar (got {raw!r})"
        ) from exc
    if not np.isfinite(value):
        raise AssertionFailedError(
            f"bounded_metric: metric {name!r} is non-finite ({value})"
        )
    if min is not None and value < min:
        raise AssertionFailedError(
            f"bounded_metric: {name} = {value} is below min = {min}"
        )
    if max is not None and value > max:
        raise AssertionFailedError(
            f"bounded_metric: {name} = {value} is above max = {max}"
        )


#: The checks for any component, before a run and after it.
PREFLIGHT: dict[str, Callable[..., None]] = {"cfl": check_cfl}
POSTFLIGHT: dict[str, Callable[..., None]] = {"bounded_metric": check_bounded_metric}


def _checked(spec: RunSpec, component: Component) -> dict[str, Callable[..., None]]:
    """The checks before a run, once every name in the spec is known."""
    if taken := sorted(set(component.checks) & (PREFLIGHT.keys() | POSTFLIGHT.keys())):
        raise ValueError(
            f"the component {component.name!r} has checks named as the run's: {taken}"
        )
    before = PREFLIGHT | dict(component.checks)
    if unknown := sorted(set(spec.assertions) - before.keys() - POSTFLIGHT.keys()):
        raise AssertionFailedError(
            f"unknown assertion {unknown}; available before a run: {sorted(before)}; "
            f"after it: {sorted(POSTFLIGHT)}"
        )
    return before


def preflight(spec: RunSpec, model: Model, component: Component) -> None:
    """Run the checks of a spec before any step is taken.

    Args:
        spec: The spec of the run.
        model: The model that was built.
        component: The component of the model, for its checks.

    Raises:
        AssertionFailedError: If a check fails or is not known.
        ValueError: If the component has a check named as one of the run's.
    """
    checks = _checked(spec, component)
    for name, params in spec.assertions.items():
        if name in checks:
            checks[name](spec, model, **params)


def asks_for_metrics(spec: RunSpec) -> bool:
    """Whether a spec has a check that reads the metrics of the run."""
    return any(name in POSTFLIGHT for name in spec.assertions)


def postflight(spec: RunSpec, metrics: Mapping[str, Any]) -> None:
    """Run the checks of a spec on the metrics of a run, before they are written.

    Args:
        spec: The spec of the run.
        metrics: Its metrics.

    Raises:
        AssertionFailedError: If a check fails.
    """
    for name, params in spec.assertions.items():
        if name in POSTFLIGHT:
            POSTFLIGHT[name](spec, metrics, **params)
