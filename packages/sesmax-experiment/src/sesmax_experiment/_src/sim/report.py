"""The lines of a run's log."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import paramax

from sesmax_core import Model
from sesmax_core.fields import values
from sesmax_experiment._src.run.frame import ChunkFrame
from sesmax_experiment._src.sim.artifacts import flat_diagnostics
from sesmax_experiment._src.sim.units import (
    format_field_stats,
    format_time_seconds,
    format_wallclock,
)


# The scalars worth a place in a line, in this order.
_HEADLINES = (
    "total_energy",
    "total_kinetic_energy",
    "energy",
    "kinetic_energy",
    "total_enstrophy",
    "enstrophy",
)


def state_line(state: Any, units: dict[str, str]) -> str:
    """The minimum, mean and maximum of each field, and how much is not finite."""
    parts = []
    for field in dataclasses.fields(state):
        array = np.asarray(values(getattr(state, field.name)))
        finite = array[np.isfinite(array)]
        low, mean, high = (
            (float("nan"),) * 3
            if finite.size == 0
            else (float(finite.min()), float(finite.mean()), float(finite.max()))
        )
        parts.append(
            format_field_stats(
                field.name,
                min_val=low,
                mean_val=mean,
                max_val=high,
                nan_count=int(array.size - finite.size),
                units=units,
            )
        )
    return " ".join(parts)


def physics_line(model: Model, state: Any) -> str:
    """The energies and enstrophies of ``diagnose``, if it gives any."""
    try:
        flat = flat_diagnostics(paramax.unwrap(model).diagnose(state))
    except Exception as exc:
        return f"diagnose failed: {type(exc).__name__}: {exc}"
    return " ".join(f"{k}={flat[k]:.3g}" for k in _HEADLINES if k in flat)


def chunk_line(model: Model, frame: ChunkFrame, units: dict[str, str]) -> str:
    """One chunk of a run, as one line."""
    info = frame.info
    parts = [
        f"chunk {info.index + 1}/{info.n_chunks} sim_t={format_time_seconds(info.t1)}",
        state_line(frame.state, units),
    ]
    if physics := physics_line(model, frame.state):
        parts.append(f"physics: {physics}")
    if info.index < 0:
        parts.append("initial state")
    else:
        parts.append(f"wall={format_wallclock(info.wall_seconds)}")
    if frame.metrics:
        numbers = " ".join(f"{k}={v:.3g}" for k, v in frame.metrics.items())
        parts.append(f"monitors: {numbers}")
    return " | ".join(parts) + "".join(f" !! {m}" for m in frame.messages)
