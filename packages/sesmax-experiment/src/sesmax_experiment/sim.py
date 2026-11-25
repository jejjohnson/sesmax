"""Public surface for runs described by a spec; needs the ``[sim]`` extra.

:func:`simulate`, :func:`spinup` and :func:`restart` take a :class:`RunSpec` and a
directory, find the component that offers the spec's model, run it in chunks and
write the artifacts. ``sesmax-experiment`` is the same on the command line.
"""

from __future__ import annotations


try:
    import cyclopts  # noqa: F401
    import loguru  # noqa: F401
    import xarray  # noqa: F401
    import yaml  # noqa: F401
    import zarr  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "sesmax_experiment.sim needs cyclopts, loguru, pyyaml, xarray and zarr. "
        "Install them with `pip install sesmax-experiment[sim]`."
    ) from exc

from sesmax_experiment._src.sim.app import app, main
from sesmax_experiment._src.sim.runner import (
    IntegrationDivergedError,
    SimulationResult,
    build_run,
    restart,
    simulate,
    spinup,
    timeline_of,
)
from sesmax_experiment._src.sim.spec import (
    DebugSpec,
    ModelSpec,
    OutputSpec,
    RunSpec,
    ScenarioSpec,
    TimesteppingSpec,
    dump_yaml,
    load_yaml,
)


__all__ = [
    "DebugSpec",
    "IntegrationDivergedError",
    "ModelSpec",
    "OutputSpec",
    "RunSpec",
    "ScenarioSpec",
    "SimulationResult",
    "TimesteppingSpec",
    "app",
    "build_run",
    "dump_yaml",
    "load_yaml",
    "main",
    "restart",
    "simulate",
    "spinup",
    "timeline_of",
]
