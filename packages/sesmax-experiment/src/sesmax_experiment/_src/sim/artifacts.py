"""What a run writes: snapshots, the final state, metrics and a manifest."""

from __future__ import annotations

import dataclasses
import json
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import jax
import numpy as np
from pipekit_ops.hashing import sha256_hex, stable_json

from sesmax_core import io
from sesmax_core.fields import values
from sesmax_experiment._src.run.frame import ChunkFrame
from sesmax_experiment._src.sim.spec import RunSpec


#: The libraries whose versions decide a run's numbers.
_LIBRARIES = (
    "sesmax-experiment",
    "sesmax-core",
    "sesmax-ocean",
    "jax",
    "diffrax",
    "equinox",
    "gridax",
    "pipekit-ops",
    "pipekit-cycle",
)


def run_attrs(
    spec: RunSpec, *, mode: str, coordinates: str, t0: float
) -> dict[str, Any]:
    """The attributes that say where a store came from, numbers as numbers.

    Args:
        spec: The spec of the run.
        mode: ``"simulate"``, ``"spinup"`` or ``"restart"``.
        coordinates: ``"si"`` or ``"nondimensional"``. The class of a state is the
            same either way, so a restart reads this to know its units.
        t0: The start of the run, later than the spec's if it is a restart.

    Returns:
        The attributes.
    """
    times = spec.timestepping
    return {
        "sesmax_sim_mode": mode,
        "sesmax_coordinates": coordinates,
        "scenario_name": spec.scenario.name,
        "model_name": spec.model.name,
        "t0": float(t0),
        "t1": float(times.t1),
        "dt": float(times.dt),
        "save_interval": float(times.save_interval),
    }


class SnapshotStore:
    """Streams snapshots to a partial store, and names it when the run succeeds.

    A run that fails leaves ``snapshots.partial.zarr`` for inspection, and no
    ``snapshots.zarr``.

    Args:
        output_dir: The directory of the run.
        attrs: The attributes of the store.
    """

    def __init__(self, output_dir: Path, attrs: dict[str, Any]) -> None:
        self.path = output_dir / "snapshots.zarr"
        self.partial = output_dir / "snapshots.partial.zarr"
        self.attrs = attrs
        self._started = False
        for stale in (self.path, self.partial):
            if stale.exists():
                shutil.rmtree(stale)

    def __call__(self, frame: ChunkFrame) -> None:
        """Append one snapshot."""
        ds = io.state_to_dataset(frame.state, time=frame.info.t1, attrs=self.attrs)
        if self._started:
            io.append_to_dataset(ds, self.partial)
        else:
            io.save_dataset(ds, self.partial, mode="w")
            self._started = True

    def finish(self) -> Path:
        """Give the store its name, a new one, so nothing is overwritten."""
        self.partial.rename(self.path)
        return self.path


def flat_diagnostics(diagnostics: Any) -> dict[str, float]:
    """The numbers of a ``Diagnostics``, for a metrics file.

    A number keeps its name, a short vector becomes ``<field>_layer_<i>``, and a
    larger array its mean, maximum and minimum.

    Args:
        diagnostics: A dataclass of arrays, or anything else for no numbers.

    Returns:
        ``name -> number``.
    """
    out: dict[str, float] = {}
    if not dataclasses.is_dataclass(diagnostics):
        return out
    for field in dataclasses.fields(diagnostics):
        value = getattr(diagnostics, field.name)
        try:
            array = np.asarray(values(value))
            if array.dtype.kind not in "biuf":
                continue
        except (TypeError, ValueError):
            continue
        array = array.astype(np.float64)  # a mean overflows sooner in float32
        if array.ndim == 0:
            out[field.name] = float(array)
        elif array.ndim == 1 and array.size <= 16:
            for i, number in enumerate(array.tolist()):
                out[f"{field.name}_layer_{i}"] = float(number)
        else:
            out[f"{field.name}_mean"] = float(array.mean())
            out[f"{field.name}_max"] = float(array.max())
            out[f"{field.name}_min"] = float(array.min())
    return out


def write_json(path: Path, data: dict[str, Any]) -> Path:
    """Write a mapping as sorted, indented JSON."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return path


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _git(directory: Path) -> dict[str, Any]:
    """The commit and whether the tree is dirty, or ``None`` outside a checkout."""

    def run(*args: str) -> str:
        command = ["git", "-C", str(directory), *args]
        out = subprocess.check_output(command, stderr=subprocess.DEVNULL)
        return out.decode().strip()

    try:
        return {
            "git_commit": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain")),
        }
    except (subprocess.SubprocessError, OSError):
        return {"git_commit": None, "git_dirty": None}


def manifest(
    spec: RunSpec,
    *,
    mode: str,
    status: str,
    reason: str | None,
    t0: float,
    t1: float,
    wall_seconds: float,
    numerics: dict[str, Any],
) -> dict[str, Any]:
    """What it takes to run this again, and to read its numbers.

    Args:
        spec: The spec of the run.
        mode: ``"simulate"``, ``"spinup"`` or ``"restart"``.
        status: ``"finished"`` or ``"stopped"``.
        reason: Why it stopped, or ``None``.
        t0: Where it started.
        t1: Where it finished or stopped.
        wall_seconds: How long it took.
        numerics: What the component says of the model's numerics; two runs that
            differ only in a filter strength can give different results.

    Returns:
        A mapping that JSON can hold.
    """
    return {
        "mode": mode,
        "status": status,
        "reason": reason,
        "versions": {name: _version(name) for name in _LIBRARIES},
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "platform": jax.default_backend(),
        "python": platform.python_version(),
        "device_count": jax.device_count(),
        "config_sha256": sha256_hex(stable_json(spec.to_dict())),
        "t0": t0,
        "t1": t1,
        "dt": spec.timestepping.dt,
        "wallclock_seconds": wall_seconds,
        "numerics": numerics,
        **_git(Path(__file__).resolve().parent),
    }
