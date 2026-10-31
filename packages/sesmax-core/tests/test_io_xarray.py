"""Round trips through ``sesmax_core.io`` on states defined here and on Lorenz-96.

somax's own suite (``tests/test_io_xarray.py`` at 9cb19b3) runs on ocean
states and moves with them. These check the same contracts in the core:
dimension conventions, round trips, Zarr persistence, the auto-recovery
allowlist, and reading stores written by somax.
"""

from __future__ import annotations

import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array

from sesmax_core import State
from sesmax_core.io import (
    dataset_to_state,
    load_dataset,
    save_dataset,
    snapshots_to_dataset,
    state_to_dataset,
)
from sesmax_core.testing import L96State


class Surface(State):
    eta: Array


class Layered(State):
    h: Array
    u: Array


def test_lorenz96_round_trips_and_recovers_its_class():
    state = L96State(x=jnp.arange(8.0))
    ds = state_to_dataset(state)
    assert ds.attrs["state_module"] == "sesmax_core._src.testing.lorenz96"
    back = dataset_to_state(ds)
    assert type(back) is L96State
    np.testing.assert_array_equal(back.x, state.x)


def test_dims_follow_the_conventions():
    surface = state_to_dataset(Surface(eta=jnp.zeros((3, 4))))
    layered = state_to_dataset(Layered(h=jnp.zeros((2, 3, 4)), u=jnp.zeros((2, 3, 4))))
    assert surface["eta"].dims == ("y", "x")
    assert layered["h"].dims == ("layer", "y", "x")


def test_explicit_class_round_trip_for_a_local_state():
    state = Layered(h=jnp.ones((2, 3, 4)), u=2.0 * jnp.ones((2, 3, 4)))
    back = dataset_to_state(state_to_dataset(state), Layered)
    np.testing.assert_array_equal(back.u, state.u)


def test_snapshots_prepend_time_and_select_by_index():
    snaps = Surface(eta=jnp.stack([jnp.full((3, 4), float(k)) for k in range(3)]))
    ds = snapshots_to_dataset(snaps, np.array([0.0, 1.0, 2.0]))
    assert ds["eta"].dims == ("time", "y", "x")
    first = dataset_to_state(ds, Surface, time_index=0)
    np.testing.assert_array_equal(first.eta, 0.0)


def test_zarr_save_and_load_round_trip(tmp_path):
    state = L96State(x=jnp.linspace(-1.0, 1.0, 6))
    save_dataset(state_to_dataset(state), tmp_path / "state.zarr")
    back = dataset_to_state(load_dataset(tmp_path / "state.zarr").load())
    np.testing.assert_allclose(back.x, state.x)


def test_a_foreign_state_module_is_refused():
    ds = state_to_dataset(L96State(x=jnp.zeros(4)))
    ds.attrs["state_module"] = "os"
    with pytest.raises(ValueError, match="allowlisted"):
        dataset_to_state(ds)


def test_a_store_written_by_somax_loads_as_the_sesmax_class():
    """somax recorded ``somax._src.models.lorenz96``; sesmax reads it."""
    ds = state_to_dataset(L96State(x=jnp.arange(4.0)))
    ds.attrs["state_module"] = "somax._src.models.lorenz96"
    back = dataset_to_state(ds)
    assert type(back) is L96State


def test_a_somax_state_with_no_sesmax_home_says_so():
    ds = state_to_dataset(L96State(x=jnp.zeros(4)))
    ds.attrs["state_module"] = "somax._src.models.qg.barotropic"
    ds.attrs["state_class"] = "BarotropicQGState"
    with pytest.raises(ValueError, match="Pass state_class"):
        dataset_to_state(ds)


@pytest.mark.parametrize(
    ("code", "extra"),
    [
        (
            "from sesmax_core.io import state_to_dataset\n"
            "from sesmax_core.testing import L96State\n"
            "import jax.numpy as jnp\n"
            "state_to_dataset(L96State(x=jnp.zeros(2)))\n",
            "sesmax-core[xarray]",
        ),
        (
            "from sesmax_core import SimulationCheckpointer\n"
            "ckpt = SimulationCheckpointer(\n"
            "    checkpoint_dir='unused', checkpoint_interval=1\n"
            ")\n"
            "ckpt.restore(0, None, None)\n",
            "sesmax-core[checkpoint]",
        ),
    ],
)
def test_missing_extra_is_named(code, extra):
    blocked = "xarray" if "xarray" in extra else "orbax"
    script = (
        "import sys\n"
        f"sys.modules[{blocked!r}] = None\n"
        "try:\n"
        + "".join(f"    {line}\n" for line in code.splitlines())
        + "except ImportError as err:\n"
        "    print(err)\n"
        "else:\n"
        "    raise SystemExit('ran without the extra')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert extra in result.stdout, result.stdout
