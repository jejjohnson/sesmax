"""Compare sesmax-ocean with the arrays somax computed.

The arrays are in ``tests/reference/*.npz``, written by
``tests/reference/make_reference.py`` with somax at 9cb19b3. That script runs
in somax's environment and reads the cases from here, so this module imports
nothing of sesmax.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import numpy as np


REFERENCE = Path(__file__).resolve().parents[1] / "reference"
#: The relative difference over the interior cells a port may show in float64.
RTOL = 1e-12

#: Interior cells of every case.
SIZE = {"nx": 16, "ny": 12}
#: Steps of the Heun scheme from ``init`` to ``end``.
N_STEPS = 20
#: ``family -> (arguments of create, time step)``.
CASES: dict[str, tuple[dict[str, Any], float]] = {
    "eval": ({"Lx": 1e6, "Ly": 8e5, "beta": 1e-11, "bc": "wall"}, 0.0),
    "barotropic_qg": (
        {
            "Lx": 1e6,
            "Ly": 8e5,
            "lateral_viscosity": 500.0,
            "bottom_drag": 1e-7,
            "wind_amplitude": 1e-11,
        },
        600.0,
    ),
    "baroclinic_qg": (
        {
            "Lx": 1e6,
            "Ly": 8e5,
            "lateral_viscosity": 500.0,
            "bottom_drag": 1e-7,
            "wind_amplitude": 1e-9,
        },
        600.0,
    ),
}


def load(family: str) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """``case -> kind -> field -> array`` of one family."""
    tree: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    with np.load(REFERENCE / f"{family}.npz") as arrays:
        for key in arrays.files:
            case, kind, field = key.split("|")
            tree.setdefault(case, {}).setdefault(kind, {})[field] = arrays[key]
    return tree


def island(shape: tuple[int, int] = (14, 18)) -> np.ndarray:
    """The wet cells of the cases named ``island``, with the ghost ring.

    For ``Mask2D.from_center``, which the reference uses too.
    """
    wet = np.zeros(shape, dtype=bool)
    wet[1:-1, 1:-1] = True
    wet[5:8, 7:10] = False
    return wet


def assert_equal_to_somax(
    got: object, expected: np.ndarray, name: str, *, rtol: float = RTOL
) -> None:
    """``max |got - expected| / max |expected|`` over the interior is below ``rtol``.

    The ghost ring of a horizontal field is left out; a number is compared as
    it is. An expected field of zeros has to be zero.
    """
    actual = np.asarray(got)
    assert actual.dtype == expected.dtype == np.float64, (name, actual.dtype)
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    if expected.ndim >= 2:
        interior = (..., slice(1, -1), slice(1, -1))
        actual, expected = actual[interior], expected[interior]
    scale = np.abs(expected).max()
    difference = np.abs(actual - expected).max()
    assert difference <= rtol * scale, (
        f"{name}: relative difference {difference / scale:.1e} from somax"
    )


@contextmanager
def float64() -> Iterator[None]:
    """Run a test in double precision, as the reference was."""
    with jax.enable_x64(True):
        yield
