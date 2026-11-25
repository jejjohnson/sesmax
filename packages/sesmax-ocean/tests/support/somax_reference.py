"""Compare sesmax-ocean with the arrays somax computed.

The arrays are in ``tests/reference/*.npz``, written by
``tests/reference/make_reference.py`` with somax at 9cb19b3. That script runs
in somax's environment and reads the cases from here, so this module imports
nothing of sesmax.
"""

from __future__ import annotations

import math
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
#: somax's radius (m) and rotation rate (1/s) of the Earth; sesmax-ocean takes
#: geophysix's.
RADIUS = 6_371_200.0
OMEGA = 2.0 * math.pi / 86_164.0
#: The band of the models on the sphere (degrees). finitevolX puts the T rows
#: half a cell too far south (gridax#433), so somax runs half a cell further
#: north.
LAT_RANGE = (-60.0, 60.0)
HALF_CELL = 0.5 * (LAT_RANGE[1] - LAT_RANGE[0]) / SIZE["ny"]
#: finitevolX's cell areas are ``R² cos φ Δλ Δφ``; the exact ones that gridax has
#: are smaller by this factor, everywhere.
EXACT_AREA = math.sin(math.radians(HALF_CELL)) / math.radians(HALF_CELL)
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
    "nonlinear_sw": (
        {
            "Lx": 1e6,
            "Ly": 8e5,
            "beta": 1e-11,
            "lateral_viscosity": 500.0,
            "bottom_drag": 1e-6,
            "wind_amplitude": 1e-6,
        },
        300.0,
    ),
    "multilayer_sw": (
        {
            "Lx": 1e6,
            "Ly": 8e5,
            "lateral_viscosity": 500.0,
            "bottom_drag": 1e-6,
            "wind_amplitude": 1e-3,
        },
        60.0,
    ),
    "reparameterized_qg": (
        {
            "Lx": 1e6,
            "Ly": 8e5,
            "lateral_viscosity": 500.0,
            "bottom_drag": 1e-6,
            "wind_amplitude": 1e-3,
        },
        60.0,
    ),
    "spherical_sw": (
        {
            "radius": RADIUS,
            "omega": OMEGA,
            "lateral_viscosity": 1e4,
            "bottom_drag": 1e-6,
            "wind_amplitude": 1e-6,
        },
        300.0,
    ),
    "spherical_qg": (
        {
            "radius": RADIUS,
            "omega": OMEGA,
            "lateral_viscosity": 1e4,
            "bottom_drag": 1e-6,
            "wind_amplitude": 1e-12,
            "cg_tol": 1e-12,
        },
        600.0,
    ),
}

#: ``family -> case -> (scheme, island)`` of the models on the sphere.
SPHERICAL: dict[str, dict[str, tuple[str, bool]]] = {
    "spherical_sw": {
        "band-upwind1": ("upwind1", False),
        "band-weno5": ("weno5", False),
        "island-upwind1": ("upwind1", True),
        "island-weno5": ("weno5", True),
    },
    "spherical_qg": {
        "band-upwind1": ("upwind1", False),
        "island-upwind1": ("upwind1", True),
    },
}

#: ``family -> case -> (bc, scheme, island)`` of the shallow-water models; a
#: scheme is the name somax gives it.
SHALLOW_WATER: dict[str, dict[str, tuple[str, str, bool]]] = {
    "nonlinear_sw": {
        "periodic-upwind1": ("periodic", "upwind1", False),
        "periodic-weno5": ("periodic", "weno5", False),
        "wall-upwind1": ("wall", "upwind1", False),
        "wall-weno5": ("wall", "weno5", False),
        "island-upwind1": ("wall", "upwind1", True),
        "island-weno5": ("wall", "weno5", True),
    },
    "multilayer_sw": {
        "periodic-upwind1": ("periodic", "upwind1", False),
        "wall-weno5": ("wall", "weno5", False),
        "island-upwind1": ("wall", "upwind1", True),
        "island-weno5": ("wall", "weno5", True),
    },
    "reparameterized_qg": {
        "wall-upwind1": ("wall", "upwind1", False),
        "island-upwind1": ("wall", "upwind1", True),
    },
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
