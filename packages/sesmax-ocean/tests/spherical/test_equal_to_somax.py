"""The models on the sphere give somax's numbers, on arrays and located fields.

What somax computed is in ``tests/reference``, with three differences that are
not the models': finitevolX's T rows are half a cell too far south, so somax ran
half a cell further north; its cell areas are not exact, by a factor that is the
same everywhere; and its radius and rotation rate are passed explicitly. The
streamfunction is solved to a tolerance, so `SphericalQG` is compared at 1e-9.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from equality import (
    CASES,
    SIZE,
    assert_reproduces_somax,
    float64,
    initial_state,
    island,
)
from gridax import CField, Mask2D, StaggerMismatchError
from gridax.schemes import WENO, Upwind
from somax_reference import EXACT_AREA, LAT_RANGE, RTOL, SPHERICAL

from sesmax_ocean.spherical import (
    SphericalQG,
    SphericalQGState,
    SphericalSWM,
    SphericalSWMState,
)


SCHEMES = {"upwind1": Upwind(1), "weno5": WENO(5)}
#: ``family -> (model, state, points, relative difference, area integrals)``
MODELS = {
    "spherical_sw": (
        SphericalSWM,
        SphericalSWMState,
        {"h": "T", "u": "U", "v": "V"},
        RTOL,
        ("energy", "enstrophy", "mass", "casimir_q3"),
    ),
    "spherical_qg": (
        SphericalQG,
        SphericalQGState,
        {"q": "T"},
        1e-9,
        ("kinetic_energy", "enstrophy", "circulation"),
    ),
}


@pytest.mark.parametrize("located", [False, True], ids=["arrays", "cfields"])
@pytest.mark.parametrize(
    ("family", "case"),
    [(family, case) for family, cases in SPHERICAL.items() for case in cases],
)
def test_equal_to_somax(family: str, case: str, located: bool) -> None:
    model_class, state_class, points, rtol, integrals = MODELS[family]
    scheme, has_island = SPHERICAL[family][case]
    with float64():
        model = model_class.create(
            **SIZE,
            **CASES[family][0],
            lat_range=LAT_RANGE,
            scheme=SCHEMES[scheme],
            mask=Mask2D.from_center(island()) if has_island else None,
        )
        given, state = (
            initial_state(
                state_class,
                family,
                case,
                points,
                ("y", "x") if located else None,
                kind=kind,
            )
            for kind in ("given", "init")
        )
        assert_reproduces_somax(
            model,
            given,
            state,
            family,
            case,
            rtol=rtol,
            factors=dict.fromkeys(integrals, EXACT_AREA),
        )


def test_a_vorticity_that_is_not_at_t_is_rejected() -> None:
    model = SphericalQG.create(**SIZE)
    q = CField(jnp.zeros(model.grid.shape), ("y", "x"), loc="X")
    with pytest.raises(StaggerMismatchError):
        model.vector_field(0.0, SphericalQGState(q=q))


def test_a_velocity_at_the_wrong_point_is_rejected() -> None:
    model = SphericalSWM.create(**SIZE)
    zeros = jnp.zeros(model.grid.shape)
    state = SphericalSWMState(
        h=CField(zeros + 1.0, ("y", "x"), loc="T"),
        u=CField(zeros, ("y", "x"), loc="V"),
        v=CField(zeros, ("y", "x"), loc="V"),
    )
    with pytest.raises(StaggerMismatchError):
        model.vector_field(0.0, state)
