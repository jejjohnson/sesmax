"""The walls and coasts of the shallow-water models: free slip, so q = f / h.

A mirrored state has the mirrored tendency, where the four walls are alike.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gridax import Mask2D, laplacian
from gridax.schemes import WENO, Upwind
from somax_reference import float64

from sesmax_ocean.spherical import SphericalSWM, SphericalSWMState
from sesmax_ocean.swm import (
    MultilayerShallowWater2D,
    MultilayerSW2DState,
    NonlinearShallowWater2D,
    NonlinearSW2DState,
)


NY, NX = 12, 16
F0 = 1e-4
MODELS: dict[str, tuple[Any, Any, tuple[int, ...]]] = {
    "nonlinear": (NonlinearShallowWater2D, NonlinearSW2DState, ()),
    "multilayer": (MultilayerShallowWater2D, MultilayerSW2DState, (3,)),
}


def _island(periodic: bool = False) -> np.ndarray:
    wet = np.full((NY + 2, NX + 2), periodic)
    wet[1:-1, 1:-1] = True
    wet[4:7, 6:10] = False
    return wet


def _box(name: str, f0: float, wet: np.ndarray | None, **kwargs: Any) -> Any:
    return MODELS[name][0].create(
        nx=NX,
        ny=NY,
        Lx=1e6,
        Ly=8e5,
        f0=f0,
        beta=0.0,
        lateral_viscosity=1e4,
        bc="wall",
        mask=None if wet is None else Mask2D.from_mask(wet),
        **kwargs,
    )


def _random_state(name: str, wet: np.ndarray | None) -> dict[str, np.ndarray]:
    layers = MODELS[name][2]
    rng = np.random.default_rng(0)
    shape = (*layers, NY + 2, NX + 2)
    fields = {
        "h": 500.0 + rng.normal(size=shape),
        "u": 0.1 * rng.normal(size=shape),
        "v": 0.1 * rng.normal(size=shape),
    }
    if wet is not None:
        mask = Mask2D.from_mask(wet)
        fields["u"] = fields["u"] * np.asarray(mask.wet(at="U"))
        fields["v"] = fields["v"] * np.asarray(mask.wet(at="V"))
    return fields


def _mirrored(x: np.ndarray, axis: int, on_faces: bool) -> np.ndarray:
    """Mirror a field about the middle of the box, along one axis.

    The faces normal to the axis have one place fewer inside the box, so their
    mirror image is one index off that of the cells.
    """
    x = np.flip(x, axis)
    return np.roll(x, -1, axis) if on_faces else x


def _mirror_state(fields: dict[str, np.ndarray], axis: str) -> dict[str, np.ndarray]:
    k = -1 if axis == "x" else -2
    normal, along = ("u", "v") if axis == "x" else ("v", "u")
    return {
        "h": _mirrored(fields["h"], k, False),
        normal: -_mirrored(fields[normal], k, True),
        along: _mirrored(fields[along], k, False),
    }


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("has_island", [False, True], ids=["nomask", "island"])
@pytest.mark.parametrize("scheme", [Upwind(1), WENO(5)], ids=["upwind1", "weno5"])
@pytest.mark.parametrize("name", list(MODELS))
def test_a_mirrored_state_has_the_mirrored_tendency(
    name, scheme, has_island, axis
) -> None:
    state_class = MODELS[name][1]
    wet = _island() if has_island else None
    mirrored_wet = None if wet is None else np.flip(wet, 1 if axis == "x" else 0)
    with float64():
        model = _box(name, F0, wet, scheme=scheme, wind_amplitude=0.0)
        # A mirror turns the rotation round.
        mirror = _box(name, -F0, mirrored_wet, scheme=scheme, wind_amplitude=0.0)

        def tendency(model: Any, fields: dict[str, np.ndarray]) -> dict[str, Any]:
            state = state_class(**{k: jnp.asarray(x) for k, x in fields.items()})
            state = model.apply_boundary_conditions(state)
            out = model.vector_field(0.0, state)
            return {k: np.array(getattr(out, k)) for k in "huv"}

        fields = _random_state(name, wet)
        # The walls are closed, before they are mirrored.
        fields["u"][..., [0, -2, -1]] = 0.0
        fields["v"][..., [0, -2, -1], :] = 0.0
        here = tendency(model, fields)
        there = tendency(mirror, _mirror_state(fields, axis))
    here = _mirror_state(here, axis)
    inside = {
        "h": (..., slice(1, -1), slice(1, -1)),
        "u": (..., slice(1, -1), slice(1, -2)),
        "v": (..., slice(1, -2), slice(1, -1)),
    }
    for field, where in inside.items():
        scale = np.abs(here[field][where]).max()
        difference = np.abs(here[field][where] - there[field][where]).max()
        assert difference <= 1e-12 * scale, (field, difference / scale)


@pytest.mark.parametrize("name", list(MODELS))
def test_the_coriolis_term_of_a_uniform_flow(name) -> None:
    """Half of it beside a wall, where the mean of `u` and zero turns it."""
    state_class, layers = MODELS[name][1:]
    shape = (*layers, NY + 2, NX + 2)
    with float64():
        model = _box(name, F0, None, wind_amplitude=0.0)
        model = eqx.tree_at(
            lambda m: m.params.lateral_viscosity, model, jnp.asarray(0.0)
        )
        state = state_class(
            h=jnp.full(shape, 500.0), u=jnp.full(shape, 0.1), v=jnp.zeros(shape)
        )
        state = model.apply_boundary_conditions(state)
        dv_dt = np.asarray(model.vector_field(0.0, state).v)[..., 6, 1:-1]
    expected = np.full(NX, -0.1 * F0)
    expected[[0, -1]] /= 2.0
    np.testing.assert_allclose(
        dv_dt, np.broadcast_to(expected, dv_dt.shape), rtol=1e-12
    )


def test_the_potential_vorticity_on_walls_and_coasts() -> None:
    """`f / h` on them, with the mean thickness of the wet cells, at every one."""
    wet = _island()
    with float64():
        model = _box("nonlinear", F0, wet)
        fields = _random_state("nonlinear", wet)
        # What is on land is not read.
        fields["h"][4:7, 6:10] = np.nan
        state = model.apply_boundary_conditions(
            NonlinearSW2DState(**{k: jnp.asarray(x) for k, x in fields.items()})
        )
        q = np.asarray(model._potential_vorticity(state.u, state.v, state.h))
        h = np.asarray(state.h)
    # the south-western corner of the box, a wall, the corner of the island and
    # its side
    assert q[0, 0] == pytest.approx(F0 / h[1, 1])
    assert q[0, 5] == pytest.approx(F0 / h[1, 5:7].mean())
    assert q[NY, 5] == pytest.approx(F0 / h[NY, 5:7].mean())
    assert q[5, 0] == pytest.approx(F0 / h[5:7, 1].mean())
    assert q[3, 5] == pytest.approx(F0 / np.mean([h[3, 5], h[3, 6], h[4, 5]]))
    assert q[3, 7] == pytest.approx(F0 / h[3, 7:9].mean())
    # inside the island
    assert q[5, 7] == 0.0


@pytest.mark.parametrize("name", list(MODELS))
def test_no_friction_along_a_straight_coast(name) -> None:
    """A uniform flow along a channel of land has no viscous tendency."""
    state_class, layers = MODELS[name][1:]
    wet = np.ones((NY + 2, NX + 2), dtype=bool)
    wet[:4] = wet[-4:] = False
    with float64():
        viscous, inviscid = (
            MODELS[name][0].create(
                nx=NX,
                ny=NY,
                Lx=1e6,
                Ly=8e5,
                f0=F0,
                beta=0.0,
                lateral_viscosity=nu,
                mask=Mask2D.from_mask(wet),
            )
            for nu in (1e4, 0.0)
        )
        shape = (*layers, NY + 2, NX + 2)
        state = state_class(
            h=jnp.full(shape, 500.0),
            u=0.1 * jnp.broadcast_to(jnp.asarray(wet), shape),
            v=jnp.zeros(shape),
        )
        state = viscous.apply_boundary_conditions(state)
        difference = np.asarray(
            viscous.vector_field(0.0, state).u - inviscid.vector_field(0.0, state).u
        )
    np.testing.assert_array_equal(difference, 0.0)


@pytest.mark.parametrize("bc", ["periodic", "wall"])
@pytest.mark.parametrize("at", ["U", "V"])
def test_the_laplacian_of_an_all_water_mask(at, bc) -> None:
    """It is gridax's Laplacian, on the plane and on the sphere.

    But on the eastern and northern walls, which the models hold at rest.
    """
    rng = np.random.default_rng(0)
    water = Mask2D.from_mask(np.ones((NY + 2, NX + 2), dtype=bool))
    with float64():
        for model, state_class in (
            (
                NonlinearShallowWater2D.create(nx=NX, ny=NY, bc=bc, mask=water),
                NonlinearSW2DState,
            ),
            (SphericalSWM.create(nx=NX, ny=NY, mask=water), SphericalSWMState),
        ):
            state = model.apply_boundary_conditions(
                state_class(*jnp.asarray(rng.normal(size=(3, NY + 2, NX + 2))))
            )
            velocity = state.u if at == "U" else state.v
            got = np.asarray(model._laplacian(velocity, at))[1:-2, 1:-2]
            want = np.asarray(laplacian(velocity, model.grid, at=at))[1:-2, 1:-2]
            np.testing.assert_allclose(
                got, want, rtol=0.0, atol=1e-12 * np.abs(want).max()
            )


def test_the_gradient_is_finite_with_no_thickness_on_land() -> None:
    wet = _island()
    model = _box("nonlinear", F0, wet)
    fields = _random_state("nonlinear", wet)
    state = NonlinearSW2DState(
        h=jnp.asarray(np.where(wet, fields["h"], 0.0)),
        u=jnp.asarray(fields["u"]),
        v=jnp.asarray(fields["v"]),
    )

    def loss(state: Any) -> jax.Array:
        out = model.vector_field(0.0, model.apply_boundary_conditions(state))
        return jnp.sum(out.u**2) + jnp.sum(out.v**2)

    grads = jax.grad(loss)(state)
    for field in (grads.h, grads.u, grads.v):
        assert bool(jnp.isfinite(field).all())


def test_the_sphere_is_symmetric_about_the_equator() -> None:
    rng = np.random.default_rng(0)
    fields = {
        "h": 500.0 + rng.normal(size=(NY + 2, NX + 2)),
        "u": 0.1 * rng.normal(size=(NY + 2, NX + 2)),
        "v": 0.1 * rng.normal(size=(NY + 2, NX + 2)),
    }
    fields["v"][[0, -2, -1]] = 0.0
    with float64():
        model = SphericalSWM.create(
            nx=NX, ny=NY, wind_profile="none", lateral_viscosity=1e6
        )

        def tendency(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            state = SphericalSWMState(**{k: jnp.asarray(x) for k, x in fields.items()})
            out = model.vector_field(0.0, model.apply_boundary_conditions(state))
            return {k: np.array(getattr(out, k)) for k in "huv"}

        here = _mirror_state(tendency(fields), "y")
        there = tendency(_mirror_state(fields, "y"))
    for field in "huv":
        rows = slice(1, -2) if field == "v" else slice(1, -1)
        expected = here[field][rows, 1:-1]
        difference = np.abs(expected - there[field][rows, 1:-1]).max()
        assert difference <= 1e-12 * np.abs(expected).max(), field
