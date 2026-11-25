"""The states a run returns have the model's boundary conditions."""

from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sesmax_core import ScaledModel, Scales
from sesmax_core.testing import LinearShallowWater2D, LinearSW2DState
from sesmax_core.train import rollout


@pytest.fixture
def model() -> LinearShallowWater2D:
    return LinearShallowWater2D.create(
        nx=12, ny=10, bc="wall", lateral_viscosity=1e3, bottom_drag=1e-6
    )


@pytest.fixture
def state(model) -> LinearSW2DState:
    rng = np.random.default_rng(0)
    h, u, v = (jnp.asarray(rng.normal(size=model.grid.shape)) for _ in range(3))
    return LinearSW2DState(h=h, u=0.1 * u, v=0.1 * v)


def _assert_filled(model, states) -> None:
    filled = jax.vmap(model.apply_boundary_conditions)(states)
    for got, want in zip(jax.tree.leaves(states), jax.tree.leaves(filled), strict=True):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "saveat",
    [
        dfx.SaveAt(t1=True),
        dfx.SaveAt(t0=True, t1=True),
        dfx.SaveAt(ts=jnp.array([100.0, 300.0])),
    ],
    ids=["t1", "t0 and t1", "ts"],
)
def test_every_saved_state_is_filled(model, state, saveat) -> None:
    solution = model.integrate(state, t0=0.0, t1=300.0, dt=30.0, saveat=saveat)
    _assert_filled(model, solution.ys)


def test_a_step_and_a_rollout_return_filled_states(model, state) -> None:
    out = model.step(state, 30.0)
    _assert_filled(model, jax.tree.map(lambda x: x[None], out))
    _assert_filled(model, rollout(model, state, dt=30.0, n_steps=3))


def test_the_interior_is_the_integrator_s(model, state) -> None:
    """Filling what is saved changes nothing that is integrated."""
    bare = dfx.diffeqsolve(
        model.build_terms(),
        dfx.Tsit5(),
        t0=0.0,
        t1=300.0,
        dt0=30.0,
        y0=model.apply_boundary_conditions(state),
    ).ys
    saved = model.integrate(state, t0=0.0, t1=300.0, dt=30.0).ys
    inner = (slice(None), slice(1, -2), slice(1, -2))
    for field in ("h", "u", "v"):
        got, want = getattr(saved, field)[inner], getattr(bare, field)[inner]
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)


def test_a_save_function_sees_filled_states(model, state) -> None:
    saveat = dfx.SaveAt(t1=True, fn=lambda t, y, args: y.h[0] - y.h[1])
    out = model.integrate(state, t0=0.0, t1=300.0, dt=30.0, saveat=saveat).ys
    np.testing.assert_array_equal(out, 0.0)


def test_several_saves_and_the_dense_solution(model, state) -> None:
    saveat = dfx.SaveAt(
        subs=[dfx.SubSaveAt(t1=True), dfx.SubSaveAt(t0=True)], dense=True
    )
    solution = model.integrate(state, t0=0.0, t1=300.0, dt=30.0, saveat=saveat)
    for ys in solution.ys:
        _assert_filled(model, ys)
    assert solution.interpolation is not None
    dense = model.integrate(
        state, t0=0.0, t1=300.0, dt=30.0, saveat=dfx.SaveAt(dense=True)
    )
    assert dense.ys is None


@pytest.mark.parametrize("adjoint", [dfx.BacksolveAdjoint(), dfx.DirectAdjoint()])
def test_the_other_adjoints(model, state, adjoint) -> None:
    out = model.integrate(
        state, t0=0.0, t1=60.0, dt=30.0, adjoint=adjoint, max_steps=16
    )
    _assert_filled(model, out.ys)


def test_a_scaled_model_fills_in_its_own_coordinates(model, state) -> None:
    scales = Scales.inertial(L=1e6, f0=1e-4, H=100.0, rossby=0.1)
    scaled = ScaledModel.from_scales(model, scales, LinearSW2DState)
    out = scaled.integrate(scaled.transform.forward(state), t0=0.0, t1=0.03, dt=0.003)
    filled = jax.vmap(scaled.apply_boundary_conditions)(out.ys)
    for got, want in zip(jax.tree.leaves(out.ys), jax.tree.leaves(filled), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_a_gradient_flows_through_the_saved_state(model, state) -> None:
    def loss(m):
        return jnp.sum(m.integrate(state, t0=0.0, t1=60.0, dt=30.0).ys.u ** 2)

    gradient = eqx.filter_grad(loss)(model)
    assert float(gradient.params.lateral_viscosity) != 0.0
