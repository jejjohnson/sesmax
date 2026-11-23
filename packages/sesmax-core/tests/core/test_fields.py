"""States whose fields know their point: `gridax.CField` leaves.

Every test problem runs the same with array fields and with located ones, and a
sum of tendencies on different points raises.
"""

from __future__ import annotations

import dataclasses
import re

import diffrax as dfx
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest
from gridax import CField, Mask1D, Mask2D, StaggerMismatchError
from gridax.schemes import WENO

from sesmax_core._src.fields import each, map_fields, placed, values
from sesmax_core._src.terms import Scaled, TermFn
from sesmax_core._src.testing.pde1d import burgers, diffusion, linear_convection
from sesmax_core._src.testing.pde1d.nonlinear_convection import (
    NonlinearConvection1D,
    NonlinearConvection1DState,
)
from sesmax_core._src.testing.pde2d import (
    burgers as burgers2d,
    diffusion as diffusion2d,
    linear_convection as linear2d,
    navier_stokes as ns,
    nonlinear_convection as nonlinear2d,
)
from sesmax_core._src.testing.pde2d.burgers_terms import Burgers2DTermModel
from sesmax_core._src.testing.swm import linear_1d, linear_2d, nonlinear_1d
from sesmax_core._src.testing.swm.linear_swm_terms import LinearSWM2DTermModel


def _wet_1d():
    wet = np.ones(22, dtype=bool)
    wet[[0, -1, 9]] = False
    return Mask1D.from_mask(wet)


def _wet_2d():
    wet = np.ones((14, 18), dtype=bool)
    wet[[0, -1]] = False
    wet[:, [0, -1]] = False
    wet[5:8, 6:9] = False
    return Mask2D.from_mask(wet)


def _problems():
    """``name -> (model, state class, point of each field)``."""
    one, two = {"nx": 20}, {"nx": 16, "ny": 12}
    sw = {"g": 1.0, "f0": 0.1, "lateral_viscosity": 0.01, "bottom_drag": 0.01}
    for masked in (False, True):
        m1 = {"mask": _wet_1d()} if masked else {}
        m2 = {"mask": _wet_2d()} if masked else {}
        tag = "-masked" if masked else ""
        yield (
            "burgers1d" + tag,
            burgers.Burgers1D.create(**one, **m1, scheme=WENO(5)),
            burgers.Burgers1DState,
            "T",
        )
        yield (
            "diffusion1d" + tag,
            diffusion.Diffusion1D.create(**one, **m1),
            diffusion.Diffusion1DState,
            "T",
        )
        yield (
            "linear1d" + tag,
            linear_convection.LinearConvection1D.create(**one, **m1),
            linear_convection.LinearConvection1DState,
            "T",
        )
        yield (
            "nonlinear1d" + tag,
            NonlinearConvection1D.create(**one, **m1),
            NonlinearConvection1DState,
            "T",
        )
        yield (
            "swe1d" + tag,
            linear_1d.LinearShallowWater1D.create(**one, **sw, **m1),
            linear_1d.LinearSW1DState,
            "TUT",
        )
        yield (
            "nonlinear_swe1d" + tag,
            nonlinear_1d.NonlinearShallowWater1D.create(**one, **sw, **m1),
            nonlinear_1d.NonlinearSW1DState,
            "TUT",
        )
        yield (
            "burgers2d" + tag,
            burgers2d.Burgers2D.create(**two, **m2),
            burgers2d.Burgers2DState,
            "TT",
        )
        yield (
            "burgers2d_terms" + tag,
            Burgers2DTermModel.create(**two, **m2),
            burgers2d.Burgers2DState,
            "TT",
        )
        yield (
            "diffusion2d" + tag,
            diffusion2d.Diffusion2D.create(**two, **m2),
            diffusion2d.Diffusion2DState,
            "T",
        )
        yield (
            "linear2d" + tag,
            linear2d.LinearConvection2D.create(**two, **m2),
            linear2d.LinearConvection2DState,
            "T",
        )
        yield (
            "nonlinear2d" + tag,
            nonlinear2d.NonlinearConvection2D.create(**two, **m2),
            nonlinear2d.NonlinearConvection2DState,
            "TT",
        )
        for bc in ("periodic", "wall"):
            yield (
                f"swe2d-{bc}" + tag,
                linear_2d.LinearShallowWater2D.create(
                    **two, **sw, Lx=8.0, Ly=6.0, H0=1.0, beta=0.02, bc=bc, **m2
                ),
                linear_2d.LinearSW2DState,
                "TUV",
            )
    for bc in ("periodic", "wall"):
        yield (
            f"swe2d_terms-{bc}",
            LinearSWM2DTermModel.create(**two, **sw, Lx=8.0, Ly=6.0, H0=1.0, bc=bc),
            linear_2d.LinearSW2DState,
            "TUV",
        )
    for problem in ("cavity", "periodic", "channel"):
        yield (
            f"ns-{problem}",
            ns.IncompressibleNS2D.create(**two, problem=problem),
            ns.NSVorticityState,
            "T",
        )


PROBLEMS = list(_problems())


def _states(model, state_cls, points):
    """The same random state with array fields and with located ones."""
    rng = np.random.default_rng(0)
    grid = model.grid
    shape = grid.shape
    names = [f.name for f in dataclasses.fields(state_cls)][: len(points)]
    arrays = {
        n: jnp.asarray(0.1 * rng.standard_normal(shape) + (n == "h")) for n in names
    }
    located = {
        n: CField(arrays[n], grid.axes, loc=p)
        for n, p in zip(names, points, strict=True)
    }
    return state_cls(**arrays), state_cls(**located)


@pytest.mark.parametrize(
    ("model", "state_cls", "points"),
    [p[1:] for p in PROBLEMS],
    ids=[p[0] for p in PROBLEMS],
)
def test_trajectories_are_identical(model, state_cls, points):
    plain, located = _states(model, state_cls, points)

    def step(y):
        filled = model.apply_boundary_conditions(y)
        rate = model.vector_field(0.0, filled)
        return jax.tree.map(lambda a, b: a + 1e-3 * b, filled, rate)

    run = jax.jit(step)
    # one program, up to the names of its results
    programs = [
        re.sub(r'"result[^"]*"', '"result"', run.lower(y).as_text())
        for y in (plain, located)
    ]
    assert programs[0] == programs[1]
    # XLA may round two runs of one program on the CPU differently
    for _ in range(5):
        plain, located = run(plain), run(located)
    for field, point in zip(dataclasses.fields(state_cls), points, strict=False):
        a, b = getattr(plain, field.name), getattr(located, field.name)
        assert b.loc == point
        np.testing.assert_allclose(b.data, a, rtol=1e-5, atol=1e-6)


def test_integrate_and_flatten_a_located_state():
    model = linear_2d.LinearShallowWater2D.create(nx=8, ny=6, Lx=8.0, Ly=6.0, g=1.0)
    plain, located = _states(model, linear_2d.LinearSW2DState, "TUV")
    kw = {
        "t0": 0.0,
        "t1": 0.1,
        "dt": 0.02,
        "saveat": dfx.SaveAt(ts=jnp.array([0.05, 0.1])),
    }
    a, b = model.integrate(plain, **kw).ys, model.integrate(located, **kw).ys
    for name in ("h", "u", "v"):
        np.testing.assert_allclose(
            getattr(b, name).data, getattr(a, name), rtol=1e-6, atol=1e-8
        )
    assert b.u.loc == "U"
    # the flattening of the DA bridge sees the arrays only
    flat_a, _ = jax.flatten_util.ravel_pytree(plain)
    flat_b, back = jax.flatten_util.ravel_pytree(located)
    np.testing.assert_array_equal(flat_a, flat_b)
    assert back(flat_b).v.loc == "V"


class TestSum:
    @staticmethod
    def _term(loc):
        def tendency(t, state, args):
            return {"u": CField(2.0 * values(state["u"]), ("y", "x"), loc=loc)}

        return TermFn(tendency)

    state = {"u": CField(jnp.ones((4, 5)), ("y", "x"), loc="U")}  # noqa: RUF012

    def test_adds_fields_on_one_point(self):
        out = (self._term("U") + 3.0 * self._term("U"))(0.0, self.state)
        assert out["u"].loc == "U"
        np.testing.assert_array_equal(out["u"].data, 8.0)

    def test_another_point_raises_at_trace_time(self):
        total = self._term("U") + Scaled(self._term("T"), 3.0)
        with pytest.raises(StaggerMismatchError):
            total(0.0, self.state)
        with pytest.raises(StaggerMismatchError):
            jax.eval_shape(lambda s: total(0.0, s), self.state)

    def test_arrays_are_added_as_before(self):
        term = TermFn(lambda t, s, args: jax.tree.map(lambda x: 2.0 * x, s))
        out = (term + term)(0.0, {"a": jnp.ones(3), "b": (jnp.ones(2),)})
        np.testing.assert_array_equal(out["a"], 4.0)
        np.testing.assert_array_equal(out["b"][0], 4.0)


class TestHelpers:
    field = CField(jnp.arange(6.0).reshape(2, 3), ("y", "x"), loc="V")

    def test_each(self):
        np.testing.assert_array_equal(each(jnp.add, jnp.ones(2), 1.0), 2.0)
        out = each(lambda a, b: a + b, self.field, self.field)
        assert out.loc == "V"
        with pytest.raises(StaggerMismatchError):
            each(jnp.add, self.field, CField(jnp.ones((2, 3)), ("y", "x"), loc="T"))

    def test_placed_is_a_field_next_to_a_field(self):
        column = jnp.ones((2, 3))
        assert placed(column, jnp.ones((2, 3)), "T") is column
        members = CField(jnp.ones((4, 2, 3)), ("member", "y", "x"), loc="U")
        assert placed(column, members, "T").dims == ("y", "x")

    def test_map_fields_mixes_arrays_and_fields(self):
        out = map_fields(lambda x, s: x * s, {"u": self.field}, {"u": 2.0})
        assert out["u"].loc == "V"
        np.testing.assert_array_equal(out["u"].data, 2.0 * self.field.data)


def test_a_dataset_keeps_the_points():
    pytest.importorskip("xarray")
    from sesmax_core._src.io.xarray import dataset_to_state, state_to_dataset

    model = linear_2d.LinearShallowWater2D.create(nx=8, ny=6)
    _, located = _states(model, linear_2d.LinearSW2DState, "TUV")
    back = dataset_to_state(
        state_to_dataset(located), state_class=linear_2d.LinearSW2DState
    )
    assert (back.h.loc, back.u.loc, back.v.loc) == ("T", "U", "V")
    np.testing.assert_array_equal(back.u.data, located.u.data)
