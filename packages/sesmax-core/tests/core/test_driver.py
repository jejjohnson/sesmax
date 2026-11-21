"""solve_pde, the PDE driver (sesmax#127).

The first six tests moved from gridax's ``tests/test_diffrax_solvers.py``
(``TestSolveOceanPDE`` and ``TestSolveOceanPDEKwargs`` at gridax ``c69768c``)
with ``solve_ocean_pde(..., bc_fn=)`` renamed to
``solve_pde(..., tendency_bc=)``; the rest cover the state correction.
"""

from __future__ import annotations

import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from integrix.march import RK3SSP, RK4Classic

from sesmax_core import solve_pde


def _decay_rhs(t, y, args):
    return -y


# --- moved from gridax ------------------------------------------------------


def test_basic_usage():
    sol = solve_pde(_decay_rhs, RK3SSP(), y0=1.0, t0=0.0, t1=1.0, dt0=0.01)
    np.testing.assert_allclose(float(sol.ys[0]), math.exp(-1.0), rtol=1e-4)


def test_with_tendency_bc():
    """The tendency correction is applied to the tendency."""

    def tendency_bc(dydt):
        return dydt

    sol = solve_pde(_decay_rhs, RK3SSP(), 1.0, 0.0, 0.1, 0.01, tendency_bc=tendency_bc)
    assert jnp.isfinite(sol.ys[0])


def test_tendency_bc_modifies_tendency():
    """A correction that zeroes the tendency freezes the state."""
    sol = solve_pde(
        _decay_rhs, RK3SSP(), 1.0, 0.0, 1.0, 0.01, tendency_bc=lambda d: 0.0 * d
    )
    np.testing.assert_allclose(float(sol.ys[0]), 1.0, atol=1e-10)


def test_with_saveat():
    """saveat reaches diffeqsolve."""
    ts = jnp.linspace(0.0, 1.0, 6)
    sol = solve_pde(
        _decay_rhs, RK4Classic(), 1.0, 0.0, 1.0, 0.01, saveat=dfx.SaveAt(ts=ts)
    )
    assert sol.ys.shape == (6,)


def test_long_fixed_step_run():
    """t1 = 10, dt0 = 1e-3 is 10^4 steps: more than diffrax's default 4096."""
    sol = solve_pde(_decay_rhs, RK3SSP(), jnp.array(1.0), 0.0, 10.0, 1e-3)
    # gridax checked 1e-8 with x64 on; this suite runs in float32
    rtol = 1e-8 if sol.ys.dtype == jnp.float64 else 1e-4
    np.testing.assert_allclose(sol.ys[0], math.exp(-10.0), rtol=rtol)
    assert int(sol.stats["num_steps"]) == 10_000


def test_kwargs_reach_diffeqsolve():
    sol = solve_pde(
        _decay_rhs,
        RK3SSP(),
        jnp.array(1.0),
        0.0,
        1.0,
        0.1,
        stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-10),
        adjoint=dfx.DirectAdjoint(),
    )
    assert int(sol.stats["num_steps"]) != 10  # the controller chose the steps
    np.testing.assert_allclose(sol.ys[0], math.exp(-1.0), rtol=1e-6)
    with pytest.raises(Exception, match="maximum number of solver steps"):
        solve_pde(_decay_rhs, RK3SSP(), jnp.array(1.0), 0.0, 1.0, 0.01, max_steps=5)


# --- the state correction ---------------------------------------------------


def test_state_and_tendency_bc_act_at_every_stage():
    # f(y) = -y on a 3-vector; B pins y₀ = 2 before each evaluation, P zeroes
    # the last tendency. The middle entry then decays as e^{-t} on its own,
    # the first entry's tendency is -2 at every stage (so y₀(1) = 1 - 2), and
    # the last never moves.
    def state_bc(y):
        return y.at[0].set(2.0)

    def tendency_bc(dydt):
        return dydt.at[-1].set(0.0)

    sol = solve_pde(
        _decay_rhs,
        RK3SSP(),
        jnp.ones(3),
        0.0,
        1.0,
        0.01,
        state_bc=state_bc,
        tendency_bc=tendency_bc,
    )
    y1 = sol.ys[0]
    np.testing.assert_allclose(y1[0], 1.0 - 2.0, rtol=1e-5)
    np.testing.assert_allclose(y1[1], math.exp(-1.0), rtol=1e-6)
    np.testing.assert_allclose(y1[2], 1.0, rtol=0, atol=0)


def test_without_corrections_equals_diffeqsolve_bit_for_bit():
    y0 = jnp.linspace(0.5, 1.5, 4)
    sol = solve_pde(_decay_rhs, RK3SSP(), y0, 0.0, 1.0, 0.01)
    reference = dfx.diffeqsolve(
        dfx.ODETerm(_decay_rhs),
        RK3SSP(),
        t0=0.0,
        t1=1.0,
        dt0=0.01,
        y0=y0,
        saveat=dfx.SaveAt(t1=True),
        max_steps=116,
    )
    assert np.array_equal(sol.ys, reference.ys)


class _Decay(eqx.Module):
    rate: jax.Array

    def __call__(self, t, y, args):
        return -self.rate * y


def test_gradient_reaches_the_parameters_of_an_rhs_module():
    # the vector field is a pytree, so filter_grad sees the module's leaves
    @eqx.filter_grad
    def grad(rhs):
        return solve_pde(rhs, RK4Classic(), jnp.array(1.0), 0.0, 1.0, 0.01).ys[0]

    rate = 0.7
    # d/dk e^{-k} = -e^{-k}
    np.testing.assert_allclose(
        grad(_Decay(jnp.array(rate))).rate, -math.exp(-rate), rtol=1e-6
    )


def test_traced_times_keep_diffraxs_default():
    @jax.jit
    def run(t1):
        return solve_pde(_decay_rhs, RK3SSP(), jnp.array(1.0), 0.0, t1, 0.01).ys[0]

    np.testing.assert_allclose(run(jnp.array(1.0)), math.exp(-1.0), rtol=1e-4)
