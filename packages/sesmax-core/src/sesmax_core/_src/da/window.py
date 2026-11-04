"""A control through a whole model run to the stacked observations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import vardax
from jaxtyping import Array, Float, PyTree

from sesmax_core._src.model import Model


class WindowOperator(eqx.Module):
    """A control through a whole model run to the stacked observations.

    G(theta) = vec[H x(t_k; theta)], for k = 1, ..., N_T, where theta is the
    control, x(t_k; theta) the model state at save time t_k and H the observation
    operator. The stack runs over time first, then over each time's observations.

    The model integrates the window itself, so a time-dependent forcing sees the
    right time at every step, which ``VardaxForwardModel`` cannot offer. The
    inverse problem has the 3D-Var cost, over the control and not the state:

    J(theta) = 1/2 ||theta - mu||^2_(Sigma_theta^-1)
    + 1/2 ||y - G(theta)||^2_(Sigma_eps^-1)

    On a chaotic model a long window can leave the minimiser far above the truth's
    cost, so compare the two costs in a twin experiment before trusting a window.

    Algorithm:
        1. Build the model for this control with ``build_model``.
        2. Integrate ``initial_state`` from ``t0`` to the last save time.
        3. Apply ``obs_op`` to the state at each save time and flatten.

    Attributes:
        build_model: Maps the control theta to the model, closing over the rest.
        initial_state: State at ``t0``.
        obs_op: The observation operator H, from one state to a 1D array.
        save_times: Increasing observation times t_k, each at or after ``t0``.
        dt: Initial integrator step.
        t0: Start of the window.
        integrate_kwargs: Passed to ``Model.integrate``, such as ``solver`` and
            ``max_steps``.

    Examples:
        A parameter as the control, with two times of three observations:

        >>> import jax.numpy as jnp
        >>> from sesmax_core.testing import L96State, Lorenz96
        >>> window = WindowOperator(
        ...     build_model=lambda control: Lorenz96.create(F=control[0]),
        ...     initial_state=L96State(x=jnp.linspace(-2.0, 2.0, 12)),
        ...     obs_op=lambda state: state.x[::4],
        ...     save_times=jnp.array([0.1, 0.2]),
        ...     dt=0.05,
        ... )
        >>> window(jnp.array([8.0])).shape
        (6,)

        The same map as a vardax observation operator, on a row of samples:

        >>> window.as_obs_op()(jnp.array([[8.0]])).shape
        (1, 6)
    """

    build_model: Callable[[Array], Model]
    initial_state: PyTree
    obs_op: Callable[[PyTree], Array]
    save_times: Float[Array, " time"]
    dt: float = eqx.field(static=True)
    t0: float = eqx.field(static=True, default=0.0)
    integrate_kwargs: dict[str, Any] = eqx.field(default_factory=dict)

    def __call__(self, control: Float[Array, " param"]) -> Float[Array, " obs"]:
        model = self.build_model(control)
        # x(t_k; theta)
        solution = model.integrate(
            self.initial_state,
            t0=self.t0,
            t1=self.save_times[-1],
            dt=self.dt,
            saveat=dfx.SaveAt(ts=self.save_times),
            **self.integrate_kwargs,
        )
        # H x(t_k; theta)
        predicted = jax.vmap(self.obs_op)(solution.ys)
        # vec[H x(t_k; theta)]
        return jnp.ravel(predicted)

    def as_obs_op(self) -> vardax.CallableObs:
        """This map as the observation operator of ``vardax.ThreeDVar``.

        vardax's states and observations have a leading row, so this maps
        ``(1, param)`` to ``(1, obs)``. A diffrax solve defines only a VJP, whose
        loop JAX cannot transpose, so the operator forms its Jacobian once, with
        one VJP per observation. That suits a small control.

        ``vardax.OptimalInterpolation`` applies its operator in forward mode, which
        this one cannot do, so use ``vardax.ThreeDVar``.

        Returns:
            The observation operator.
        """

        def on_a_row(row: Float[Array, "1 param"]) -> Float[Array, "1 obs"]:
            return self(row[0])[None]

        return vardax.CallableObs(fn=on_a_row, jac="bwd", materialise=True)
