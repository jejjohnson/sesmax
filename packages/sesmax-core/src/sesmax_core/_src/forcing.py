"""Forcing protocol for sesmax models."""

from __future__ import annotations

import abc

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from functax.interp import LocalScheme, Policies, SeriesScheme
from integrix.march import SampledPath
from jaxtyping import Array


class ForcingProtocol(eqx.Module):
    """Base class for forcing terms.

    Forcing objects are callable modules that return a forcing field
    given a time and grid. They compose with sesmax models via the
    ``forcing`` attribute.
    """

    @abc.abstractmethod
    def __call__(self, t: float, grid: eqx.Module) -> Array:
        """Evaluate forcing at time ``t`` on ``grid``."""
        ...


class ConstantForcing(ForcingProtocol):
    """Time-independent forcing field."""

    field: Array

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        return self.field


class NoForcing(ForcingProtocol):
    """Zero forcing (free evolution)."""

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        return jnp.float32(0.0)


class SeasonalWindForcing(ForcingProtocol):
    """Sinusoidal wind forcing with learnable amplitude.

    Produces ``tau0 * cos(omega * t + phase)`` where ``tau0`` is a
    differentiable amplitude field visible to ``jax.grad``.

    Args:
        tau0: Learnable amplitude array (visible to ``jax.grad``).
        omega: Angular frequency, e.g. ``2 * pi / T`` (differentiable leaf).
        phase: Phase offset in radians (differentiable leaf).
    """

    tau0: Array
    omega: Array = eqx.field(converter=jnp.asarray)
    phase: Array = eqx.field(converter=jnp.asarray, default=0.0)

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        """Evaluate seasonal forcing at time ``t``."""
        return self.tau0 * jnp.cos(self.omega * t + self.phase)


class InterpolatedForcing(ForcingProtocol):
    """Forcing from samples (t_k, u_k), read between them as a diffrax path.

    The path is a ``diffrax.LinearInterpolation``, a
    ``diffrax.CubicInterpolation``, or an integrix ``SampledPath`` over any
    functax 1-D scheme, such as a C² spline or a PCHIP that stays between
    neighbouring samples.

    The reconstruction has a kink at each t_k, and a step that straddles one
    loses order. Pass ``step_ts`` to the step control, or take fixed steps
    that divide the sample spacing.

    Args:
        path: A diffrax path built from data.

    Examples:
        River discharge through PCHIP, which cannot go negative:

        >>> import jax.numpy as jnp
        >>> from functax.interp import Pchip
        >>> from sesmax_core.forcing import InterpolatedForcing
        >>> ts = jnp.array([0.0, 1.0, 2.0, 3.0])
        >>> discharge = jnp.array([0.0, 0.0, 4.0, 0.0])
        >>> river = InterpolatedForcing.from_data(ts, discharge, method=Pchip())
        >>> bool(river(0.5, None) >= 0.0)
        True
        >>> river.step_ts.tolist()
        [0.0, 1.0, 2.0, 3.0]
    """

    path: dfx.AbstractPath

    def __call__(self, t: float, grid: eqx.Module) -> Array:
        """Evaluate interpolated forcing at time ``t``."""
        return self.path.evaluate(t)

    @property
    def step_ts(self) -> Array:
        """The sample times t_k; pass them to a step controller's ``step_ts``.

        Every path that `from_data` builds carries them; a path of another
        type needs a ``ts`` of its own.
        """
        return self.path.ts

    @staticmethod
    def from_data(
        ts: Array,
        values: Array,
        method: str | LocalScheme | SeriesScheme = "linear",
        *,
        outside: str = "extrapolate",
    ) -> InterpolatedForcing:
        """Build an ``InterpolatedForcing`` from tabulated data.

        Args:
            ts: Time coordinates with shape ``(T,)``.
            values: Forcing values with shape ``(T, ...)``.
            method: ``"linear"`` (default) or ``"cubic"`` for diffrax's paths,
                whose cubic takes its slopes from the backward secants, or a
                functax 1-D scheme (``functax.interp.Linear()``,
                ``CubicSpline()``, ``Pchip()``, ...).
            outside: What a scheme does past the samples: ``"extrapolate"``
                (default, as diffrax's paths do), ``"clamp"`` to hold the end
                values, or ``"nan"``. A cubic scheme's end piece can leave the
                data's range. Diffrax's two methods always extrapolate.

        Returns:
            An ``InterpolatedForcing`` instance.

        Raises:
            ValueError: If ``method`` is an unknown string, if ``outside`` is
                set for a diffrax method, or if ``method`` is neither a string
                nor a functax scheme.
        """
        if not isinstance(method, str):
            policies = Policies(outside=outside)
            path = SampledPath.fit(method, ts, values, policies=policies)
            return InterpolatedForcing(path=path)
        if outside != "extrapolate":
            msg = f"method={method!r} always extrapolates, got outside={outside!r}"
            raise ValueError(msg)
        if method == "linear":
            path = dfx.LinearInterpolation(ts=ts, ys=values)
        elif method == "cubic":
            coeffs = dfx.backward_hermite_coefficients(ts=ts, ys=values)
            path = dfx.CubicInterpolation(ts=ts, coeffs=coeffs)
        else:
            msg = f"Unknown method {method!r}, expected 'linear' or 'cubic'"
            raise ValueError(msg)
        return InterpolatedForcing(path=path)
