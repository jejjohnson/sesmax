"""Checkpoints of a model with constrained parameters.

Moved from sesmax-core's ``test_paramax_linear_swm.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import paramax
import pytest

from sesmax_core import positive
from sesmax_core.testing import LinearShallowWater2D, LinearSW2DState
from sesmax_experiment.checkpoint import SimulationCheckpointer


pytest.importorskip("orbax.checkpoint")


def _sw_state(field):
    """A linear shallow-water state with ``field`` in every component."""
    return LinearSW2DState(h=field, u=field, v=field)


class TestConstrainedModelsRoundTripThroughACheckpoint:
    """Orbax has no handler for the transform callable in ``Parameterize``."""

    @staticmethod
    def _constrained_model():
        model = LinearShallowWater2D.create(nx=8, ny=8)
        return eqx.tree_at(lambda m: m.params.lateral_viscosity, model, positive(100.0))

    def test_it_saves_and_restores(self, tmp_path):
        model = self._constrained_model()
        state = _sw_state(jnp.zeros((10, 10)))
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path), checkpoint_interval=1
        )
        ckpt.save(1, state, model)
        _, params, step = ckpt.restore(1, state, model)

        assert step == 1
        assert float(paramax.unwrap(params).lateral_viscosity) == pytest.approx(
            100.0, rel=1e-5
        )

    def test_the_restored_parameter_is_still_a_wrapper(self, tmp_path):
        model = self._constrained_model()
        state = _sw_state(jnp.zeros((10, 10)))
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path), checkpoint_interval=1
        )
        ckpt.save(1, state, model)
        _, params, _ = ckpt.restore(1, state, model)
        assert isinstance(params.lateral_viscosity, paramax.Parameterize)

    def test_a_plain_model_still_round_trips(self, tmp_path):
        model = LinearShallowWater2D.create(nx=8, ny=8)
        state = _sw_state(jnp.zeros((10, 10)))
        ckpt = SimulationCheckpointer(
            checkpoint_dir=str(tmp_path), checkpoint_interval=1
        )
        ckpt.save(2, state, model)
        _, params, step = ckpt.restore(2, state, model)
        assert step == 2
        np.testing.assert_allclose(
            np.asarray(params.lateral_viscosity),
            np.asarray(model.params.lateral_viscosity),
        )
