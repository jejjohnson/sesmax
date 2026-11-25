"""Public surface for sesmax-experiment's orbax checkpointer.

:class:`SimulationCheckpointer` saves and restores a run's state and a
model's parameters with orbax-checkpoint, which the ``[checkpoint]`` extra
installs (``pip install sesmax-experiment[checkpoint]``); orbax is imported only
when a checkpoint is written or read.
"""

from __future__ import annotations

from sesmax_experiment._src.checkpoint import SimulationCheckpointer


__all__ = ["SimulationCheckpointer"]
