"""Public surface for sesmax-core's orbax checkpointer.

:class:`SimulationCheckpointer` saves and restores a run's state and a
model's parameters with orbax-checkpoint, which the ``[checkpoint]`` extra
installs (``pip install sesmax-core[checkpoint]``); orbax is imported only
when a checkpoint is written or read.

It is also importable from the top-level ``sesmax_core``.
"""

from __future__ import annotations

from sesmax_core._src.checkpoint import SimulationCheckpointer


__all__ = ["SimulationCheckpointer"]
