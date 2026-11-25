"""What travels through the stages of a run."""

from __future__ import annotations

import dataclasses
from typing import Any

from jaxtyping import PyTree

from sesmax_experiment._src.monitor import ChunkInfo


@dataclasses.dataclass(frozen=True)
class ChunkFrame:
    """The state after a chunk, with what is known about the chunk.

    Args:
        state: The model's state at ``info.t1``.
        info: The chunk: its index, times, wall time and solver statistics.
        metrics: What the monitors measured, as ``"monitor.metric" -> value``.
        messages: What the monitors said, as ``"[monitor] message"``.
    """

    state: PyTree
    info: ChunkInfo
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    messages: tuple[str, ...] = ()


class RunStopped(RuntimeError):
    """A monitor stopped the run.

    Args:
        reason: The monitor and its reason.
        frame: The frame it stopped at, which no later stage has seen.
        resumable: Whether the state is good, so the run may be continued from it.
    """

    def __init__(
        self, reason: str, frame: ChunkFrame, *, resumable: bool = False
    ) -> None:
        self.reason = reason
        self.frame = frame
        self.resumable = resumable
        info = frame.info
        super().__init__(
            f"run stopped in chunk {info.index + 1}/{info.n_chunks}, between "
            f"t={info.t0:g} and t={info.t1:g}: {reason}"
        )

    def __reduce__(self) -> Any:
        return type(self), (self.reason, self.frame), {"resumable": self.resumable}
