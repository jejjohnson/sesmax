"""When a stage of a run is due: predicates on a frame, for ``pipekit_ops.Branch``."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from sesmax_experiment._src.run.frame import ChunkFrame


class Due:
    """A predicate on a :class:`ChunkFrame` that combines with ``|`` and ``&``.

    Both sides of a combination are asked at each chunk, so a schedule that counts
    keeps its count.

    Args:
        due: Whether a stage is due after this chunk.
    """

    def __init__(self, due: Callable[[ChunkFrame], bool]) -> None:
        self._due = due

    def __call__(self, frame: ChunkFrame) -> bool:
        return bool(self._due(frame))

    def __or__(self, other: Due) -> Due:
        return Due(lambda frame: any([self(frame), other(frame)]))

    def __and__(self, other: Due) -> Due:
        return Due(lambda frame: all([self(frame), other(frame)]))


def at_snapshots() -> Due:
    """Due at the snapshots of the run's timeline."""
    return Due(lambda frame: frame.info.is_snapshot)


def every_chunks(n: int) -> Due:
    """Due after every ``n``-th chunk.

    Args:
        n: A positive number of chunks.

    Returns:
        The schedule.

    Raises:
        ValueError: If ``n`` is not positive.

    Examples:
        >>> from sesmax_experiment.run import every_chunks, every_wall_seconds
        >>> checkpoint_due = every_chunks(20) | every_wall_seconds(1800.0)
    """
    if n < 1:
        raise ValueError(f"every_chunks needs a positive number, got {n}")
    return Due(lambda frame: (frame.info.index + 1) % n == 0)


def every_sim_seconds(interval: float, *, t0: float = 0.0) -> Due:
    """Due after the chunk in which a multiple of ``interval`` is reached.

    Args:
        interval: A positive simulation time.
        t0: The time the multiples count from.

    Returns:
        The schedule.

    Raises:
        ValueError: If ``interval`` is not positive.
    """
    if not interval > 0.0:
        raise ValueError(f"every_sim_seconds needs a positive time, got {interval}")

    def count(t: float) -> int:
        return math.floor((t - t0) / interval * (1.0 + 1e-9))

    return Due(lambda frame: count(frame.info.t1) > count(frame.info.t0))


def every_wall_seconds(
    interval: float, *, clock: Callable[[], float] = time.monotonic
) -> Due:
    """Due after the first chunk that ends ``interval`` seconds of wall time later.

    For checkpoints and health records, which are not results: a snapshot must not
    depend on how fast the machine ran.

    Args:
        interval: A positive wall time. The clock starts when the schedule is made,
            and again each time it is due.
        clock: The clock, for tests.

    Returns:
        The schedule.

    Raises:
        ValueError: If ``interval`` is not positive.
    """
    if not interval > 0.0:
        raise ValueError(f"every_wall_seconds needs a positive time, got {interval}")
    last = [clock()]

    def due(frame: ChunkFrame) -> bool:
        now = clock()
        if now - last[0] < interval:
            return False
        last[0] = now
        return True

    return Due(due)
