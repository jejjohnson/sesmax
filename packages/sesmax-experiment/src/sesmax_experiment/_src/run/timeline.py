"""The times of a run: the ends of its chunks, and which are snapshots."""

from __future__ import annotations

import dataclasses
import math

import numpy as np


# Two times closer than this, relative to the larger, are one.
_CLOSE = 1e-9


def _multiples(t0: float, t1: float, interval: float) -> list[float]:
    """``t0 + k interval`` up to ``t1``, which closes the list."""
    if not (math.isfinite(interval) and interval > 0.0):
        raise ValueError(f"an interval must be a positive number, got {interval!r}")
    count = math.floor((t1 - t0) / interval * (1.0 + _CLOSE))
    times = [t0 + k * interval for k in range(count + 1)]
    if abs(times[-1] - t1) > _CLOSE * max(abs(t1), 1.0):
        times.append(t1)
    else:
        times[-1] = t1
    return times


@dataclasses.dataclass(frozen=True, eq=False)
class Timeline:
    """When the chunks of a run end, and which ends are snapshots.

    Build one with :meth:`build`. The times are in float64 on the host, so a long
    run does not collect the rounding of single precision in them.

    Args:
        edges: ``t₀ < t₁ < … < t_N``, the start of the run and the end of each of
            its ``N`` chunks, of shape ``(N + 1,)``.
        snapshots: The indices of the edges that are snapshots.
    """

    edges: np.ndarray
    snapshots: frozenset[int]

    @property
    def n_chunks(self) -> int:
        """The number of chunks."""
        return len(self.edges) - 1

    @classmethod
    def build(
        cls,
        t0: float,
        t1: float,
        *,
        chunk: float,
        snapshot: float | None = None,
    ) -> Timeline:
        """Chunks of one length, cut where a snapshot falls inside one.

        Both series start at ``t0``, and ``t1`` ends both, so the last chunk and
        the last snapshot interval may be shorter than the others.

        Args:
            t0: Start of the run.
            t1: End of the run, after ``t0``.
            chunk: Time between two looks at the state.
            snapshot: Time between two snapshots; by default only the start and
                the end are.

        Returns:
            The timeline.

        Raises:
            ValueError: If ``t1`` is not after ``t0``, or an interval is not a
                positive number.

        Examples:
            >>> from sesmax_experiment.run import Timeline
            >>> timeline = Timeline.build(0.0, 10.0, chunk=4.0, snapshot=5.0)
            >>> timeline.edges.tolist()
            [0.0, 4.0, 5.0, 8.0, 10.0]
            >>> sorted(timeline.snapshots)
            [0, 2, 4]
        """
        t0, t1 = float(t0), float(t1)
        if not t1 > t0:
            raise ValueError(f"a run needs t1 > t0, got t0={t0!r}, t1={t1!r}")
        saved = [t0, t1] if snapshot is None else _multiples(t0, t1, float(snapshot))
        edges: list[float] = []
        snapshots: set[int] = set()
        flagged = [(t, True) for t in saved]
        flagged += [(t, False) for t in _multiples(t0, t1, float(chunk))]
        # A snapshot first, where a chunk ends at the same time.
        for t, is_snapshot in sorted(flagged, key=lambda item: (item[0], not item[1])):
            close = edges and abs(t - edges[-1]) <= _CLOSE * max(abs(t), abs(t1 - t0))
            if not close:
                edges.append(t)
            elif is_snapshot:
                edges[-1] = t
            if is_snapshot:
                snapshots.add(len(edges) - 1)
        return cls(edges=np.asarray(edges), snapshots=frozenset(snapshots))

    def after(self, t: float) -> Timeline:
        """The rest of the timeline, from a time on.

        The edges that are left are the same numbers, so a run that continues from
        an edge takes the steps it would have taken.

        Args:
            t: A time of the run. If it is not an edge, the first chunk starts there
                and ends at the next one.

        Returns:
            The timeline, or this one if ``t`` is not after its start.

        Raises:
            ValueError: If ``t`` is not before the end.

        Examples:
            >>> from sesmax_experiment.run import Timeline
            >>> timeline = Timeline.build(0.0, 10.0, chunk=2.0, snapshot=4.0)
            >>> rest = timeline.after(6.0)
            >>> rest.edges.tolist(), sorted(rest.snapshots)
            ([6.0, 8.0, 10.0], [1, 2])
        """
        close = _CLOSE * max(abs(t), abs(self.edges[-1] - self.edges[0]))
        if t <= self.edges[0] + close:
            return self
        if t >= self.edges[-1] - close:
            raise ValueError(f"nothing is left of the run after t={t!r}")
        # The last edge that is not after ``t``, which becomes the first.
        first = int(np.searchsorted(self.edges, t + close, side="right")) - 1
        edges = self.edges[first:].copy()
        kept = {k - first for k in self.snapshots if k >= first}
        if abs(edges[0] - t) > close:
            edges[0] = t
            kept.discard(0)
        return Timeline(edges=edges, snapshots=frozenset(kept))

    def subdivided(self, n: int) -> Timeline:
        """The same snapshots, with each chunk cut into ``n`` equal ones.

        Args:
            n: The chunks in place of one, at least one.

        Returns:
            The timeline.

        Raises:
            ValueError: If ``n`` is less than one.

        Examples:
            >>> from sesmax_experiment.run import Timeline
            >>> timeline = Timeline.build(0.0, 10.0, chunk=6.0, snapshot=6.0)
            >>> timeline.subdivided(2).edges.tolist()
            [0.0, 3.0, 6.0, 8.0, 10.0]
        """
        if n < 1:
            raise ValueError(f"a chunk is cut into one or more, got {n!r}")
        starts, lengths = self.edges[:-1], np.diff(self.edges)
        inner = starts[:, None] + lengths[:, None] * np.arange(n) / n
        edges = np.append(inner.ravel(), self.edges[-1])
        return Timeline(edges=edges, snapshots=frozenset(n * k for k in self.snapshots))
