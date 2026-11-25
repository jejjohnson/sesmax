"""When a stage is due."""

from __future__ import annotations

import pytest

from sesmax_experiment.monitor import ChunkInfo
from sesmax_experiment.run import (
    ChunkFrame,
    at_snapshots,
    every_chunks,
    every_sim_seconds,
    every_wall_seconds,
)


def _frames(n: int, *, length: float = 1.0, t0: float = 0.0) -> list[ChunkFrame]:
    return [
        ChunkFrame(
            state=None,
            info=ChunkInfo(
                index=k,
                n_chunks=n,
                t0=t0 + k * length,
                t1=t0 + (k + 1) * length,
                wall_seconds=0.0,
                is_snapshot=k % 2 == 1,
            ),
        )
        for k in range(n)
    ]


def test_every_chunks() -> None:
    due = every_chunks(3)
    assert [due(f) for f in _frames(7)] == [False, False, True] * 2 + [False]


def test_at_snapshots() -> None:
    assert [at_snapshots()(f) for f in _frames(4)] == [False, True, False, True]


def test_every_sim_seconds_fires_once_per_multiple() -> None:
    """Whatever the length of the chunks, and of their rounding."""
    due = every_sim_seconds(1.0)
    fired = [f.info.t1 for f in _frames(100, length=0.1) if due(f)]
    assert fired == pytest.approx([float(k) for k in range(1, 11)])
    due = every_sim_seconds(1.0)
    assert [due(f) for f in _frames(3, length=2.5)] == [True, True, True]


def test_every_sim_seconds_counts_from_its_start() -> None:
    due = every_sim_seconds(2.0, t0=1.0)
    assert [due(f) for f in _frames(4, t0=1.0)] == [False, True, False, True]


def test_every_wall_seconds_starts_again_when_due() -> None:
    now = [0.0]
    due = every_wall_seconds(10.0, clock=lambda: now[0])
    out = []
    for frame, t in zip(_frames(5), [4.0, 9.0, 11.0, 15.0, 22.0], strict=True):
        now[0] = t
        out.append(due(frame))
    assert out == [False, False, True, False, True]


def test_schedules_combine_and_keep_counting() -> None:
    now = [0.0]
    wall = every_wall_seconds(2.0, clock=lambda: now[0])
    either = every_chunks(2) | wall
    both = every_chunks(2) & every_chunks(3)
    out = []
    for k, frame in enumerate(_frames(6)):
        now[0] = float(k)
        out.append(either(frame))
    # chunks 2, 4 and 6, and the wall at 2 s and 4 s
    assert out == [False, True, True, True, True, True]
    assert [both(f) for f in _frames(6)] == [False] * 5 + [True]


@pytest.mark.parametrize(
    "make",
    [
        lambda: every_chunks(0),
        lambda: every_sim_seconds(0.0),
        lambda: every_wall_seconds(-1.0),
    ],
)
def test_a_schedule_that_is_never_or_always_due_is_rejected(make) -> None:
    with pytest.raises(ValueError):
        make()
