"""The times of a run."""

from __future__ import annotations

import numpy as np
import pytest

from sesmax_experiment.run import Timeline


def test_only_the_ends_are_snapshots_by_default() -> None:
    timeline = Timeline.build(0.0, 1.0, chunk=0.25)
    np.testing.assert_array_equal(timeline.edges, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert timeline.snapshots == {0, 4}
    assert timeline.n_chunks == 4


def test_a_snapshot_cuts_a_chunk() -> None:
    timeline = Timeline.build(0.0, 10.0, chunk=4.0, snapshot=5.0)
    np.testing.assert_array_equal(timeline.edges, [0.0, 4.0, 5.0, 8.0, 10.0])
    assert timeline.snapshots == {0, 2, 4}


def test_the_last_interval_may_be_shorter() -> None:
    timeline = Timeline.build(0.0, 365.0, chunk=30.0, snapshot=30.0)
    assert timeline.n_chunks == 13
    assert timeline.edges[-2:].tolist() == [360.0, 365.0]
    assert len(timeline.snapshots) == 14


@pytest.mark.parametrize("t0", [0.0, 0.3, 1e7])
def test_rounding_makes_no_chunk_of_its_own(t0: float) -> None:
    """0.1 is not a binary number, and 100 of them still make 100 chunks."""
    timeline = Timeline.build(t0, t0 + 10.0, chunk=0.1, snapshot=0.5)
    assert timeline.n_chunks == 100
    assert len(timeline.snapshots) == 21
    assert timeline.edges[-1] == t0 + 10.0
    np.testing.assert_allclose(np.diff(timeline.edges), 0.1, rtol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        {"t0": 1.0, "t1": 1.0, "chunk": 0.1},
        {"t0": 0.0, "t1": 1.0, "chunk": 0.0},
        {"t0": 0.0, "t1": 1.0, "chunk": float("nan")},
        {"t0": 0.0, "t1": 1.0, "chunk": 0.1, "snapshot": -1.0},
    ],
)
def test_a_run_that_makes_no_sense_is_rejected(bad: dict[str, float]) -> None:
    t0, t1 = bad.pop("t0"), bad.pop("t1")
    with pytest.raises(ValueError):
        Timeline.build(t0, t1, **bad)


def test_chunks_are_cut_and_the_snapshots_kept() -> None:
    timeline = Timeline.build(0.0, 1.0, chunk=0.3, snapshot=0.3)
    cut = timeline.subdivided(3)
    assert cut.n_chunks == 3 * timeline.n_chunks
    for whole, part in zip(
        sorted(timeline.snapshots), sorted(cut.snapshots), strict=True
    ):
        # The snapshots' times are the same numbers, not close ones.
        assert cut.edges[part] == timeline.edges[whole]
    assert (np.diff(cut.edges) > 0.0).all()
    with pytest.raises(ValueError, match="one or more"):
        timeline.subdivided(0)
