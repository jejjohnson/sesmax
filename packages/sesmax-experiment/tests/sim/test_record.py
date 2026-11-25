"""The record of a run: a line of numbers per chunk."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from run_specs import shallow_water

from sesmax_experiment._src.sim.record import HealthRecord
from sesmax_experiment.monitor import ChunkInfo
from sesmax_experiment.run import ChunkFrame
from sesmax_experiment.sim import JsonLinesWriter, MetricWriter, simulate


def _frame(index: int, t0: float, t1: float, wall: float, **metrics) -> ChunkFrame:
    info = ChunkInfo(
        index=index,
        n_chunks=4,
        t0=t0,
        t1=t1,
        wall_seconds=wall,
        is_snapshot=True,
        stats={"num_steps": 50, "result_successful": True},
    )
    return ChunkFrame(state=None, info=info, metrics=metrics)


class _Keep:
    def __init__(self) -> None:
        self.lines: list[tuple[int, dict]] = []

    def write_scalars(self, step, scalars) -> None:
        self.lines.append((step, dict(scalars)))

    def flush(self) -> None: ...

    def close(self) -> None: ...


def test_a_writer_is_a_protocol() -> None:
    assert isinstance(_Keep(), MetricWriter)


def test_the_speed_leaves_the_first_chunk_out() -> None:
    """It compiled: 10 s of it, against 2 s of each of the others."""
    times = iter([0.0, 10.0, 12.0, 14.0])
    writer = _Keep()
    record = HealthRecord(writer, t1=40.0, first_step=7, clock=lambda: next(times))
    record(_frame(0, 0.0, 10.0, 9.0))
    record(_frame(1, 10.0, 20.0, 1.5))
    record(_frame(2, 20.0, 30.0, 1.0, **{"monitor.x": 3}))
    assert [step for step, _ in writer.lines] == [8, 9, 10]
    first, second, third = (scalars for _, scalars in writer.lines)
    assert first["eta_seconds"] == 30.0
    assert second["eta_seconds"] == 4.0
    assert third["eta_seconds"] == 2.0
    assert (third["t"], third["wall_seconds"], third["host_seconds"]) == (30, 1, 1)
    assert third["steps_per_second"] == 50.0
    assert third["monitor.x"] == 3.0


def test_the_lines_are_json(tmp_path: Path) -> None:
    writer = JsonLinesWriter(tmp_path / "m.jsonl")
    writer.write_scalars(1, {"a": 0.5, "b": np.nan, "c": -np.inf})
    writer.close()
    line = json.loads((tmp_path / "m.jsonl").read_text())
    assert line == {"step": 1, "a": 0.5, "b": None, "c": None}


def test_a_run_writes_a_line_per_chunk(tmp_path: Path) -> None:
    simulate(shallow_water(save_interval=200.0), tmp_path, diagnostics_per_save=2)
    lines = [
        json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().split("\n")[:-1]
    ]
    assert [line["step"] for line in lines] == [1, 2, 3, 4, 5, 6]
    assert [line["t"] for line in lines] == [100, 200, 300, 400, 500, 600]
    assert lines[-1]["eta_seconds"] == 0.0
    assert lines[0]["num_steps"] == 10
