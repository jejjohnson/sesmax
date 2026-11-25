"""Checkpoints of a run: versioned stores, each finished by a rename."""

from __future__ import annotations

import dataclasses
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jaxtyping import PyTree
from pipekit_ops import Branch, Identity, Operator, Retry, Sink, Try

from sesmax_core import io
from sesmax_experiment._src.run.frame import ChunkFrame
from sesmax_experiment._src.run.schedules import Due


#: The directory of a run's checkpoints, and the one store somax overwrote.
DIRECTORY, SOMAX_STORE = "checkpoints", "checkpoint.zarr"
_COMPLETE = re.compile(r"(\d+)\.zarr")


@dataclasses.dataclass(frozen=True)
class Checkpointing:
    """When a run writes a checkpoint, and which ones it keeps.

    Args:
        due: When one is written, for example ``every_chunks(20)``.
        keep_last: How many of the newest are kept.
        keep_every: Keep those after every ``keep_every``-th chunk as well, or
            ``None`` for no archive.

    Raises:
        ValueError: If a number is not positive.

    Examples:
        >>> from sesmax_experiment.run import every_chunks, every_wall_seconds
        >>> from sesmax_experiment.sim import Checkpointing
        >>> due = every_chunks(20) | every_wall_seconds(1800.0)
        >>> Checkpointing(due, keep_last=2, keep_every=100).keep_last
        2
    """

    due: Due
    keep_last: int = 3
    keep_every: int | None = None

    def __post_init__(self) -> None:
        if self.keep_last < 1:
            raise ValueError(f"keep_last must be positive, got {self.keep_last}")
        if self.keep_every is not None and self.keep_every < 1:
            raise ValueError(f"keep_every must be positive, got {self.keep_every}")


class CheckpointStore:
    """The checkpoints of a run, ``<directory>/<step>.zarr``.

    Each is written as ``<step>.partial.zarr`` and renamed to a new name, so a crash
    leaves a partial store and every older checkpoint as it was. Some network mounts
    refuse a rename onto an existing store, so none is made.

    Args:
        directory: Where they are.
        keep_last: How many of the newest are kept.
        keep_every: Keep every ``keep_every``-th step as well.
    """

    def __init__(
        self, directory: Path, *, keep_last: int = 3, keep_every: int | None = None
    ) -> None:
        self.directory = directory
        self.keep_last = keep_last
        self.keep_every = keep_every

    def path(self, step: int) -> Path:
        """The store of a step."""
        return self.directory / f"{step:08d}.zarr"

    def steps(self) -> list[int]:
        """The steps that have a complete checkpoint, in order."""
        if not self.directory.is_dir():
            return []
        names = (_COMPLETE.fullmatch(p.name) for p in self.directory.iterdir())
        return sorted(int(match[1]) for match in names if match)

    def latest(self) -> Path | None:
        """The newest complete checkpoint, or ``None`` if there is none."""
        steps = self.steps()
        return self.path(steps[-1]) if steps else None

    def save(
        self, state: PyTree, *, step: int, time: float, attrs: dict[str, Any]
    ) -> Path:
        """Write the checkpoint of a step, and remove those not kept.

        Args:
            state: The state.
            step: The chunks of the whole run before it.
            time: Its time.
            attrs: The attributes of the store; ``sesmax_sim_t`` and
                ``sesmax_step`` are added.

        Returns:
            The store. One that is there already is left as it is.
        """
        path = self.path(step)
        if not path.exists():
            partial = path.with_suffix(".partial.zarr")
            if partial.exists():
                shutil.rmtree(partial)
            self.directory.mkdir(parents=True, exist_ok=True)
            attrs = attrs | {"sesmax_sim_t": float(time), "sesmax_step": int(step)}
            ds = io.state_to_dataset(state, time=time, attrs=attrs)
            io.save_dataset(ds, partial, mode="w")
            partial.rename(path)
        self._prune()
        return path

    def _prune(self) -> None:
        for step in self.steps()[: -self.keep_last]:
            if self.keep_every is None or step % self.keep_every:
                shutil.rmtree(self.path(step))


def resolve(path: Path) -> Path:
    """The store to restart from: ``path``, or the newest checkpoint inside it.

    Args:
        path: A zarr store, a directory of checkpoints, or the directory of a run.
            A run of somax has one ``checkpoint.zarr``.

    Returns:
        The store.

    Raises:
        FileNotFoundError: If there is no complete checkpoint.
    """
    if any((path / name).exists() for name in ("zarr.json", ".zgroup")):
        return path
    for directory in (path, path / DIRECTORY):
        if (found := CheckpointStore(directory).latest()) is not None:
            return found
    if (path / SOMAX_STORE).is_dir():
        return path / SOMAX_STORE
    raise FileNotFoundError(f"no store and no complete checkpoint at {path}")


def remove_stale(output_dir: Path) -> None:
    """Remove the checkpoints an earlier run left in a directory.

    They would restart a run that is not the one in the directory.
    """
    for name in (DIRECTORY, SOMAX_STORE):
        if (output_dir / name).is_dir():
            shutil.rmtree(output_dir / name)


def checkpoint_stage(
    write: Callable[[ChunkFrame], Any],
    due: Due,
    warn: Callable[[str], Any],
    *,
    delay: float = 1.0,
) -> Operator:
    """A stage that writes a checkpoint when it is due, and never stops a run.

    A write that fails with an ``OSError``, as a network mount may, is tried once
    more. If that fails too the run goes on without this checkpoint.

    Args:
        write: Writes the checkpoint of a frame.
        due: When.
        warn: Takes the message if both writes fail.
        delay: Seconds before the second attempt.

    Returns:
        The stage.
    """

    def skipped(frame: ChunkFrame) -> None:
        warn(
            f"the checkpoint of chunk {frame.info.index + 1} could not be written; "
            "the run goes on without it"
        )

    attempts = Retry(Sink(write), attempts=2, base_delay=delay, on=(OSError,))
    return Branch(due, Try(attempts, Sink(skipped), on=(OSError,)), Identity())
