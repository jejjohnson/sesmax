"""The command line: ``sesmax-experiment``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from loguru import logger

from sesmax_experiment._src.components import components
from sesmax_experiment._src.monitor import Monitor, WatchdogMonitor, default_monitors
from sesmax_experiment._src.run.schedules import every_chunks, every_wall_seconds
from sesmax_experiment._src.sim import runner
from sesmax_experiment._src.sim.checkpoints import Checkpointing
from sesmax_experiment._src.sim.spec import RunSpec, load_yaml


app = App(
    name="sesmax-experiment",
    help="Run sesmax models from a YAML spec: fresh runs, spinups, restarts.",
    version_flags=["--version", "-V"],
)

#: The exit code of a run that stopped and can be resumed, ``EX_TEMPFAIL``.
EXIT_STOPPED = 75

Config = Annotated[Path, Parameter(help="Path to a YAML run spec.")]
Output = Annotated[Path, Parameter(help="Directory to write to.")]
Debug = Annotated[
    bool, Parameter(help="Apply the spec's debug block (smaller grid, shorter run).")
]
PerSave = Annotated[
    int,
    Parameter(help="Chunks between two snapshots, each a line in run.log."),
]
EveryChunks = Annotated[
    int, Parameter(help="Write a checkpoint every so many chunks; 0 for never.")
]
EverySeconds = Annotated[
    float,
    Parameter(help="Write one every so many seconds of wall time; 0 for never."),
]
KeepLast = Annotated[int, Parameter(help="How many of the newest are kept.")]
KeepEvery = Annotated[
    int | None, Parameter(help="Keep those of every so many chunks as well.")
]
MaxWall = Annotated[
    float | None,
    Parameter(
        help="Stop after so many seconds of wall time, with a checkpoint to "
        "restart from and exit code 75."
    ),
]
Verbose = Annotated[bool, Parameter(help="Show the lines of run.log on stderr.")]


def _configure_logging(verbose: bool) -> None:
    """One short, coloured sink on stderr."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> <level>{level: <7}</level> "
            "<cyan>sesmax-experiment</cyan> | <level>{message}</level>"
        ),
    )


def _load(config: Path, *, debug: bool) -> RunSpec:
    """Load a spec, with its debug block merged if asked."""
    spec = load_yaml(str(config))
    if debug:
        spec = spec.with_debug_applied()
        spec.validate()
    return spec


def _monitors(max_wall_seconds: float | None) -> list[Monitor]:
    """The default monitors, and one that watches the wall time."""
    monitors: list[Monitor] = [*default_monitors()]
    if max_wall_seconds is not None:
        monitors.append(WatchdogMonitor(max_wall_s=max_wall_seconds))
    return monitors


def _checkpointing(
    chunks: int, seconds: float, keep_last: int, keep_every: int | None
) -> Checkpointing | None:
    """The schedule of the options, or ``None`` if there is none."""
    if chunks < 0 or seconds < 0.0:
        raise ValueError(
            f"a checkpoint cadence must be >= 0, got {chunks} chunks and {seconds} s"
        )
    due = [every_chunks(chunks)] if chunks else []
    due += [every_wall_seconds(seconds)] if seconds else []
    if not due:
        return None
    both = due[0] if len(due) == 1 else due[0] | due[1]
    return Checkpointing(both, keep_last=keep_last, keep_every=keep_every)


def _exit(result: runner.SimulationResult) -> None:
    """Tell a scheduler that the run is to be continued."""
    if result.status == "stopped":
        logger.warning("restart from {}", result.checkpoint_path)
        sys.exit(EXIT_STOPPED)


@app.command
def simulate(
    *,
    config: Config,
    output_dir: Output,
    debug: Debug = False,
    diagnostics_per_save: PerSave = 1,
    checkpoint_every_n_chunks: EveryChunks = 0,
    checkpoint_every_seconds: EverySeconds = 0.0,
    keep_last: KeepLast = 3,
    keep_every: KeepEvery = None,
    max_wall_seconds: MaxWall = None,
    verbose: Verbose = False,
) -> None:
    """Run a spec from its initial condition."""
    _configure_logging(verbose)
    result = runner.simulate(
        _load(config, debug=debug),
        output_dir,
        diagnostics_per_save=diagnostics_per_save,
        monitors=_monitors(max_wall_seconds),
        checkpointing=_checkpointing(
            checkpoint_every_n_chunks, checkpoint_every_seconds, keep_last, keep_every
        ),
    )
    _exit(result)


@app.command
def spinup(
    *,
    config: Config,
    output_dir: Output,
    debug: Debug = False,
    diagnostics_per_save: PerSave = 1,
    checkpoint_every_n_chunks: EveryChunks = 0,
    checkpoint_every_seconds: EverySeconds = 0.0,
    keep_last: KeepLast = 3,
    keep_every: KeepEvery = None,
    max_wall_seconds: MaxWall = None,
    verbose: Verbose = False,
) -> None:
    """Run a spec for its final state alone, which a restart starts from."""
    _configure_logging(verbose)
    result = runner.spinup(
        _load(config, debug=debug),
        output_dir,
        diagnostics_per_save=diagnostics_per_save,
        monitors=_monitors(max_wall_seconds),
        checkpointing=_checkpointing(
            checkpoint_every_n_chunks, checkpoint_every_seconds, keep_last, keep_every
        ),
    )
    _exit(result)


@app.command
def restart(
    *,
    config: Config,
    from_: Annotated[
        Path,
        Parameter(
            name=["--from", "-f"],
            help="A state, a directory of checkpoints, or the directory of a run.",
        ),
    ],
    output_dir: Output,
    debug: Debug = False,
    diagnostics_per_save: PerSave = 1,
    checkpoint_every_n_chunks: EveryChunks = 0,
    checkpoint_every_seconds: EverySeconds = 0.0,
    keep_last: KeepLast = 3,
    keep_every: KeepEvery = None,
    max_wall_seconds: MaxWall = None,
    verbose: Verbose = False,
) -> None:
    """Run a spec from a saved state, or from a run's newest checkpoint."""
    _configure_logging(verbose)
    result = runner.restart(
        _load(config, debug=debug),
        output_dir,
        restart_from=from_,
        diagnostics_per_save=diagnostics_per_save,
        monitors=_monitors(max_wall_seconds),
        checkpointing=_checkpointing(
            checkpoint_every_n_chunks, checkpoint_every_seconds, keep_last, keep_every
        ),
    )
    _exit(result)


@app.command(name="list-models")
def list_models() -> None:
    """List the models that can be run, by component."""
    for component in components().values():
        print(f"{component.name}:")
        for name in sorted(component.models):
            print(f"  - {name}")


@app.command(name="list-scenarios")
def list_scenarios() -> None:
    """List the scenarios that can be run, with the models each goes with."""
    for component in components().values():
        if component.scenarios:
            print(f"{component.name}:")
        for name, models in sorted(component.scenarios.items()):
            print(f"  - {name}: {', '.join(models)}")


@app.command(name="show-config")
def show_config(path: Config) -> None:
    """Print a spec as it is after loading and validation."""
    import yaml

    spec = load_yaml(str(path))
    print(yaml.safe_dump(spec.to_dict(), sort_keys=False, default_flow_style=False))


def main() -> None:
    """Entry point of the console script."""
    sys.exit(app() or 0)
