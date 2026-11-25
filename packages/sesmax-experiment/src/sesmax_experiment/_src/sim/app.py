"""The command line: ``sesmax-experiment``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter
from loguru import logger

from sesmax_experiment._src.components import components
from sesmax_experiment._src.sim import runner
from sesmax_experiment._src.sim.spec import RunSpec, load_yaml


app = App(
    name="sesmax-experiment",
    help="Run sesmax models from a YAML spec: fresh runs, spinups, restarts.",
    version_flags=["--version", "-V"],
)

Config = Annotated[Path, Parameter(help="Path to a YAML run spec.")]
Output = Annotated[Path, Parameter(help="Directory to write to.")]
Debug = Annotated[
    bool, Parameter(help="Apply the spec's debug block (smaller grid, shorter run).")
]
PerSave = Annotated[
    int,
    Parameter(help="Chunks between two snapshots, each a line in run.log."),
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


@app.command
def simulate(
    *,
    config: Config,
    output_dir: Output,
    debug: Debug = False,
    diagnostics_per_save: PerSave = 1,
    verbose: Verbose = False,
) -> None:
    """Run a spec from its initial condition."""
    _configure_logging(verbose)
    runner.simulate(
        _load(config, debug=debug),
        output_dir,
        diagnostics_per_save=diagnostics_per_save,
    )


@app.command
def spinup(
    *,
    config: Config,
    output_dir: Output,
    debug: Debug = False,
    diagnostics_per_save: PerSave = 1,
    verbose: Verbose = False,
) -> None:
    """Run a spec for its final state alone, which a restart starts from."""
    _configure_logging(verbose)
    runner.spinup(
        _load(config, debug=debug),
        output_dir,
        diagnostics_per_save=diagnostics_per_save,
    )


@app.command
def restart(
    *,
    config: Config,
    from_: Annotated[
        Path,
        Parameter(
            name=["--from", "-f"], help="Zarr store with the state to start from."
        ),
    ],
    output_dir: Output,
    debug: Debug = False,
    diagnostics_per_save: PerSave = 1,
    verbose: Verbose = False,
) -> None:
    """Run a spec from a saved state."""
    _configure_logging(verbose)
    runner.restart(
        _load(config, debug=debug),
        output_dir,
        restart_from=from_,
        diagnostics_per_save=diagnostics_per_save,
    )


@app.command(name="list-models")
def list_models() -> None:
    """List the models that can be run, by component."""
    for component in components().values():
        print(f"{component.name}:")
        for name in sorted(component.models):
            print(f"  - {name}")


@app.command(name="show-config")
def show_config(path: Config) -> None:
    """Print a spec as it is after loading and validation."""
    import yaml

    spec = load_yaml(str(path))
    print(yaml.safe_dump(spec.to_dict(), sort_keys=False, default_flow_style=False))


def main() -> None:
    """Entry point of the console script."""
    sys.exit(app() or 0)
