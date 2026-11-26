"""The command line ``sesmax-ocean``: the data of real basins, through DVC.

No bundles are hosted: whoever runs this has a DVC remote of their own. The commands
run in the directory of the package's ``dvc.yaml``.

Moved from somax 9cb19b3: ``somax/_src/cli/_data.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

from cyclopts import App, Parameter
from loguru import logger

from sesmax_ocean._src.basins.bundle import BASINS, placeholder_bundle, write_bundle


#: The names of `BASINS`, which the command line lists.
BasinName = Literal["gulf_stream", "med_sea", "north_atlantic", "southern_ocean"]

#: The name of the DVC remote.
REMOTE = "sesmax-data"
#: ``scheme of a URL -> extra of dvc that reads it``.
_EXTRAS = {"azure": "azure", "gdrive": "gdrive", "gs": "gs", "s3": "s3"}

app = App(name="sesmax-ocean", help="The data of real basins.")
data = App(name="data", help="Fetch and build the bundles of the basins, with DVC.")
app.command(data)

Name = Annotated[BasinName, Parameter(help="The basin.")]


def bundle_path(name: str) -> Path:
    """Where the bundle of a basin is, from the directory of ``dvc.yaml``."""
    return Path("data") / "basin" / f"{name}.zarr"


@data.command
def init(
    url: Annotated[
        str,
        Parameter(
            help="gdrive://<folder-id>, s3://<bucket>/<key>, gs://<bucket>/<key>, "
            "azure://<container>/<path>, or an absolute path."
        ),
    ],
    *,
    force: Annotated[
        bool, Parameter(help="Replace the remote if there is one.")
    ] = False,
) -> None:
    """Set the DVC remote the bundles are in, for this checkout alone."""
    scheme, separator, _ = url.partition("://")
    if separator and scheme not in _EXTRAS:
        raise SystemExit(f"unknown scheme {scheme!r}; known: {sorted(_EXTRAS)}")
    if not separator and not Path(url).is_absolute():
        raise SystemExit(f"neither a URL nor an absolute path: {url!r}")
    flags = ["--force"] if force else []
    _dvc("remote", "add", "--local", "--default", *flags, REMOTE, url)
    logger.info("the remote {} is {}", REMOTE, url)
    if separator:
        logger.info("if DVC misses a package: pip install 'dvc[{}]'", _EXTRAS[scheme])


@data.command
def fetch(basin: Name) -> None:
    """Get the bundle of a basin from the remote."""
    _need_remote()
    _dvc("pull", str(bundle_path(basin)))
    logger.info("fetched {}", bundle_path(basin))


@data.command
def build(
    basin: Name,
    *,
    force: Annotated[bool, Parameter(help="Even if DVC finds nothing new.")] = False,
    push: Annotated[bool, Parameter(help="Send the bundle to the remote.")] = False,
) -> None:
    """Build the bundle of a basin, a stage of ``dvc.yaml``."""
    if push:
        _need_remote()
    _dvc("repro", *(["--force"] if force else []), f"build-basin@{basin}")
    logger.info("built {}", bundle_path(basin))
    if push:
        _dvc("push", str(bundle_path(basin)))
        logger.info("pushed {}", bundle_path(basin))


@data.command
def status() -> None:
    """Show the remote, and which bundles are here."""
    url = _remote_url()
    print(f"remote: {url or 'none, see sesmax-ocean data init'}")
    for name in sorted(BASINS):
        mark = "x" if (bundle_path(name) / "zarr.json").is_file() else " "
        print(f"  [{mark}] {name}")


@data.command
def write(
    basin: Name,
    *,
    out: Annotated[Path | None, Parameter(help="Where to, a .zarr store.")] = None,
    nx: Annotated[int | None, Parameter(help="Cells in longitude.")] = None,
    ny: Annotated[int | None, Parameter(help="Cells in latitude.")] = None,
    min_depth: Annotated[float, Parameter(help="Smallest wet depth (m).")] = 20.0,
) -> None:
    """Write the bundle of a basin without DVC; what its stage runs."""
    bundle = placeholder_bundle(basin, nx=nx, ny=ny, min_depth=min_depth)
    logger.info("wrote {}", write_bundle(bundle, out or bundle_path(basin)))


def _run(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    if shutil.which("dvc") is None:
        raise SystemExit("dvc is not on PATH: pip install dvc")
    return subprocess.run(
        ["dvc", *args], check=False, capture_output=capture, text=True
    )


def _dvc(*args: str) -> None:
    """Run DVC, and stop if it fails."""
    code = _run(*args).returncode
    if code != 0:
        raise SystemExit(f"dvc {' '.join(args)} failed (exit {code})")


def _remote_url() -> str | None:
    """The URL of the remote, or ``None`` if there is none.

    Not from ``dvc remote list``, which wraps a long one.
    """
    if _run("root", capture=True).returncode != 0:
        raise SystemExit("not in a DVC repository: run this in a checkout of sesmax")
    found = _run("config", f"remote.{REMOTE}.url", capture=True)
    return found.stdout.strip() if found.returncode == 0 else None


def _need_remote() -> None:
    if _remote_url() is None:
        raise SystemExit(f"no DVC remote {REMOTE!r}: run sesmax-ocean data init <url>")


def main() -> None:
    """Entry point of the console script."""
    sys.exit(app() or 0)
