"""Without the ``[sim]`` extra, the import and the command name it."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize("blocked", ["cyclopts", "loguru", "yaml", "xarray", "zarr"])
def test_the_import_names_the_extra(blocked: str) -> None:
    code = (
        "import sys\n"
        f"sys.modules[{blocked!r}] = None\n"
        "import sesmax_experiment, sesmax_experiment.components\n"
        "try:\n"
        "    import sesmax_experiment.sim\n"
        "except ImportError as err:\n"
        "    print(err)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "sesmax-experiment[sim]" in result.stdout, result.stdout


def test_the_command_names_the_extra() -> None:
    code = (
        "import sys\n"
        "sys.modules['cyclopts'] = None\n"
        "from sesmax_experiment._src.sim.launch import main\n"
        "main()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "sesmax-experiment[sim]" in result.stderr
    assert "Traceback" not in result.stderr
