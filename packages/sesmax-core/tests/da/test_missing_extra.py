"""Importing an optional module without its extra names the extra.

``sesmax_core.da`` needs filterax and vardax (``[da]``), and
``sesmax_core.flows`` needs flowjax (``[flows]``). Without them the import
must say which extra to install, not fail with a bare
``ModuleNotFoundError`` naming a package the user never asked for.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "blocked", "extra"),
    [
        ("sesmax_core.da", "filterax", "sesmax-core[da]"),
        ("sesmax_core.flows", "flowjax", "sesmax-core[flows]"),
    ],
)
def test_import_without_the_extra_names_it(module, blocked, extra):
    code = (
        "import sys\n"
        f"sys.modules[{blocked!r}] = None\n"
        "try:\n"
        f"    import {module}\n"
        "except ImportError as err:\n"
        "    print(err)\n"
        "else:\n"
        "    raise SystemExit('imported without the extra')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert extra in result.stdout, result.stdout
