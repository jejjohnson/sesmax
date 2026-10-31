"""Top-level names that are imported on first use.

``OperatorTerm`` needs pipekit, so the top-level package exports it lazily:
``from sesmax_core import OperatorTerm`` works, and a plain
``import sesmax_core`` does not import pipekit.
"""

from __future__ import annotations

import subprocess
import sys

import sesmax_core
import sesmax_core.terms


def test_operator_term_is_exported_from_the_top_level() -> None:
    assert "OperatorTerm" in sesmax_core.__all__
    assert sesmax_core.OperatorTerm is sesmax_core.terms.OperatorTerm


def test_plain_import_does_not_load_pipekit() -> None:
    code = (
        "import sys, sesmax_core\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "stray = loaded & {'pipekit', 'pipekit_jax'}\n"
        "assert not stray, stray\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_attribute_still_raises() -> None:
    try:
        sesmax_core.NotAName  # noqa: B018
    except AttributeError as err:
        assert "NotAName" in str(err)
    else:
        raise AssertionError("expected AttributeError")


def test_io_functions_are_top_level_and_xarray_stays_lazy() -> None:
    import sesmax_core.io

    assert sesmax_core.state_to_dataset is sesmax_core.io.state_to_dataset
    code = (
        "import sys, sesmax_core\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "stray = loaded & {'xarray', 'zarr', 'orbax'}\n"
        "assert not stray, stray\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
