"""Run the inversion tests in float64, as they ran in gridax-jax.

gridax-jax's session conftest enables x64 for every test, and the tests
moved from it compare solves at ``rtol=1e-6`` and below. The flag is
switched on for each module here and restored afterwards, so the rest of the
suite keeps its float32 default; a test about float32 builds float32 inputs.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield
