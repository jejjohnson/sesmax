"""Run the tests moved from gridax-jax in float64, as they ran there.

gridax-jax's session conftest enables x64 for every test, and these tests
compare operators at ``rtol=1e-7``. The flag is switched on for each module
here and restored afterwards, so the rest of the suite keeps its float32
default.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import pytest


@pytest.fixture(autouse=True, scope="module")
def _float64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield
