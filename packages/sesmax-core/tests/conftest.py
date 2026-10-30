"""Shared pytest configuration for the sesmax-core tests.

Automatic marker assignment, as in somax (``tests/conftest.py`` at 9cb19b3):
every test whose name begins with ``test_grad`` is tagged ``slow``, because
those tests are dominated by JIT-compiled differentiation through long
integrations. Applying the marker here keeps the split correct as tests are
added, without decorating each one.

The PR CI job runs ``pytest -m "not slow"``; the full suite runs in the
manual ``full-tests.yml`` workflow.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Tag every ``test_grad*`` test as ``slow``."""
    slow = pytest.mark.slow
    for item in items:
        name = getattr(item, "originalname", None) or item.name
        if name.startswith("test_grad"):
            item.add_marker(slow)
