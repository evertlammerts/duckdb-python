"""Shared fixtures and interpreter-capability markers."""

from __future__ import annotations

import pytest

from ._support import gil_enabled


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip free-threading tests on interpreters that still hold the GIL."""
    if list(item.iter_markers(name="freethreaded")) and gil_enabled():
        pytest.skip("requires a free-threaded interpreter")
