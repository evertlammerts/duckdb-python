"""Test-suite hooks that depend on how this Python interpreter was built."""

from __future__ import annotations

import pytest

from ._support import gil_enabled


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests that need a Python built without the global interpreter lock."""
    if list(item.iter_markers(name="freethreaded")) and gil_enabled():
        pytest.skip("requires a free-threaded interpreter")
