"""Test-suite hooks that depend on how this Python interpreter was built and what is installed beside it."""

from __future__ import annotations

import importlib.util

import pytest

from ._support import gil_enabled


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests that need a Python built without the global interpreter lock, or a library that is not installed."""
    if list(item.iter_markers(name="freethreaded")) and gil_enabled():
        pytest.skip("requires a free-threaded interpreter")
    for marker in item.iter_markers(name="requires"):
        for module in marker.args:
            if importlib.util.find_spec(module) is None:
                pytest.skip(f"requires {module}, which is not installed")
