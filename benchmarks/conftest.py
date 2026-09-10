"""Shared fixtures and marker registration for the benchmark suite. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers; unregistered ones fail collection here, and a benchmark carries exactly one."""
    config.addinivalue_line(
        "markers",
        "gate: binding-dominated, instruction-count gate-able under Callgrind (deterministic).",
    )
    config.addinivalue_line(
        "markers",
        "informational: engine/library-diluted or streaming; reported, never gated.",
    )


@pytest.fixture
def con() -> Iterator[duckdb.Connection]:
    """A fresh single-threaded connection; `threads=1` keeps counts steady whatever the runner's core count."""
    connection = duckdb.connect(threads="1")
    yield connection
    connection.close()
