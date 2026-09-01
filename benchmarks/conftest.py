"""Shared fixtures and marker registration for the benchmark suite. See benchmarks/README.md.

Markers are registered here because the repo's pytest config sets
`filterwarnings = ["error"]` and `--strict-markers`, so an unregistered mark
would fail collection. Every benchmark carries exactly one of `gate` /
`informational`, so the CI steps `-m gate` and `-m informational` cover the
suite with no overlap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator


def pytest_configure(config: pytest.Config) -> None:
    """Register the gate/informational markers (required under filterwarnings=error)."""
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
    """Yield a fresh single-threaded connection, closed on teardown.

    `threads=1` pins engine parallelism so counts and walltime do not shift
    with the runner's core count.
    """
    connection = duckdb.connect(threads="1")
    yield connection
    connection.close()
