"""Shared fixtures + marker registration for the CodSpeed benchmark suite.

Central home (INFRA-6) for the `con` fixture, the `threads=1` isolation default, and the gate/informational
marker registration (INFRA-1). Markers are registered HERE (not via pyproject `markers=`) to keep the suite
self-contained. Registration is REQUIRED: pyproject sets `filterwarnings = ["error"]`, so an unregistered
mark would raise `PytestUnknownMarkWarning` as a collection error.

Marker semantics
  gate          Binding-dominated, GIL-held, deterministic under Callgrind (instruction-count). These are the
                paths where a threshold breach means a *binding* regression. Gate-able. (Enforcement against a
                committed baseline is a later phase; for now they run and report.)
  informational Engine/parallel/IO/library-diluted, streaming drains, or arrow-export re-run paths. Reported,
                never gated: their instruction count is dominated by non-binding work (engine aggregate, the
                bundled DuckDB submodule, pyarrow/polars library code), so gating them would false-positive on
                engine/submodule bumps rather than catch binding regressions.

Every benchmark (a test using the `benchmark` fixture) must carry EXACTLY ONE of these markers so the two CI
steps (`-m gate`, `-m informational`) together cover the suite with no overlap. Non-benchmark guards (e.g. the
tracemalloc assertion in test_produce_numpy_perf.py) are intentionally left unmarked and run in neither step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import bench_scale, scaled  # noqa: F401  (re-exported here as the shared home; used by the modules)

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator


# ENV-GATED ROW COUNTS (INFRA-4): the O(rows) / per-row-object benchmarks route their N through `scaled()`
# (benchmarks/_scale.py). Unset `BENCH_SCALE` -> full N (local walltime A/B is unchanged); the CI Callgrind
# sweep sets `BENCH_SCALE=<divisor>` to shrink N so the sweep fits under the job timeout. A gate benchmark and
# its engine-control floor (FLOOR_MAP in compare_baseline.py) share a base N, so both scale identically and the
# Option-B binding fraction stays valid. Scaling changes ONLY row counts, never the Do-NOT-regress data patterns.


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
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh single-threaded connection, closed on teardown.

    `threads=1` pins engine parallelism so per-run instruction counts and walltime do not shift with the CI
    runner core count (INFRA-6). The concurrency module (COV-1, a later phase) overrides this deliberately.
    """
    c = duckdb.connect(config={"threads": 1})
    yield c
    c.close()
