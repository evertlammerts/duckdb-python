"""CodSpeed benchmark: relational-API expression construction. Standalone, not in CI's binding gate.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_relational_construction_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

SCOPE: this is relational-API *construction* (ColumnExpression / ConstantExpression / operator overloads),
NOT the binding-pressure surface the rest of the suite targets. It was moved here out of test_fetch_perf.py
(MEAS-5) because it is out of scope for the binding-pressure gate. It is KEPT because it carries a real signal
(a measured ~35% expression-construction delta at the cutover), so it stays visible -- but it is marked
`informational`, so it runs and reports and is NEVER part of the gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# informational: relational-API construction, deliberately excluded from the binding-pressure gate (MEAS-5).
pytestmark = pytest.mark.informational


def test_expr_many(benchmark: BenchmarkFixture) -> None:
    """Benchmark building many column/constant expressions."""

    def run() -> int:
        out = []
        for i in range(2000):
            col = duckdb.ColumnExpression(f"col_{i}")
            const = duckdb.ConstantExpression(i)
            out.append(((col + const) * duckdb.ConstantExpression(2)).alias(f"a{i}"))
        return len(out)

    benchmark(run)
