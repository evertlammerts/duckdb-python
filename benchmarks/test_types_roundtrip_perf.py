"""CodSpeed benchmark: the type x direction produce matrix (fetchall / df / to_arrow per type). Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_types_roundtrip_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

One logical type per column across three directions, so a regression localizes to (type, direction). Includes
the wide types the narrow-numeric benchmarks miss: hugeint, uuid, decimal128, long varchar. Note: to_arrow on a
materialized result re-runs the query with the GIL released, so the arrow column is engine-parallel and
walltime-noisy: informational, not a hard gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(100_000)  # env-gated: full N locally, shrunk under BENCH_SCALE in the CI Callgrind sweep (INFRA-4)

# one logical type per column; long-varchar is intentionally > 64 chars
TYPE_EXPR = {
    "int64": "i::BIGINT",
    "double": "(i * 1.5)::DOUBLE",
    "varchar_short": "('str_' || i)",
    "varchar_long": "('row_' || i || '_' || repeat('payload ', 9))",
    "date": "DATE '2020-01-01' + (i % 3650)::INTEGER",
    "bool": "(i % 2 = 0)",
    "timestamp": "TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND)",
    "decimal64": "((i::DECIMAL(18, 3)) / 1000)",
    "decimal128": "((i * 1.5)::DECIMAL(28, 6))",
    "hugeint": "(i::HUGEINT * 1000000000000)",
    "uuid": "gen_random_uuid()",
    "struct": "{'a': i, 'b': i + 1}",
    "list": "[i, i + 1, i + 2]",
}
TYPES = list(TYPE_EXPR)


# `con` fixture + threads=1 live in conftest.py.
def _query(type_name: str) -> str:
    return f"SELECT {TYPE_EXPR[type_name]} AS c FROM range({N}) t(i)"


@pytest.mark.gate  # OUT-row fetchall -> binding-dominated per-type dispatch
@pytest.mark.parametrize("type_name", TYPES)
def test_out_row_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, type_name: str) -> None:
    """Benchmark fetchall of one logical type per column."""
    q = _query(type_name)
    con.execute(q).fetchall()  # warm
    benchmark(lambda: con.execute(q).fetchall())


@pytest.mark.gate  # OUT-col df() -> binding-dominated ArrayWrapper fill per type
@pytest.mark.parametrize("type_name", TYPES)
def test_out_col_df(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, type_name: str) -> None:
    """Benchmark df() of one logical type per column."""
    q = _query(type_name)
    con.sql(q).df()  # warm
    benchmark(lambda: con.sql(q).df())


@pytest.mark.informational  # to_arrow_table re-runs the query GIL-released (engine-parallel) -> not gated
@pytest.mark.parametrize("type_name", TYPES)
def test_out_arrow_table(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, type_name: str) -> None:
    """Benchmark to_arrow_table() of one logical type per column (informational only)."""
    # informational only: PromoteMaterializedToArrow re-runs the query with the GIL released (noisy)
    q = _query(type_name)
    con.sql(q).to_arrow_table()  # warm
    benchmark(lambda: con.sql(q).to_arrow_table())
