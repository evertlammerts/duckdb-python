"""type x direction produce matrix: fetchall / df / to_arrow per logical type. See benchmarks/README.md.

One logical type per column across three directions, so a regression localizes to (type, direction). Includes the
wide types the narrow-numeric benches miss: hugeint, uuid, decimal128, long varchar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(100_000)

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


def _query(type_name: str) -> str:
    return f"SELECT {TYPE_EXPR[type_name]} AS c FROM range({N}) t(i)"


@pytest.mark.gate  # OUT-row: binding-dominated per-type dispatch
@pytest.mark.parametrize("type_name", TYPES)
def test_out_row_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, type_name: str) -> None:
    q = _query(type_name)
    con.execute(q).fetchall()  # warm
    benchmark(lambda: con.execute(q).fetchall())


@pytest.mark.gate  # OUT-col: binding-dominated ArrayWrapper fill per type
@pytest.mark.parametrize("type_name", TYPES)
def test_out_col_df(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, type_name: str) -> None:
    q = _query(type_name)
    con.sql(q).df()  # warm
    benchmark(lambda: con.sql(q).df())


@pytest.mark.informational  # to_arrow_table re-runs the query GIL-released (engine-parallel, noisy) -> not gated
@pytest.mark.parametrize("type_name", TYPES)
def test_out_arrow_table(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, type_name: str) -> None:
    q = _query(type_name)
    con.sql(q).to_arrow_table()  # warm
    benchmark(lambda: con.sql(q).to_arrow_table())
