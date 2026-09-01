"""OUT-row fetch: rows(), the iterator, and the dbapi fetch loops. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# gate: OUT-row fetch materializes every row to Python (binding-dominated);
# the range() scan is cheap.
pytestmark = pytest.mark.gate

# scaled() shrinks N under BENCH_SCALE in the CI sweep; full N locally. The
# range(2048) probe is the compile+fetch fixed-cost baseline, NOT scaled.
N_ROW = scaled(200_000)
N_STR = scaled(100_000)
N_NEST = scaled(50_000)


def _bench_rows(benchmark: BenchmarkFixture, con: duckdb.Connection, query: str) -> None:
    duckdb.sql(query).rows(con)  # warm the engine before measuring
    benchmark(lambda: duckdb.sql(query).rows(con))


def test_rows_small_probe(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_rows(benchmark, con, "SELECT i::BIGINT AS a FROM range(2048) t(i)")


def test_rows_int(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_rows(benchmark, con, f"SELECT i::BIGINT AS a FROM range({N_ROW}) t(i)")


def test_rows_double(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_rows(benchmark, con, f"SELECT (i * 1.5)::DOUBLE AS a FROM range({N_ROW}) t(i)")


def test_rows_2int(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_rows(benchmark, con, f"SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range({N_ROW}) t(i)")


def test_rows_null_int(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    # Real NULLs, or the cheap all-valid path is measured instead.
    _bench_rows(
        benchmark, con, f"SELECT CASE WHEN i % 13 = 0 THEN NULL ELSE i END::BIGINT AS a FROM range({N_ROW}) t(i)"
    )


def test_rows_str(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_rows(benchmark, con, f"SELECT ('str_value_' || i) AS s FROM range({N_STR}) t(i)")


def test_rows_mixed(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    query = (
        "SELECT i::BIGINT AS bi, ('str_' || i) AS s, [i, i + 1, i + 2] AS lst, "
        f"{{'a': i, 'b': i + 1}} AS st FROM range({N_NEST}) t(i)"
    )
    _bench_rows(benchmark, con, query)


def test_iter_rows(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    query = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N_STR}) t(i)"
    plan = duckdb.sql(query)
    sum(1 for _ in plan.iter_rows(con))  # warm

    def drain() -> int:
        return sum(1 for _ in plan.iter_rows(con))

    benchmark(drain)


def test_dbapi_fetchone_loop(benchmark: BenchmarkFixture) -> None:
    connection = duckdb.dbapi.connect()
    query = f"SELECT i::BIGINT AS a FROM range({N_STR}) t(i)"

    def drain() -> int:
        cursor = connection.cursor()
        cursor.execute(query)
        n = 0
        while cursor.fetchone() is not None:
            n += 1
        return n

    drain()  # warm
    benchmark(drain)
    connection.close()


def test_dbapi_fetchmany_loop(benchmark: BenchmarkFixture) -> None:
    connection = duckdb.dbapi.connect()
    query = f"SELECT i::BIGINT AS a FROM range({N_STR}) t(i)"

    def drain() -> int:
        cursor = connection.cursor()
        cursor.execute(query)
        n = 0
        while batch := cursor.fetchmany(1024):
            n += len(batch)
        return n

    drain()  # warm
    benchmark(drain)
    connection.close()
