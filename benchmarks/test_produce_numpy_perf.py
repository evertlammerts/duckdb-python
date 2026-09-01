"""Columnar egress: to_numpy and to_pandas. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# gate: the egress path (ChunkView flatten, buffer copies, mask expansion,
# dtype assembly) is client code; the range() scan feeding it is cheap.
pytestmark = pytest.mark.gate

N_NUM = scaled(500_000)
N_NULL = scaled(200_000)
N_STR = scaled(100_000)

Q_NUMERIC = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N_NUM}) t(i)"
# Real NULLs, or the cheap all-valid path is measured instead.
Q_NULL = f"SELECT CASE WHEN i % 13 = 0 THEN NULL ELSE i END::BIGINT AS a FROM range({N_NULL}) t(i)"
Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N_STR}) t(i)"
Q_TS = f"SELECT (TIMESTAMP '2024-01-01' + INTERVAL (i % 3600) SECOND) AS ts FROM range({N_NULL}) t(i)"


def _bench_numpy(benchmark: BenchmarkFixture, con: duckdb.Connection, query: str) -> None:
    duckdb.sql(query).to_numpy(con)  # warm the engine before measuring
    benchmark(lambda: duckdb.sql(query).to_numpy(con))


def _bench_pandas(benchmark: BenchmarkFixture, con: duckdb.Connection, query: str) -> None:
    duckdb.sql(query).to_pandas(con)  # warm the engine before measuring
    benchmark(lambda: duckdb.sql(query).to_pandas(con))


def test_to_numpy_small_probe(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_numpy(benchmark, con, "SELECT i::BIGINT AS a FROM range(2048) t(i)")


def test_to_numpy_numeric(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_numpy(benchmark, con, Q_NUMERIC)


def test_to_numpy_null_int(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_numpy(benchmark, con, Q_NULL)


def test_to_numpy_str(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_numpy(benchmark, con, Q_STR)


def test_to_pandas_numeric(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_pandas(benchmark, con, Q_NUMERIC)


def test_to_pandas_null_int(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_pandas(benchmark, con, Q_NULL)


def test_to_pandas_str(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_pandas(benchmark, con, Q_STR)


def test_to_pandas_timestamp(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench_pandas(benchmark, con, Q_TS)
