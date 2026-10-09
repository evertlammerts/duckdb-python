"""Whole results going out to pandas, on the previous client's own write cases. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# gate: assembling the frame is this package's work; the query feeding it is cheap.
pytestmark = pytest.mark.gate

N = scaled(500_000)

Q_NUMERIC = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"
# Real NULLs, or the cheap all-valid path is measured instead of the nullable build.
Q_NULLS = (
    "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
    f"CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range({N}) t(i)"
)
Q_TIMESTAMP = f"SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range({N}) t(i)"


def _bench_pandas(benchmark: BenchmarkFixture, con: duckdb.frame.Connection, query: str) -> None:
    duckdb.frame.sql(query).to_pandas(con)  # warm the engine before measuring
    benchmark(lambda: duckdb.frame.sql(query).to_pandas(con))


def test_to_pandas_numeric(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench_pandas(benchmark, con, Q_NUMERIC)


def test_to_pandas_string(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench_pandas(benchmark, con, Q_STR)


def test_to_pandas_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench_pandas(benchmark, con, Q_NULLS)


def test_to_pandas_timestamp(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench_pandas(benchmark, con, Q_TIMESTAMP)
