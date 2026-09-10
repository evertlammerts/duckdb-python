"""The DuckDB floor: `sum()` returns one row, so almost nothing crosses into Python. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

pytestmark = pytest.mark.informational

# A floor and the bench it floors go through scaled() with the same base N, or the comparison means nothing.
Q_1C_SMALL = "SELECT sum(i::BIGINT) FROM range(2048) t(i)"
Q_1C_200K = f"SELECT sum(i::BIGINT) FROM range({scaled(200_000)}) t(i)"
Q_2C_500K = (
    f"SELECT sum(a), sum(b) FROM (SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({scaled(500_000)}) t(i))"
)


def _bench(benchmark: BenchmarkFixture, con: duckdb.Connection, query: str) -> None:
    duckdb.sql(query).rows(con)  # warm the engine before measuring
    benchmark(lambda: duckdb.sql(query).rows(con))


def test_engine_sum_1col_small(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench(benchmark, con, Q_1C_SMALL)


def test_engine_sum_1col_200k(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench(benchmark, con, Q_1C_200K)


def test_engine_sum_2col_500k(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench(benchmark, con, Q_2C_500K)
