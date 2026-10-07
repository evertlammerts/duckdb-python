"""Converting Python UDF returns to their declared types, row by row. See benchmarks/README.md."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb
from duckdb.frame import sql

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# gate: the query is one projection over range(), so nearly all the work is converting the returned values.
pytestmark = pytest.mark.gate


def _bench(
    benchmark: BenchmarkFixture,
    con: duckdb.frame.Connection,
    returned: object,
    declared: str,
    rows: int,
) -> None:
    con.create_function("back", lambda _: returned, ["BIGINT"], declared)

    def run() -> None:
        sql(f"SELECT count(back(i)) FROM range({rows}) t(i)").rows(con)

    run()  # warm the engine before measuring
    benchmark(run)


def test_return_date(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, datetime.date(2020, 1, 2), "DATE", scaled(200_000))


def test_return_timestamp(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, datetime.datetime(2020, 1, 2, 3, 4, 5), "TIMESTAMP", scaled(200_000))


def test_return_nested_struct(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    returned = {"a": 1, "b": "x", "deep": {"c": 2.5, "d": True}}
    _bench(benchmark, con, returned, "STRUCT(a BIGINT, b VARCHAR, deep STRUCT(c DOUBLE, d BOOLEAN))", scaled(50_000))


def test_return_union_of_lists(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, [1, 2, 3], "UNION(n BIGINT[], w VARCHAR[])", scaled(20_000))


def test_return_dict_as_map(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, {1: "a", 2: "b"}, "MAP(BIGINT, VARCHAR)", scaled(20_000))
