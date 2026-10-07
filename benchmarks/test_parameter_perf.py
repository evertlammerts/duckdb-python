"""Binding nested Python values as query parameters: wide lists, and lists nested deep. See benchmarks/README.md."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_codspeed import BenchmarkFixture

# gate: the query reads one constant, so nearly all the work is turning the value into an engine value.
pytestmark = pytest.mark.gate

DEPTH = 400


def nested(depth: int, *, beside_null: bool) -> object:
    value: object = datetime.datetime(2020, 1, 1)
    for _ in range(depth):
        value = [value, None] if beside_null else [value]
    return value


def _bench(benchmark: BenchmarkFixture, con: duckdb.frame.Connection, value: object) -> None:
    def bind() -> None:
        with con._execute("SELECT $1 IS NULL", [value]) as result:
            result.drain()

    bind()  # warm the engine before measuring
    benchmark(bind)


def test_bind_wide_list(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, list(range(scaled(200_000))))


def test_bind_wide_list_of_timestamps(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    # Siblings that are dates or times are checked against the type the engine combines them to, one by one.
    first = datetime.datetime(2020, 1, 1)
    _bench(benchmark, con, [first + datetime.timedelta(seconds=i) for i in range(scaled(100_000))])


def test_bind_wide_list_of_structs_with_nulls(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, [{"a": i, "b": None if i % 2 else "x"} for i in range(scaled(20_000))])


def test_bind_deep_list(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, nested(DEPTH, beside_null=False))


def test_bind_deep_list_beside_nulls(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    _bench(benchmark, con, nested(DEPTH, beside_null=True))


def _contains(con: duckdb.frame.Connection, value: object) -> Callable[[], None]:
    def bind() -> None:
        with con._execute("SELECT list_contains($1, 'x')", [value]) as result:
            result.drain()

    return bind


def test_bind_a_typed_list_baseline(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    # The twin of the stand-in case below; the difference between the two is the extra bind's cost.
    bind = _contains(con, ["y"])
    bind()
    benchmark(bind)


def test_bind_a_stand_in_pays_one_extra_bind(benchmark: BenchmarkFixture, con: duckdb.frame.Connection) -> None:
    # A parameter of only NULLs has no type of its own, so the statement is bound once more to learn the expected one.
    bind = _contains(con, [None])
    bind()
    benchmark(bind)
