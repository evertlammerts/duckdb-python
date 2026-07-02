"""Native Python-object ingest: values() list/tuple/dict, executemany. See benchmarks/README.md.

Every cell goes through TransformPythonValue; dicts recurse to STRUCT; executemany re-binds per row. Note: one
list arg to values() is ONE row whose columns are the list items, so a list of N items transforms N cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

EXECMANY_N = scaled(20_000)  # executemany re-binds + executes per row, keep moderate
WIDE_N = scaled(10_000)  # values() builds a 1-row x N-col relation; cap N so the binder stays sane

# gate: native ingest eagerly transforms every cell / re-binds per row; the engine side is negligible.
pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def rows_3col() -> list[tuple[int, float, str]]:
    return [(i, i * 1.5, f"str_value_{i}") for i in range(EXECMANY_N)]


@pytest.fixture(scope="module")
def scalars_wide() -> list[int]:
    return list(range(WIDE_N))


@pytest.fixture(scope="module")
def tuples_wide() -> list[tuple[int, int, int]]:
    return [(i, i + 1, i + 2) for i in range(WIDE_N)]


@pytest.fixture(scope="module")
def dicts_wide() -> list[dict[str, int | str]]:
    return [{"a": i, "b": i + 1, "c": f"s{i}"} for i in range(WIDE_N)]


# executemany: bind + execute one parameter set per row, into a real table (CREATE OR REPLACE so it doesn't grow).


def test_ingest_executemany_3col(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, rows_3col: list[tuple[int, float, str]]
) -> None:
    con.execute("CREATE OR REPLACE TABLE t (a BIGINT, b DOUBLE, c VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows_3col)  # warm

    def run() -> None:
        con.execute("CREATE OR REPLACE TABLE t (a BIGINT, b DOUBLE, c VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows_3col)

    benchmark(run)


# values(): EAGER per-cell TransformPythonValue. Drain with fetchall to complete the round-trip.


def test_ingest_values_scalars(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, scalars_wide: list[int]
) -> None:
    con.values(scalars_wide).fetchall()  # warm
    benchmark(lambda: con.values(scalars_wide).fetchall())


def test_ingest_values_tuples(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, tuples_wide: list[tuple[int, int, int]]
) -> None:
    # each tuple cell -> LIST value (TransformPythonValue recursion)
    con.values(tuples_wide).fetchall()  # warm
    benchmark(lambda: con.values(tuples_wide).fetchall())


def test_ingest_values_dicts(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, dicts_wide: list[dict[str, int | str]]
) -> None:
    # each dict cell -> STRUCT value (TransformDictionaryToStruct recursion)
    con.values(dicts_wide).fetchall()  # warm
    benchmark(lambda: con.values(dicts_wide).fetchall())
