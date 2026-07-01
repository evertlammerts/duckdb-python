"""CodSpeed benchmark: native Python-object ingest (list/tuple/dict -> duckdb). Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_ingest_native_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Every cell goes through TransformPythonValue; dicts recurse to STRUCT; executemany re-binds per row. Note: one
list arg to values() is ONE row whose columns are the list items, so a list of N items transforms N cells.
executemany writes to a real table (CREATE OR REPLACE each round so it doesn't grow across repeats).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

EXECMANY_N = 20_000  # executemany re-binds + executes per row, keep moderate
WIDE_N = 10_000  # values() builds a 1-row x N-col relation; cap N so the binder stays sane


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh connection, closed on teardown."""
    c = duckdb.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def rows_3col() -> list[tuple[int, float, str]]:
    """Return parameter rows for a 3-column executemany."""
    return [(i, i * 1.5, f"str_value_{i}") for i in range(EXECMANY_N)]


@pytest.fixture(scope="module")
def scalars_wide() -> list[int]:
    """Return a wide row of scalar ints for values()."""
    return list(range(WIDE_N))


@pytest.fixture(scope="module")
def tuples_wide() -> list[tuple[int, int, int]]:
    """Return a wide row of tuples for values()."""
    return [(i, i + 1, i + 2) for i in range(WIDE_N)]


@pytest.fixture(scope="module")
def dicts_wide() -> list[dict[str, int | str]]:
    """Return a wide row of dicts for values()."""
    return [{"a": i, "b": i + 1, "c": f"s{i}"} for i in range(WIDE_N)]


# --------------------------------------------------------------------------- #
# executemany: bind + execute one parameter set per row, into a real table.
# --------------------------------------------------------------------------- #


def test_ingest_executemany_3col(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, rows_3col: list[tuple[int, float, str]]
) -> None:
    """Benchmark executemany INSERT of 3-column rows."""
    con.execute("CREATE OR REPLACE TABLE t (a BIGINT, b DOUBLE, c VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows_3col)  # warm

    def run() -> None:
        con.execute("CREATE OR REPLACE TABLE t (a BIGINT, b DOUBLE, c VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows_3col)

    benchmark(run)


# --------------------------------------------------------------------------- #
# values(): EAGER per-cell TransformPythonValue. Drain with fetchall to complete the round-trip.
# --------------------------------------------------------------------------- #


def test_ingest_values_scalars(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, scalars_wide: list[int]
) -> None:
    """Benchmark values() over a wide row of scalars."""
    con.values(scalars_wide).fetchall()  # warm
    benchmark(lambda: con.values(scalars_wide).fetchall())


def test_ingest_values_tuples(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, tuples_wide: list[tuple[int, int, int]]
) -> None:
    """Benchmark values() over a wide row of tuples."""
    # each tuple cell -> LIST value (TransformPythonValue recursion)
    con.values(tuples_wide).fetchall()  # warm
    benchmark(lambda: con.values(tuples_wide).fetchall())


def test_ingest_values_dicts(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, dicts_wide: list[dict[str, int | str]]
) -> None:
    """Benchmark values() over a wide row of dicts."""
    # each dict cell -> STRUCT value (TransformDictionaryToStruct recursion)
    con.values(dicts_wide).fetchall()  # warm
    benchmark(lambda: con.values(dicts_wide).fetchall())
