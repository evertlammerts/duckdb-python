"""Standalone CodSpeed benchmark module for NATIVE Python-object ingest (Python list/tuple/dict -> duckdb) —
NOT integrated (not in pyproject, not in CI, not committed). Run under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_ingest_native_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_ingest_native_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

WHY THIS MODULE: native Python-object ingest had ZERO coverage. Every cell goes through TransformPythonValue
and the GetPythonObjectType ladder (python_conversion.cpp); dicts recurse through TransformDictionaryToStruct;
executemany re-binds a parameter set per row (pyconnection.cpp ExecuteMany loop).

FULL MATERIALIZE: executemany lands N rows in a real table (CREATE OR REPLACE each round so the table does not
grow across codspeed's repeated invocations). values() builds the value vectors EAGERLY inside the call
(TransformPythonParamList), and we drain the resulting relation with fetchall so the round-trip is complete.

NOTE on values() shape: a single list argument to values() becomes ONE row whose COLUMNS are the list items
(see DuckDBPyConnection::Values, pyconnection.cpp) -- so a list of N scalars is 1 row x N columns and runs
TransformPythonValue N times; a list of N tuples is 1 row x N nested(LIST) columns; a list of N dicts is
1 row x N STRUCT columns (TransformDictionaryToStruct). All three exercise the per-cell transform N times.
"""

import duckdb
import pytest

EXECMANY_N = 20_000  # executemany re-binds + executes per row, keep moderate
WIDE_N = 10_000  # values() builds a 1-row x N-col relation; cap N so the binder stays sane


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def rows_3col():
    return [(i, i * 1.5, f"str_value_{i}") for i in range(EXECMANY_N)]


@pytest.fixture(scope="module")
def scalars_wide():
    return [i for i in range(WIDE_N)]


@pytest.fixture(scope="module")
def tuples_wide():
    return [(i, i + 1, i + 2) for i in range(WIDE_N)]


@pytest.fixture(scope="module")
def dicts_wide():
    return [{"a": i, "b": i + 1, "c": f"s{i}"} for i in range(WIDE_N)]


# --------------------------------------------------------------------------- #
# executemany: bind + execute one parameter set per row, into a real table.
# --------------------------------------------------------------------------- #


def test_ingest_executemany_3col(benchmark, con, rows_3col):
    con.execute("CREATE OR REPLACE TABLE t (a BIGINT, b DOUBLE, c VARCHAR)")
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows_3col)  # warm

    def run():
        con.execute("CREATE OR REPLACE TABLE t (a BIGINT, b DOUBLE, c VARCHAR)")
        con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows_3col)

    benchmark(run)


# --------------------------------------------------------------------------- #
# values(): EAGER per-cell TransformPythonValue. Drain with fetchall to complete the round-trip.
# --------------------------------------------------------------------------- #


def test_ingest_values_scalars(benchmark, con, scalars_wide):
    con.values(scalars_wide).fetchall()  # warm
    benchmark(lambda: con.values(scalars_wide).fetchall())


def test_ingest_values_tuples(benchmark, con, tuples_wide):
    # each tuple cell -> LIST value (TransformPythonValue recursion)
    con.values(tuples_wide).fetchall()  # warm
    benchmark(lambda: con.values(tuples_wide).fetchall())


def test_ingest_values_dicts(benchmark, con, dicts_wide):
    # each dict cell -> STRUCT value (TransformDictionaryToStruct recursion)
    con.values(dicts_wide).fetchall()  # warm
    benchmark(lambda: con.values(dicts_wide).fetchall())
