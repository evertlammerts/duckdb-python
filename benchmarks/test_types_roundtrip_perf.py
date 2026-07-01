"""Standalone CodSpeed benchmark module: the TYPE x DIRECTION produce matrix — NOT integrated (not in
pyproject, not in CI, not committed). Run under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_types_roundtrip_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_types_roundtrip_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

WHY THIS MODULE: a single systematic sweep of one logical type per column across the three produce directions
  * OUT-row   = fetchall()          -> FromValue per cell (python_objects.cpp)
  * OUT-col   = df()                -> ArrayWrapper / ConvertColumn (array_wrapper.cpp)
  * OUT-arrow = to_arrow_table()    -> arrow export converters
so a regression localizes to (type, direction). Includes the iqmo/bareduckdb cross-check breadth that the
narrow-numeric homogeneous benchmarks miss: HUGEINT (PyLong_FromString / hugeint->double / int128 export),
UUID (uuid.UUID per row / UUIDConvert), DECIMAL(28,6) int128-internal (ConvertDecimalInternal<hugeint_t>),
and a long-varchar (>64 chars) that shifts the string paths from overhead-bound to copy-bound.

FULL CONSUME: fetchall and df materialize everything; to_arrow_table is eager. NOTE: to_arrow_table on a
materialized result re-runs the query with the GIL released (PromoteMaterializedToArrow), so the OUT-arrow
column is engine-parallel and walltime-NOISY -- treat it as informational, not a hard gate.

numpy/pandas/pyarrow are pinned to the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import pytest

N = 100_000

# one logical type per column; long-varchar is intentionally > 64 chars
TYPE_EXPR = {
    "int64": "i::BIGINT",
    "double": "(i * 1.5)::DOUBLE",
    "varchar_short": "('str_' || i)",
    "varchar_long": "('row_' || i || '_' || repeat('payload ', 9))",
    "timestamp": "TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND)",
    "decimal64": "((i::DECIMAL(18, 3)) / 1000)",
    "decimal128": "((i * 1.5)::DECIMAL(28, 6))",
    "hugeint": "(i::HUGEINT * 1000000000000)",
    "uuid": "gen_random_uuid()",
    "struct": "{'a': i, 'b': i + 1}",
    "list": "[i, i + 1, i + 2]",
}
TYPES = list(TYPE_EXPR)


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def _query(type_name):
    return f"SELECT {TYPE_EXPR[type_name]} AS c FROM range({N}) t(i)"


@pytest.mark.parametrize("type_name", TYPES)
def test_out_row_fetchall(benchmark, con, type_name):
    q = _query(type_name)
    con.execute(q).fetchall()  # warm
    benchmark(lambda: con.execute(q).fetchall())


@pytest.mark.parametrize("type_name", TYPES)
def test_out_col_df(benchmark, con, type_name):
    q = _query(type_name)
    con.sql(q).df()  # warm
    benchmark(lambda: con.sql(q).df())


@pytest.mark.parametrize("type_name", TYPES)
def test_out_arrow_table(benchmark, con, type_name):
    # informational only: PromoteMaterializedToArrow re-runs the query with the GIL released (noisy)
    q = _query(type_name)
    con.sql(q).to_arrow_table()  # warm
    benchmark(lambda: con.sql(q).to_arrow_table())
