"""Standalone CodSpeed benchmark module for the Python UDF binding paths (src/python_udf.cpp) — NOT integrated
(not in pyproject, not in CI, not committed). Run under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_udf_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_udf_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

WHY THIS MODULE: the whole UDF subsystem had ZERO benchmark coverage. The NATIVE scalar UDF is the single
biggest untested per-call-overhead path in the binding -- per row it builds a TupleBuilder of args, calls
PyObject_CallObject, and runs TransformPythonObject on the result (python_udf.cpp). The ARROW (vectorized) UDF
is the columnar counterpart: ConvertDataChunkToPyArrowTable + the Python call + ConvertArrowTableToVector cast.

FULL CONSUME (same discipline as the other modules): every UDF benchmark wraps the call in a sum()/length()
aggregate so the ENGINE evaluates the UDF on every row (count(*) would skip it). The aggregate output is a
single row, so the measured cost is the per-row (native) / per-chunk (arrow) UDF invocation, not the fetch.

numpy/pandas/pyarrow are pinned to the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import pytest
from duckdb.sqltypes import BIGINT, DOUBLE, VARCHAR

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

NATIVE_N = 200_000  # native = one Python call per row, keep moderate
ARROW_N = 1_000_000  # arrow = one Python call per chunk (vectorized), can be large


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def _bench(benchmark, con, query):
    con.execute(query).fetchall()  # warm the engine + import caches before measuring
    benchmark(lambda: con.execute(query).fetchall())


# --------------------------------------------------------------------------- #
# NATIVE scalar UDF: per-row TupleBuilder(args) + PyObject_CallObject + TransformPythonObject(result).
# --------------------------------------------------------------------------- #


def test_udf_native_int_1arg(benchmark, con):
    con.create_function("add_one", lambda x: x + 1, [BIGINT], BIGINT)
    _bench(benchmark, con, f"SELECT sum(add_one(i::BIGINT)) FROM range({NATIVE_N}) t(i)")


def test_udf_native_int_2arg(benchmark, con):
    con.create_function("add2", lambda a, b: a + b, [BIGINT, BIGINT], BIGINT)
    _bench(benchmark, con, f"SELECT sum(add2(i::BIGINT, (i + 1)::BIGINT)) FROM range({NATIVE_N}) t(i)")


def test_udf_native_double_1arg(benchmark, con):
    con.create_function("scale", lambda x: x * 1.5, [DOUBLE], DOUBLE)
    _bench(benchmark, con, f"SELECT sum(scale((i * 1.0)::DOUBLE)) FROM range({NATIVE_N}) t(i)")


def test_udf_native_string(benchmark, con):
    con.create_function("up", lambda s: s.upper(), [VARCHAR], VARCHAR)
    _bench(
        benchmark,
        con,
        f"SELECT sum(length(up(s))) FROM (SELECT ('str_value_' || i) AS s FROM range({NATIVE_N}) t(i))",
    )


def test_udf_native_null_inputs(benchmark, con):
    # DEFAULT null handling: NULL inputs short-circuit (SetNull) WITHOUT calling the UDF -- this measures the
    # validity short-circuit, not the Python call, so the UDF only ever sees non-NULL rows.
    con.create_function("add_one", lambda x: x + 1, [BIGINT], BIGINT)
    _bench(
        benchmark,
        con,
        "SELECT sum(add_one(v)) FROM "
        f"(SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END AS v FROM range({NATIVE_N}) t(i))",
    )


# --------------------------------------------------------------------------- #
# ARROW (vectorized) UDF: ConvertDataChunkToPyArrowTable -> pc op -> ConvertArrowTableToVector cast.
# --------------------------------------------------------------------------- #


def test_udf_arrow_int(benchmark, con):
    con.create_function("arrow_add_one", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    _bench(benchmark, con, f"SELECT sum(arrow_add_one(i::BIGINT)) FROM range({ARROW_N}) t(i)")


def test_udf_arrow_double(benchmark, con):
    con.create_function("arrow_scale", lambda x: pc.multiply(x, 1.5), [DOUBLE], DOUBLE, type="arrow")
    _bench(benchmark, con, f"SELECT sum(arrow_scale((i * 1.0)::DOUBLE)) FROM range({ARROW_N}) t(i)")


def test_udf_arrow_null_inputs(benchmark, con):
    # DEFAULT null handling on the vectorized path: the binding compacts the validity (selvec) before the call
    # and reconstructs the result vector afterwards -- this is the selvec compaction/reconstruction cost.
    con.create_function("arrow_add_one", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    _bench(
        benchmark,
        con,
        "SELECT sum(arrow_add_one(v)) FROM "
        f"(SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END AS v FROM range({ARROW_N}) t(i))",
    )
