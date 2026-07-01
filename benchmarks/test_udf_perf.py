"""CodSpeed benchmark: Python UDF paths (native scalar + vectorized arrow). Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_udf_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Native scalar = one Python call per row (arg build + PyObject_CallObject + result transform); arrow = one call
per chunk. Full consume: each UDF is wrapped in a sum()/length() aggregate so the engine runs it on every row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb.sqltypes import BIGINT, DOUBLE, VARCHAR

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

NATIVE_N = 200_000  # native = one Python call per row, keep moderate
ARROW_N = 1_000_000  # arrow = one Python call per chunk (vectorized), can be large


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh connection, closed on teardown."""
    c = duckdb.connect()
    yield c
    c.close()


def _bench(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.execute(query).fetchall()  # warm the engine + import caches before measuring
    benchmark(lambda: con.execute(query).fetchall())


# --------------------------------------------------------------------------- #
# NATIVE scalar UDF: per-row TupleBuilder(args) + PyObject_CallObject + TransformPythonObject(result).
# --------------------------------------------------------------------------- #


def test_udf_native_int_1arg(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark a 1-arg native int scalar UDF."""
    con.create_function("add_one", lambda x: x + 1, [BIGINT], BIGINT)
    _bench(benchmark, con, f"SELECT sum(add_one(i::BIGINT)) FROM range({NATIVE_N}) t(i)")


def test_udf_native_int_2arg(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark a 2-arg native int scalar UDF."""
    con.create_function("add2", lambda a, b: a + b, [BIGINT, BIGINT], BIGINT)
    _bench(benchmark, con, f"SELECT sum(add2(i::BIGINT, (i + 1)::BIGINT)) FROM range({NATIVE_N}) t(i)")


def test_udf_native_double_1arg(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark a 1-arg native double scalar UDF."""
    con.create_function("scale", lambda x: x * 1.5, [DOUBLE], DOUBLE)
    _bench(benchmark, con, f"SELECT sum(scale((i * 1.0)::DOUBLE)) FROM range({NATIVE_N}) t(i)")


def test_udf_native_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark a native string scalar UDF."""
    con.create_function("up", lambda s: s.upper(), [VARCHAR], VARCHAR)
    _bench(
        benchmark,
        con,
        f"SELECT sum(length(up(s))) FROM (SELECT ('str_value_' || i) AS s FROM range({NATIVE_N}) t(i))",
    )


def test_udf_native_null_inputs(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark the validity short-circuit for NULL inputs to a native UDF."""
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


def test_udf_arrow_int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark a vectorized arrow int UDF."""
    con.create_function("arrow_add_one", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    _bench(benchmark, con, f"SELECT sum(arrow_add_one(i::BIGINT)) FROM range({ARROW_N}) t(i)")


def test_udf_arrow_double(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark a vectorized arrow double UDF."""
    con.create_function("arrow_scale", lambda x: pc.multiply(x, 1.5), [DOUBLE], DOUBLE, type="arrow")
    _bench(benchmark, con, f"SELECT sum(arrow_scale((i * 1.0)::DOUBLE)) FROM range({ARROW_N}) t(i)")


def test_udf_arrow_null_inputs(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark the selvec compaction for NULL inputs to a vectorized arrow UDF."""
    # DEFAULT null handling on the vectorized path: the binding compacts the validity (selvec) before the call
    # and reconstructs the result vector afterwards -- this is the selvec compaction/reconstruction cost.
    con.create_function("arrow_add_one", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
    _bench(
        benchmark,
        con,
        "SELECT sum(arrow_add_one(v)) FROM "
        f"(SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END AS v FROM range({ARROW_N}) t(i))",
    )
