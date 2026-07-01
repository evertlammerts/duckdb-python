"""CodSpeed benchmark: Arrow read/write paths. Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_arrow_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Data must be fully moved or nothing is measured: READ aggregates over real columns (sum/length, not count(*),
which arrow answers from metadata); WRITE materializes the result (to_arrow_reader is lazy, so it is drained).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from _scale import scaled

import numpy as np

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(500_000)  # env-gated: full N locally, shrunk under BENCH_SCALE in the CI Callgrind sweep (INFRA-4)
DICT_UNIQUE = [2, 1_000, 50_000]  # cardinality sweep: UNIQUE-value counts (not row counts) -> NOT scaled
WRITE_Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
WRITE_Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"

# informational: every benchmark here is engine-parallel or library/streaming dominated -> reported, not gated.
#   READ (sum over registered arrow) -> engine aggregate dominates; the near-zero-copy scan is a small fraction.
#   WRITE to_arrow_table/to_arrow_reader/pl() -> PromoteMaterializedToArrow re-runs the query GIL-released
#   (engine-parallel), and pl() also runs polars library code. Their counts would trip on engine/submodule
#   bumps, not binding regressions. `con` fixture + threads=1 live in conftest.py.
pytestmark = pytest.mark.informational


@pytest.fixture(scope="module")
def arrow_numeric() -> pa.Table:
    """Return a two-column numeric arrow table."""
    return pa.table(
        {
            "a": pa.array(range(N), type=pa.int64()),
            "b": pa.array([i * 1.5 for i in range(N)], type=pa.float64()),
        }
    )


@pytest.fixture(scope="module")
def arrow_string() -> pa.Table:
    """Return a single-column string arrow table."""
    return pa.table({"s": pa.array([f"str_value_{i}" for i in range(N)], type=pa.string())})


@pytest.fixture(scope="module")
def arrow_numeric_batches(arrow_numeric: pa.Table) -> tuple[pa.Schema, list[pa.RecordBatch]]:
    """Return the schema and record batches for the numeric table."""
    # RecordBatches are immutable/re-readable, so a fresh reader can be built from them every round
    return arrow_numeric.schema, arrow_numeric.to_batches(max_chunksize=50_000)


@pytest.fixture(scope="module")
def arrow_dict_tables() -> dict[int, pa.Table]:
    """Return dictionary-encoded arrow tables keyed by number of unique values (a cardinality sweep)."""
    # deterministic indices (i % U) so the instruction count is reproducible (no PRNG)
    tables = {}
    for u in DICT_UNIQUE:
        uniques = pa.array([f"category_value_{i}" for i in range(u)], type=pa.string())
        idx = pa.array(np.arange(N, dtype="int32") % u, type=pa.int32())
        tables[u] = pa.table({"c": pa.DictionaryArray.from_arrays(idx, uniques)})
    return tables


# --------------------------------------------------------------------------- #
# READ: arrow -> duckdb. The engine must scan every value (sum/length force it).
# --------------------------------------------------------------------------- #


def test_read_arrow_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_numeric: pa.Table
) -> None:
    """Benchmark scanning a numeric arrow table."""
    con.register("t_num", arrow_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t_num").fetchall()  # warm (MEAS-3)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t_num").fetchall())


def test_read_arrow_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_string: pa.Table) -> None:
    """Benchmark scanning a string arrow table."""
    con.register("t_str", arrow_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t_str").fetchall()  # warm (MEAS-3)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t_str").fetchall())


# ADDED: RecordBatchReader ingest -- the SAME PythonTableArrowArrayStreamFactory but STREAMING (distinct from
# the materialized Table read above). A fresh reader is built per round (the engine drains it); sum() forces a
# full scan of every value.


def test_read_arrow_reader_numeric(
    benchmark: BenchmarkFixture,
    con: duckdb.DuckDBPyConnection,
    arrow_numeric_batches: tuple[pa.Schema, list[pa.RecordBatch]],
) -> None:
    """Benchmark scanning a streaming record-batch reader."""
    schema, batches = arrow_numeric_batches

    def run() -> list:
        reader = pa.RecordBatchReader.from_batches(schema, iter(batches))
        con.register("t_rdr", reader)
        return con.execute("SELECT sum(a), sum(b) FROM t_rdr").fetchall()

    run()  # warm
    benchmark(run)


# ADDED (COV-4): dictionary-encoded arrow ingest, cardinality sweep (unique in {2, 1k, high}). Mirrors core's
# test_arrow_dictionaries_scan. The engine aggregate dominates (hence informational), but the per-value
# dictionary DECODE in the arrow scan is the binding interest, and its cost slopes with the unique count.


@pytest.mark.parametrize("unique", DICT_UNIQUE)
def test_read_arrow_dictionary(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_dict_tables: dict[int, pa.Table], unique: int
) -> None:
    """Benchmark scanning a dictionary-encoded arrow column at a given cardinality."""
    con.register("t_dict", arrow_dict_tables[unique])
    con.execute("SELECT count(c), sum(length(c)) FROM t_dict").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(c), sum(length(c)) FROM t_dict").fetchall())


# --------------------------------------------------------------------------- #
# WRITE: duckdb -> arrow, consumer fully materializes / fully drains the stream.
# --------------------------------------------------------------------------- #


def test_write_arrow_table_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a numeric result to an arrow table."""
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table())


def test_write_arrow_table_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a string result to an arrow table."""
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table())


def test_write_arrow_reader_consumed(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark draining a lazy arrow record-batch reader."""

    def run() -> int:
        reader = con.sql(WRITE_Q_NUM).to_arrow_reader(100_000)
        rows = 0
        for batch in reader:  # drain the lazy stream so duckdb actually produces every batch
            rows += batch.num_rows
        return rows

    benchmark(run)


def test_write_polars_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a numeric result to a polars frame."""
    benchmark(lambda: con.sql(WRITE_Q_NUM).pl())


def test_write_polars_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a string result to a polars frame."""
    benchmark(lambda: con.sql(WRITE_Q_STR).pl())
