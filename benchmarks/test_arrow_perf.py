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

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

N = 500_000
WRITE_Q_NUM = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(500000) t(i)"
WRITE_Q_STR = "SELECT ('str_value_' || i) AS s FROM range(500000) t(i)"


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh connection, closed on teardown."""
    c = duckdb.connect()
    yield c
    c.close()


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


# --------------------------------------------------------------------------- #
# READ: arrow -> duckdb. The engine must scan every value (sum/length force it).
# --------------------------------------------------------------------------- #


def test_read_arrow_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_numeric: pa.Table
) -> None:
    """Benchmark scanning a numeric arrow table."""
    con.register("t_num", arrow_numeric)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t_num").fetchall())


def test_read_arrow_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, arrow_string: pa.Table) -> None:
    """Benchmark scanning a string arrow table."""
    con.register("t_str", arrow_string)
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
