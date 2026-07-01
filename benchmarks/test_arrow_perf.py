"""Standalone CodSpeed benchmark module for the Arrow read/write binding paths — NOT integrated
(not in pyproject, not in CI, not committed). Run under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-cutover/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-cutover
  $M -m pytest benchmarks/test_arrow_perf.py --codspeed --codspeed-mode=walltime -o addopts=
  $C -m pytest benchmarks/test_arrow_perf.py --codspeed --codspeed-mode=walltime -o addopts=

DESIGN — the data must be FULLY MOVED, not lazily wrapped, or the benchmark measures nothing:
  * READ (arrow -> duckdb): the duckdb ENGINE must scan every value. We aggregate over the actual
    columns (sum/length), NOT count(*) -- count(*) is answered from arrow metadata without touching data.
  * WRITE (duckdb -> arrow): the CONSUMER must materialize everything.
      - to_arrow_table() / pl() are EAGER (the full table / polars DataFrame is built).
      - to_arrow_reader() is LAZY -- duckdb only produces a batch when it is pulled -- so we iterate the
        whole stream to actually exercise and consume the write path.

pyarrow/polars are pinned to the SAME version in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import pyarrow as pa
import pytest

N = 500_000
WRITE_Q_NUM = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(500000) t(i)"
WRITE_Q_STR = "SELECT ('str_value_' || i) AS s FROM range(500000) t(i)"


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def arrow_numeric():
    return pa.table(
        {
            "a": pa.array(range(N), type=pa.int64()),
            "b": pa.array([i * 1.5 for i in range(N)], type=pa.float64()),
        }
    )


@pytest.fixture(scope="module")
def arrow_string():
    return pa.table({"s": pa.array([f"str_value_{i}" for i in range(N)], type=pa.string())})


@pytest.fixture(scope="module")
def arrow_numeric_batches(arrow_numeric):
    # RecordBatches are immutable/re-readable, so a fresh reader can be built from them every round
    return arrow_numeric.schema, arrow_numeric.to_batches(max_chunksize=50_000)


# --------------------------------------------------------------------------- #
# READ: arrow -> duckdb. The engine must scan every value (sum/length force it).
# --------------------------------------------------------------------------- #


def test_read_arrow_numeric(benchmark, con, arrow_numeric):
    con.register("t_num", arrow_numeric)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t_num").fetchall())


def test_read_arrow_string(benchmark, con, arrow_string):
    con.register("t_str", arrow_string)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t_str").fetchall())


# ADDED: RecordBatchReader ingest -- the SAME PythonTableArrowArrayStreamFactory but STREAMING (distinct from
# the materialized Table read above). A fresh reader is built per round (the engine drains it); sum() forces a
# full scan of every value.


def test_read_arrow_reader_numeric(benchmark, con, arrow_numeric_batches):
    schema, batches = arrow_numeric_batches

    def run():
        reader = pa.RecordBatchReader.from_batches(schema, iter(batches))
        con.register("t_rdr", reader)
        return con.execute("SELECT sum(a), sum(b) FROM t_rdr").fetchall()

    run()  # warm
    benchmark(run)


# --------------------------------------------------------------------------- #
# WRITE: duckdb -> arrow, consumer fully materializes / fully drains the stream.
# --------------------------------------------------------------------------- #


def test_write_arrow_table_numeric(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table())


def test_write_arrow_table_string(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table())


def test_write_arrow_reader_consumed(benchmark, con):
    def run():
        reader = con.sql(WRITE_Q_NUM).to_arrow_reader(100_000)
        rows = 0
        for batch in reader:  # drain the lazy stream so duckdb actually produces every batch
            rows += batch.num_rows
        return rows

    benchmark(run)


def test_write_polars_numeric(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_NUM).pl())


def test_write_polars_string(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_STR).pl())
