"""Standalone CodSpeed benchmark module for the pandas read/write binding paths, comparing NUMPY-backed vs
ARROW-backed DataFrames — NOT integrated (not in pyproject, not in CI, not committed). Run under each build:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-cutover/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-cutover
  $M -m pytest benchmarks/test_pandas_perf.py --codspeed --codspeed-mode=walltime -o addopts=
  $C -m pytest benchmarks/test_pandas_perf.py --codspeed --codspeed-mode=walltime -o addopts=

WHY BOTH BACKINGS: when duckdb scans a pandas DataFrame, the binding path depends on each column's backing:
  * numpy-backed columns (dtype int64 / float64 / object) -> the NUMPY scan path (NumpyArray facade,
    RawArrayWrapper, pandas/bind.cpp, analyzer.cpp) -- this is the path the nanobind cutover reworked
    NON-TRIVIALLY, so it gets first-class coverage here.
  * arrow-backed columns (pandas ArrowDtype, e.g. int64[pyarrow]) -> the ARROW scan path (near zero-copy).
On the WRITE side, duckdb's native pandas output (rel.df()) is NUMPY-backed; an arrow-backed pandas frame is
produced via duckdb-arrow + pyarrow.to_pandas(ArrowDtype) (pyarrow.to_pandas is identical on both builds, so
the A/B delta is still the duckdb binding).

FULL CONSUME (same discipline as the arrow module): READ aggregates over the actual columns (sum/length, NOT
count(*) which is answered from metadata), and WRITE materializes the entire DataFrame.

numpy/pandas/pyarrow are pinned to the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

N = 500_000
WRITE_Q_NUM = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(500000) t(i)"
WRITE_Q_STR = "SELECT ('str_value_' || i) AS s FROM range(500000) t(i)"
_STRINGS = [f"str_value_{i}" for i in range(N)]


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def df_numpy_numeric():
    return pd.DataFrame({"a": np.arange(N, dtype="int64"), "b": np.arange(N, dtype="float64") * 1.5})


@pytest.fixture(scope="module")
def df_numpy_string():
    # explicit object dtype -> classic numpy-backed object-string column (the reworked object/analyzer path)
    return pd.DataFrame({"s": pd.array(_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_arrow_numeric():
    return pd.DataFrame(
        {
            "a": pd.array(np.arange(N), dtype=pd.ArrowDtype(pa.int64())),
            "b": pd.array(np.arange(N) * 1.5, dtype=pd.ArrowDtype(pa.float64())),
        }
    )


@pytest.fixture(scope="module")
def df_arrow_string():
    return pd.DataFrame({"s": pd.array(_STRINGS, dtype=pd.ArrowDtype(pa.string()))})


# --------------------------------------------------------------------------- #
# READ: pandas -> duckdb. Engine scans every value (sum/length force it).
# --------------------------------------------------------------------------- #


def test_read_pandas_numpy_numeric(benchmark, con, df_numpy_numeric):
    con.register("t", df_numpy_numeric)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


def test_read_pandas_numpy_string(benchmark, con, df_numpy_string):
    con.register("t", df_numpy_string)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


def test_read_pandas_arrow_numeric(benchmark, con, df_arrow_numeric):
    con.register("t", df_arrow_numeric)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


def test_read_pandas_arrow_string(benchmark, con, df_arrow_string):
    con.register("t", df_arrow_string)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# --------------------------------------------------------------------------- #
# WRITE: duckdb -> pandas. df() is NUMPY-backed (the reworked production path);
# the arrow-backed frame goes via duckdb-arrow + pyarrow.to_pandas(ArrowDtype).
# Both eagerly materialize the whole DataFrame.
# --------------------------------------------------------------------------- #


def test_write_pandas_numpy_numeric(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_NUM).df())


def test_write_pandas_numpy_string(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_STR).df())


# ADDED: the numpy-backed df() WRITE with REAL nulls -> the masked_array build + masked->pd.NA rewrite that the
# cutover reworked (a no-null column takes the cheap std::move path and would measure the wrong thing), plus a
# datetime column (TimestampConvert + ConvertDateTimeTypes).


def test_write_pandas_numpy_numeric_with_nulls(benchmark, con):
    q = (
        "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
        "CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range(500000) t(i)"
    )
    benchmark(lambda: con.sql(q).df())


def test_write_pandas_numpy_timestamp(benchmark, con):
    q = "SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range(500000) t(i)"
    benchmark(lambda: con.sql(q).df())


def test_write_pandas_arrow_numeric(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))


def test_write_pandas_arrow_string(benchmark, con):
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))
