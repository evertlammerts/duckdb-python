"""CodSpeed benchmark: pandas read/write, numpy-backed vs arrow-backed DataFrames. Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_pandas_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

The binding path depends on column backing: numpy-backed columns take the NumpyArray scan path, arrow-backed
(pandas ArrowDtype) take the near-zero-copy arrow path. Full consume: READ aggregates over real columns (not
count(*)), WRITE materializes the whole frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from _scale import scaled

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(500_000)  # env-gated: full N locally, shrunk under BENCH_SCALE in the CI Callgrind sweep (INFRA-4)
WRITE_Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
WRITE_Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"
_STRINGS = [f"str_value_{i}" for i in range(N)]

# `con` fixture + threads=1 live in conftest.py. READ benchmarks (`sum()` over a registered frame) are
# engine-aggregate dominated -> informational. Only the NUMPY-backed df() WRITE is binding-dominated -> gate.
# The arrow-backed WRITE goes through to_arrow_table().to_pandas() (pyarrow library code, MEAS-2) -> informational.


@pytest.fixture(scope="module")
def df_numpy_numeric() -> pd.DataFrame:
    """Return a numpy-backed numeric frame."""
    return pd.DataFrame({"a": np.arange(N, dtype="int64"), "b": np.arange(N, dtype="float64") * 1.5})


@pytest.fixture(scope="module")
def df_numpy_string() -> pd.DataFrame:
    """Return a numpy-backed object-string frame."""
    # explicit object dtype -> classic numpy-backed object-string column (the reworked object/analyzer path)
    return pd.DataFrame({"s": pd.array(_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_arrow_numeric() -> pd.DataFrame:
    """Return an arrow-backed numeric frame."""
    return pd.DataFrame(
        {
            "a": pd.array(np.arange(N), dtype=pd.ArrowDtype(pa.int64())),
            "b": pd.array(np.arange(N) * 1.5, dtype=pd.ArrowDtype(pa.float64())),
        }
    )


@pytest.fixture(scope="module")
def df_arrow_string() -> pd.DataFrame:
    """Return an arrow-backed string frame."""
    return pd.DataFrame({"s": pd.array(_STRINGS, dtype=pd.ArrowDtype(pa.string()))})


# --------------------------------------------------------------------------- #
# READ: pandas -> duckdb. Engine scans every value (sum/length force it).
# --------------------------------------------------------------------------- #


@pytest.mark.informational
def test_read_pandas_numpy_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_numpy_numeric: pd.DataFrame
) -> None:
    """Benchmark scanning a numpy-backed numeric frame."""
    con.register("t", df_numpy_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t").fetchall()  # warm (MEAS-3)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_numpy_string(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_numpy_string: pd.DataFrame
) -> None:
    """Benchmark scanning a numpy-backed string frame."""
    con.register("t", df_numpy_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm (MEAS-3)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_arrow_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_arrow_numeric: pd.DataFrame
) -> None:
    """Benchmark scanning an arrow-backed numeric frame."""
    con.register("t", df_arrow_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t").fetchall()  # warm (MEAS-3)
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_arrow_string(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_arrow_string: pd.DataFrame
) -> None:
    """Benchmark scanning an arrow-backed string frame."""
    con.register("t", df_arrow_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm (MEAS-3)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# --------------------------------------------------------------------------- #
# WRITE: duckdb -> pandas. df() is NUMPY-backed (the reworked production path);
# the arrow-backed frame goes via duckdb-arrow + pyarrow.to_pandas(ArrowDtype).
# Both eagerly materialize the whole DataFrame.
# --------------------------------------------------------------------------- #


@pytest.mark.gate
def test_write_pandas_numpy_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a numeric result to a numpy-backed frame."""
    benchmark(lambda: con.sql(WRITE_Q_NUM).df())


@pytest.mark.gate
def test_write_pandas_numpy_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a string result to a numpy-backed frame."""
    benchmark(lambda: con.sql(WRITE_Q_STR).df())


# ADDED: the numpy-backed df() WRITE with REAL nulls -> the masked_array build + masked->pd.NA rewrite that the
# cutover reworked (a no-null column takes the cheap std::move path and would measure the wrong thing), plus a
# datetime column (TimestampConvert + ConvertDateTimeTypes).


@pytest.mark.gate
def test_write_pandas_numpy_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a null-heavy numeric result to a numpy-backed frame."""
    q = (
        "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
        f"CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range({N}) t(i)"
    )
    benchmark(lambda: con.sql(q).df())


@pytest.mark.gate
def test_write_pandas_numpy_timestamp(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a timestamp result to a numpy-backed frame."""
    q = f"SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range({N}) t(i)"
    benchmark(lambda: con.sql(q).df())


@pytest.mark.informational  # to_arrow_table().to_pandas() -> the to_pandas half is pyarrow library code (MEAS-2)
def test_write_pandas_arrow_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a numeric result to an arrow-backed frame."""
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))


@pytest.mark.informational  # to_arrow_table().to_pandas() -> the to_pandas half is pyarrow library code (MEAS-2)
def test_write_pandas_arrow_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark materializing a string result to an arrow-backed frame."""
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))
