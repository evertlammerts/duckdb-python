"""pandas read/write, numpy-backed vs arrow-backed frames. See benchmarks/README.md.

Column backing selects the path: numpy-backed -> NumpyArray scan; arrow-backed (ArrowDtype) -> zero-copy arrow.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest
from _scale import scaled

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

N = scaled(500_000)
WRITE_Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
WRITE_Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"
_STRINGS = [f"str_value_{i}" for i in range(N)]

# READ (sum over a registered frame) is engine-aggregate dominated -> informational. Only the NUMPY-backed df()
# WRITE is binding-dominated -> gate; the arrow-backed WRITE goes through pyarrow's to_pandas -> informational.


@pytest.fixture(scope="module")
def df_numpy_numeric() -> pd.DataFrame:
    return pd.DataFrame({"a": np.arange(N, dtype="int64"), "b": np.arange(N, dtype="float64") * 1.5})


@pytest.fixture(scope="module")
def df_numpy_string() -> pd.DataFrame:
    # explicit object dtype -> the reworked numpy-backed object-string / analyzer path
    return pd.DataFrame({"s": pd.Series(_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_arrow_numeric() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": pd.array(np.arange(N), dtype=pd.ArrowDtype(pa.int64())),
            "b": pd.array(np.arange(N) * 1.5, dtype=pd.ArrowDtype(pa.float64())),
        }
    )


@pytest.fixture(scope="module")
def df_arrow_string() -> pd.DataFrame:
    return pd.DataFrame({"s": pd.array(_STRINGS, dtype=pd.ArrowDtype(pa.string()))})


@pytest.fixture(scope="module", params=["naive", "aware"])
def df_object_datetime(request: pytest.FixtureRequest) -> pd.DataFrame:
    # explicit object dtype keeps datetime objects, so the scan converts each one instead of a datetime64 column
    zone = datetime.timezone(datetime.timedelta(hours=2)) if request.param == "aware" else None
    start = datetime.datetime(2020, 1, 1, 12, 30, 15, 250_000, tzinfo=zone)
    return pd.DataFrame({"t": pd.Series([start + datetime.timedelta(seconds=i) for i in range(N)], dtype=object)})


# READ: pandas -> duckdb. sum/length force a full scan.


@pytest.mark.informational
def test_read_pandas_numpy_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_numpy_numeric: pd.DataFrame
) -> None:
    con.register("t", df_numpy_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_numpy_string(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_numpy_string: pd.DataFrame
) -> None:
    con.register("t", df_numpy_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_object_datetime(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_object_datetime: pd.DataFrame
) -> None:
    con.register("t", df_object_datetime)
    con.execute("SELECT count(t), max(t) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(t), max(t) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_arrow_numeric(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_arrow_numeric: pd.DataFrame
) -> None:
    con.register("t", df_arrow_numeric)
    con.execute("SELECT sum(a), sum(b) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM t").fetchall())


@pytest.mark.informational
def test_read_pandas_arrow_string(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_arrow_string: pd.DataFrame
) -> None:
    con.register("t", df_arrow_string)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# WRITE: duckdb -> pandas. df() is the reworked numpy-backed path; the arrow-backed frame goes via
# duckdb-arrow + pyarrow.to_pandas(ArrowDtype). Both eagerly materialize the whole frame.


@pytest.mark.gate
def test_write_pandas_numpy_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_NUM).df())


@pytest.mark.gate
def test_write_pandas_numpy_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_STR).df())


@pytest.mark.gate
def test_write_pandas_numpy_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    # REAL nulls -> the masked_array build + masked-to-pd.NA rewrite the cutover reworked (see README traps)
    q = (
        "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
        f"CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range({N}) t(i)"
    )
    benchmark(lambda: con.sql(q).df())


@pytest.mark.gate
def test_write_pandas_numpy_timestamp(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    q = f"SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range({N}) t(i)"
    benchmark(lambda: con.sql(q).df())


@pytest.mark.informational  # to_pandas() half is pyarrow library code
def test_write_pandas_arrow_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_NUM).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))


@pytest.mark.informational  # to_pandas() half is pyarrow library code
def test_write_pandas_arrow_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    benchmark(lambda: con.sql(WRITE_Q_STR).to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype))
