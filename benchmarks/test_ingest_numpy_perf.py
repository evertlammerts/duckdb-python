"""numpy ingest: object-string scan, NaN-to-NULL, masked scan, analyzer bind. See benchmarks/README.md.

Gotchas: the object-string bench MUST mix ASCII + non-ASCII + a null or it misses the transcode ladder (see
README traps); analyzer bind is the one place count(*) is correct (cost is at bind, not scan).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

# scaling changes ONLY the row count, never the mixed ASCII+non-ASCII+null pattern below.
N = scaled(500_000)
ANALYZER_N = scaled(200_000)

NPDICT = {"a": np.arange(N, dtype="int64"), "b": np.arange(N, dtype="float64") * 1.5}

# mixed ASCII + non-ASCII + null sentinel -> forces the transcode + null-detection ladder (NOT ASCII-only)
_MIXED = ["ascii_value_", "café_", "naïve_", "日本語_", None]
_MIXED_STRINGS = [None if _MIXED[i % 5] is None else f"{_MIXED[i % 5]}{i}" for i in range(N)]

# mixed python types in an object column -> the analyzer must sample/widen through the type ladder at bind
_MIXED_TYPES = [(i if i % 3 == 0 else (float(i) if i % 3 == 1 else f"s{i}")) for i in range(ANALYZER_N)]

# READ (sum over a registered frame) is engine-aggregate dominated -> informational. The analyzer BIND (count(*),
# no scan) is a pure per-bind binding cost -> gate.


@pytest.fixture(scope="module")
def df_double_with_nan() -> pd.DataFrame:
    a = np.arange(N, dtype="float64") * 1.5
    a[::10] = np.nan  # real NaNs -> NaN-to-NULL conversion loop
    return pd.DataFrame({"a": a})


@pytest.fixture(scope="module")
def df_object_string_mixed() -> pd.DataFrame:
    return pd.DataFrame({"s": pd.array(_MIXED_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_masked_int() -> pd.DataFrame:
    # pandas nullable Int64 -> numpy values + validity mask -> ScanNumpyMasked + ApplyMask
    arr = pd.array(np.arange(N), dtype="Int64")
    arr[::10] = pd.NA
    return pd.DataFrame({"a": arr})


@pytest.fixture(scope="module")
def df_object_mixed_types() -> pd.DataFrame:
    return pd.DataFrame({"v": pd.array(_MIXED_TYPES, dtype=object)})


# READ: numpy -> duckdb. sum/length force a full scan.


@pytest.mark.informational
def test_read_numpy_dict_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    con.register("npdict", NPDICT)  # register explicitly, not via replacement-scan frame inspection
    con.execute("SELECT sum(a), sum(b) FROM npdict").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a), sum(b) FROM npdict").fetchall())


@pytest.mark.informational
def test_read_numpy_double_with_nan(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_double_with_nan: pd.DataFrame
) -> None:
    con.register("t", df_double_with_nan)
    con.execute("SELECT sum(a) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a) FROM t").fetchall())


@pytest.mark.informational
def test_read_numpy_masked_int(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_masked_int: pd.DataFrame
) -> None:
    con.register("t", df_masked_int)
    con.execute("SELECT sum(a) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT sum(a) FROM t").fetchall())


@pytest.mark.informational
def test_read_numpy_object_string_mixed(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_object_string_mixed: pd.DataFrame
) -> None:
    con.register("t", df_object_string_mixed)
    con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# BIND: PandasAnalyzer sampling cost. count(*) is correct HERE ONLY: the cost is at bind, so forcing a scan would
# drown the per-bind signal. Re-binds the object column each call.


@pytest.mark.gate
def test_bind_analyzer_object(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_object_mixed_types: pd.DataFrame
) -> None:
    con.register("t", df_object_mixed_types)
    con.execute("SELECT count(*) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(*) FROM t").fetchall())
