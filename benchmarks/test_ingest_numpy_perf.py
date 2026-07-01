"""Standalone CodSpeed benchmark module for the NUMPY ingest paths (numpy / numpy-backed pandas -> duckdb)
— NOT integrated (not in pyproject, not in CI, not committed). Run under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_ingest_numpy_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_ingest_numpy_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

WHY THIS MODULE: the numpy scan (NumpyScan / NumpyArray facade / RawArrayWrapper / pandas-bind / analyzer) is
the IN-numpy half the nanobind cutover reworked, and several of its branches were untested:
  * I0-2 object-string scan: the per-row isinstance + PyUnicodeIsCompactASCII zero-copy vs DecodePythonUnicode
    transcode ladder (numpy_scan.cpp). GOTCHA (encoded): a meaningful benchmark MUST mix ASCII + non-ASCII +
    a null sentinel -- ASCII-only misses the transcode + null-detection ladder entirely.
  * I0-1 double NaN->NULL loop (numpy_scan.cpp) -- the reworked float path.
  * NULL-heavy masked scan: ScanNumpyMasked + ApplyMask (pandas nullable Int64).
  * I1-3 analyzer bind: PandasAnalyzer::Analyze samples rows through the GetItemType ladder. This is a per-BIND
    cost, independent of row count, so it is the ONE place count(*) is the correct consume (the cost is at bind,
    not scan); every other READ here aggregates over real columns (sum/length) to force a full engine scan.
  * I1-8 numpy ndarray / dict-of-arrays via the replacement scan (resolved from a module global).

numpy/pandas are pinned to the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import numpy as np
import pandas as pd
import pytest

N = 500_000
ANALYZER_N = 200_000

# Module-global for the replacement-scan-from-variable path (frame resolution finds f_globals reliably).
NPDICT = {"a": np.arange(N, dtype="int64"), "b": np.arange(N, dtype="float64") * 1.5}

# Mixed ASCII + non-ASCII + null sentinel -> forces the transcode + null-detection ladder (NOT ASCII-only).
_MIXED = ["ascii_value_", "café_", "naïve_", "日本語_", None]
_MIXED_STRINGS = [None if _MIXED[i % 5] is None else f"{_MIXED[i % 5]}{i}" for i in range(N)]

# Mixed python types in an object column -> the analyzer must sample/widen through the type ladder at bind.
_MIXED_TYPES = [(i if i % 3 == 0 else (float(i) if i % 3 == 1 else f"s{i}")) for i in range(ANALYZER_N)]


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def df_double_with_nan():
    a = np.arange(N, dtype="float64") * 1.5
    a[::10] = np.nan  # real NaNs -> NaN->NULL conversion loop
    return pd.DataFrame({"a": a})


@pytest.fixture(scope="module")
def df_object_string_mixed():
    return pd.DataFrame({"s": pd.array(_MIXED_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_masked_int():
    # pandas nullable Int64 -> numpy values + validity mask -> ScanNumpyMasked + ApplyMask
    arr = pd.array(np.arange(N), dtype="Int64")
    arr[::10] = pd.NA
    return pd.DataFrame({"a": arr})


@pytest.fixture(scope="module")
def df_object_mixed_types():
    return pd.DataFrame({"v": pd.array(_MIXED_TYPES, dtype=object)})


# --------------------------------------------------------------------------- #
# READ: numpy -> duckdb. Engine scans every value (sum/length force it).
# --------------------------------------------------------------------------- #


def test_read_numpy_dict_numeric(benchmark, con):
    benchmark(lambda: con.sql("SELECT sum(a), sum(b) FROM NPDICT").fetchall())


def test_read_numpy_double_with_nan(benchmark, con, df_double_with_nan):
    con.register("t", df_double_with_nan)
    benchmark(lambda: con.execute("SELECT sum(a) FROM t").fetchall())


def test_read_numpy_masked_int(benchmark, con, df_masked_int):
    con.register("t", df_masked_int)
    benchmark(lambda: con.execute("SELECT sum(a) FROM t").fetchall())


def test_read_numpy_object_string_mixed(benchmark, con, df_object_string_mixed):
    con.register("t", df_object_string_mixed)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# --------------------------------------------------------------------------- #
# BIND: PandasAnalyzer sampling cost. count(*) is correct HERE ONLY -- the cost is at bind, not scan, so we
# must NOT force a scan (that would drown the per-bind analyzer signal). Re-binds the object column each call.
# --------------------------------------------------------------------------- #


def test_bind_analyzer_object(benchmark, con, df_object_mixed_types):
    con.register("t", df_object_mixed_types)
    con.execute("SELECT count(*) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(*) FROM t").fetchall())
