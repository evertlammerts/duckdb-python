"""CodSpeed benchmark: numpy ingest paths (numpy / numpy-backed pandas -> duckdb). Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_ingest_numpy_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Covers the object-string scan (ASCII zero-copy vs transcode ladder), the NaN->NULL float loop, the masked
scan, and analyzer bind. Gotchas: the object-string benchmark MUST mix ASCII + non-ASCII + a null or it misses
the ladder; analyzer bind is the one place count(*) is correct (cost is at bind, not scan) while every other
READ aggregates over real columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

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
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh connection, closed on teardown."""
    c = duckdb.connect()
    yield c
    c.close()


@pytest.fixture(scope="module")
def df_double_with_nan() -> pd.DataFrame:
    """Return a numpy-backed double frame with real NaNs."""
    a = np.arange(N, dtype="float64") * 1.5
    a[::10] = np.nan  # real NaNs -> NaN->NULL conversion loop
    return pd.DataFrame({"a": a})


@pytest.fixture(scope="module")
def df_object_string_mixed() -> pd.DataFrame:
    """Return an object-string frame mixing ASCII, non-ASCII, and nulls."""
    return pd.DataFrame({"s": pd.array(_MIXED_STRINGS, dtype=object)})


@pytest.fixture(scope="module")
def df_masked_int() -> pd.DataFrame:
    """Return a nullable-Int64 frame that scans masked."""
    # pandas nullable Int64 -> numpy values + validity mask -> ScanNumpyMasked + ApplyMask
    arr = pd.array(np.arange(N), dtype="Int64")
    arr[::10] = pd.NA
    return pd.DataFrame({"a": arr})


@pytest.fixture(scope="module")
def df_object_mixed_types() -> pd.DataFrame:
    """Return an object frame of mixed python types for analyzer bind."""
    return pd.DataFrame({"v": pd.array(_MIXED_TYPES, dtype=object)})


# --------------------------------------------------------------------------- #
# READ: numpy -> duckdb. Engine scans every value (sum/length force it).
# --------------------------------------------------------------------------- #


def test_read_numpy_dict_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark scanning a numpy-dict replacement scan."""
    benchmark(lambda: con.sql("SELECT sum(a), sum(b) FROM NPDICT").fetchall())


def test_read_numpy_double_with_nan(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_double_with_nan: pd.DataFrame
) -> None:
    """Benchmark scanning a numpy double column with NaNs."""
    con.register("t", df_double_with_nan)
    benchmark(lambda: con.execute("SELECT sum(a) FROM t").fetchall())


def test_read_numpy_masked_int(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_masked_int: pd.DataFrame
) -> None:
    """Benchmark scanning a masked nullable-int column."""
    con.register("t", df_masked_int)
    benchmark(lambda: con.execute("SELECT sum(a) FROM t").fetchall())


def test_read_numpy_object_string_mixed(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_object_string_mixed: pd.DataFrame
) -> None:
    """Benchmark scanning a mixed object-string column."""
    con.register("t", df_object_string_mixed)
    benchmark(lambda: con.execute("SELECT count(s), sum(length(s)) FROM t").fetchall())


# --------------------------------------------------------------------------- #
# BIND: PandasAnalyzer sampling cost. count(*) is correct HERE ONLY -- the cost is at bind, not scan, so we
# must NOT force a scan (that would drown the per-bind analyzer signal). Re-binds the object column each call.
# --------------------------------------------------------------------------- #


def test_bind_analyzer_object(
    benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, df_object_mixed_types: pd.DataFrame
) -> None:
    """Benchmark the analyzer bind of a mixed-type object column."""
    con.register("t", df_object_mixed_types)
    con.execute("SELECT count(*) FROM t").fetchall()  # warm
    benchmark(lambda: con.execute("SELECT count(*) FROM t").fetchall())
