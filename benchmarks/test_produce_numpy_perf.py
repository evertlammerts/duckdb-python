"""CodSpeed benchmark: columnar produce paths (df(), fetchnumpy(), fetch_df_chunk()). Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_produce_numpy_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Covers the with-NULLS branch (masked_array build), datetime, and wide-internal types (hugeint/uuid/decimal128).
Gotcha: NULL benchmarks use real DuckDB nulls (CASE WHEN); a no-null column takes the cheap path and measures
the wrong thing. Full consume: df()/fetchnumpy() materialize the columns; fetch_df_chunk is drained in a loop.
"""

from __future__ import annotations

import gc
import sys
import tracemalloc
from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb
import numpy as np  # noqa: F401  (pinned identically A/B; imported so the env matches the other modules)

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

N = scaled(500_000)  # env-gated: full N locally, shrunk under BENCH_SCALE in the CI Callgrind sweep (INFRA-4)
TYPE_N = scaled(200_000)  # wide-internal types (hugeint/uuid/decimal128) are heavier per cell

Q_NUM = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N}) t(i)"
Q_NUM_NULLS = (
    "SELECT CASE WHEN i % 10 = 0 THEN NULL ELSE i::BIGINT END AS a, "
    f"CASE WHEN i % 10 = 0 THEN NULL ELSE (i * 1.5)::DOUBLE END AS b FROM range({N}) t(i)"
)
Q_STR = f"SELECT ('str_value_' || i) AS s FROM range({N}) t(i)"
Q_TS = f"SELECT TIMESTAMP '2020-01-01' + (i * INTERVAL 1 SECOND) AS t FROM range({N}) t(i)"
Q_HUGEINT = f"SELECT (i::HUGEINT * 1000000000000) AS h FROM range({TYPE_N}) t(i)"
Q_UUID = f"SELECT gen_random_uuid() AS u FROM range({TYPE_N}) t(i)"
Q_DEC128 = f"SELECT ((i * 1.5)::DECIMAL(28, 6)) AS d FROM range({TYPE_N}) t(i)"


# gate: df()/fetchnumpy() fully materialize numpy-backed columns -> binding-dominated (ArrayWrapper fill),
# GIL-held, deterministic under Callgrind. `con` fixture + threads=1 live in conftest.py.
def _bench_df(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.sql(query).df()  # warm
    benchmark(lambda: con.sql(query).df())


def _bench_numpy(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.sql(query).fetchnumpy()  # warm
    benchmark(lambda: con.sql(query).fetchnumpy())


# --------------------------------------------------------------------------- #
# df(): the production NUMPY-backed columnar path. no-null vs REAL-null vs string vs timestamp.
# --------------------------------------------------------------------------- #


@pytest.mark.gate
def test_df_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a numeric result."""
    _bench_df(benchmark, con, Q_NUM)


@pytest.mark.gate
def test_df_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a null-heavy numeric result."""
    # REAL nulls -> HAS_NULLS=true -> masked_array build + masked->pd.NA rewrite (the reworked branch)
    _bench_df(benchmark, con, Q_NUM_NULLS)


@pytest.mark.gate
def test_df_string(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a string result."""
    _bench_df(benchmark, con, Q_STR)


@pytest.mark.gate
def test_df_timestamp(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a timestamp result."""
    _bench_df(benchmark, con, Q_TS)


@pytest.mark.gate
def test_df_hugeint(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a hugeint result."""
    _bench_df(benchmark, con, Q_HUGEINT)


@pytest.mark.gate
def test_df_uuid(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a uuid result."""
    _bench_df(benchmark, con, Q_UUID)


@pytest.mark.gate
def test_df_decimal128(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark df() of a 128-bit decimal result."""
    _bench_df(benchmark, con, Q_DEC128)


# --------------------------------------------------------------------------- #
# fetchnumpy(): same FetchNumpyInternal without the DataFrame wrap.
# --------------------------------------------------------------------------- #


@pytest.mark.gate
def test_fetchnumpy_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchnumpy() of a numeric result."""
    _bench_numpy(benchmark, con, Q_NUM)


@pytest.mark.gate
def test_fetchnumpy_numeric_with_nulls(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchnumpy() of a null-heavy numeric result."""
    _bench_numpy(benchmark, con, Q_NUM_NULLS)


# --------------------------------------------------------------------------- #
# fetch_df_chunk(): per-chunk DataFrame production, drained in a loop.
# --------------------------------------------------------------------------- #


@pytest.mark.informational  # per-chunk streaming drain (GIL-per-chunk) -> walltime-informational, not gated
def test_fetch_df_chunk_loop(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark draining a result with fetch_df_chunk()."""

    def run() -> int:
        rel = con.sql(Q_NUM)
        rows = 0
        while True:
            chunk = rel.fetch_df_chunk()
            if len(chunk) == 0:
                break
            rows += len(chunk)
        return rows

    con.sql(Q_NUM).fetch_df_chunk()  # warm
    benchmark(run)


# --------------------------------------------------------------------------- #
# torch(): FetchNumpyInternal + per-column from_numpy. SKIPPED cleanly if torch is absent (identical A/B).
# --------------------------------------------------------------------------- #


@pytest.mark.informational  # torch is local-only (importorskip -> skipped in CI); torch lib work dilutes it
def test_torch_numeric(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark torch() of a numeric result (skipped if torch is absent)."""
    pytest.importorskip("torch")
    q = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({TYPE_N}) t(i)"
    con.sql(q).torch()  # warm
    benchmark(lambda: con.sql(q).torch())


# --------------------------------------------------------------------------- #
# MEMORY GUARD (secondary signal, not a codspeed benchmark). codspeed walltime tracks neither memory nor
# allocations, and conversion regressions are often memory-shaped (the recorded fetchall list->tuple edge-copy;
# the df() masked_array branch). We use tracemalloc to capture the PEAK Python-tracked allocation of ONE
# df()-with-nulls call. Correctness notes:
#   * reset_peak() is called AFTER the warm (and after freeing the warm result) so the warm does not establish
#     a high-water mark that swallows the measured call -- the prior getrusage(ru_maxrss) version was broken
#     precisely because ru_maxrss is monotonic and the warm pre-set the peak, making the delta ~0.
#   * tracemalloc reports BYTES on every platform (no macOS-bytes / Linux-KiB skew that the getrusage version
#     had), so the ceiling is portable to the Linux CI target.
# CAVEAT: tracemalloc only sees Python-level allocations; the raw numpy column buffers are allocated in C and
# are NOT visible here. So this catches a gross PYTHON-object-shaped blowup (the masked->pd.NA rewrite / a
# per-row object materialization regression) but is not a total-RSS gate -- the authoritative CI gate for the
# C-buffer payload is codspeed memory mode (--codspeed-mode=memory).
# --------------------------------------------------------------------------- #


def test_mem_df_with_nulls() -> None:
    """Guard the Python-tracked peak allocation of a null-heavy df() call."""
    con = duckdb.connect(config={"threads": 1})
    try:
        tracemalloc.start()
        warm = con.sql(Q_NUM_NULLS).df()  # populate one-time import / type caches
        del warm
        gc.collect()
        tracemalloc.reset_peak()  # discount the warm's transient peak BEFORE the measured call
        out = con.sql(Q_NUM_NULLS).df()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del out
    finally:
        con.close()
    print(f"\n[mem] df()-with-nulls tracemalloc peak = {peak / 1e6:.1f} MB", file=sys.stderr)
    # Python-tracked allocations for a 500k x 2-col masked df are a few MB; a gross conversion-memory blowup
    # (e.g. a per-row Python object list, the masked->pd.NA rewrite gone wrong) is tens+ MB. 100 MB ceiling
    # catches that without flaking, and is bytes on all platforms.
    assert peak < 100_000_000
