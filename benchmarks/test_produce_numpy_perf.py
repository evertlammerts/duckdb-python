"""Standalone CodSpeed benchmark module for the COLUMNAR produce paths (duckdb -> numpy/pandas), i.e. df(),
fetchnumpy(), fetch_df_chunk() — NOT integrated (not in pyproject, not in CI, not committed). Run under each
build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_produce_numpy_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_produce_numpy_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

WHY THIS MODULE: the columnar OUT path (FetchNumpyInternal -> ArrayWrapper ConvertColumnRegular) is exactly
what the nanobind cutover reworked. The under-covered cases are: (1) the WITH-NULLS branch (HAS_NULLS=true ->
masked_array build -> masked->pd.NA rewrite, array_wrapper.cpp / pyresult.cpp) -- NEVER previously benchmarked
and the most-changed code; (2) datetime; (3) fetchnumpy without the DataFrame wrap; (4) fetch_df_chunk; and
the wide-internal types HUGEINT (->double cast), UUID (UUIDConvert), DECIMAL(28,x) (ConvertDecimalInternal
<hugeint_t>) that exercise distinct OUT-col converters.

GOTCHA (encoded below): OUT-col NULL benchmarks use REAL DuckDB nulls (CASE WHEN .. THEN NULL). A no-null
column silently takes the cheap std::move path and the masked-array branch never triggers, so it would measure
the wrong thing.

FULL CONSUME: df() / fetchnumpy() eagerly materialize the whole column set; fetch_df_chunk is drained in a loop.

numpy/pandas are pinned to the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import gc
import sys
import tracemalloc

import duckdb
import numpy as np  # noqa: F401  (pinned identically A/B; imported so the env matches the other modules)
import pytest

N = 500_000
TYPE_N = 200_000  # wide-internal types (hugeint/uuid/decimal128) are heavier per cell

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


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def _bench_df(benchmark, con, query):
    con.sql(query).df()  # warm
    benchmark(lambda: con.sql(query).df())


def _bench_numpy(benchmark, con, query):
    con.sql(query).fetchnumpy()  # warm
    benchmark(lambda: con.sql(query).fetchnumpy())


# --------------------------------------------------------------------------- #
# df(): the production NUMPY-backed columnar path. no-null vs REAL-null vs string vs timestamp.
# --------------------------------------------------------------------------- #


def test_df_numeric(benchmark, con):
    _bench_df(benchmark, con, Q_NUM)


def test_df_numeric_with_nulls(benchmark, con):
    # REAL nulls -> HAS_NULLS=true -> masked_array build + masked->pd.NA rewrite (the reworked branch)
    _bench_df(benchmark, con, Q_NUM_NULLS)


def test_df_string(benchmark, con):
    _bench_df(benchmark, con, Q_STR)


def test_df_timestamp(benchmark, con):
    _bench_df(benchmark, con, Q_TS)


def test_df_hugeint(benchmark, con):
    _bench_df(benchmark, con, Q_HUGEINT)


def test_df_uuid(benchmark, con):
    _bench_df(benchmark, con, Q_UUID)


def test_df_decimal128(benchmark, con):
    _bench_df(benchmark, con, Q_DEC128)


# --------------------------------------------------------------------------- #
# fetchnumpy(): same FetchNumpyInternal without the DataFrame wrap.
# --------------------------------------------------------------------------- #


def test_fetchnumpy_numeric(benchmark, con):
    _bench_numpy(benchmark, con, Q_NUM)


def test_fetchnumpy_numeric_with_nulls(benchmark, con):
    _bench_numpy(benchmark, con, Q_NUM_NULLS)


# --------------------------------------------------------------------------- #
# fetch_df_chunk(): per-chunk DataFrame production, drained in a loop.
# --------------------------------------------------------------------------- #


def test_fetch_df_chunk_loop(benchmark, con):
    def run():
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


def test_torch_numeric(benchmark, con):
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


def test_mem_df_with_nulls():
    con = duckdb.connect()
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
