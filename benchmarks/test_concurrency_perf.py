"""CodSpeed benchmark: concurrency / GIL pressure (COV-1). informational / WALLTIME. Standalone, not gated.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_concurrency_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

This is the ONE dimension the rest of the suite (single-threaded) cannot see: Python objects threading through
PARALLEL core execution. It varies `SET threads` and measures the binding under parallel scan / parallel UDF
invocation. All benchmarks are `informational` and their PRIMARY signal is LOCAL WALLTIME:
  * scan benches           -> parallel speedup; a per-batch Produce GIL regression shows as reduced speedup.
  * native UDF             -> ~flat scaling = the GIL tax on per-row Python calls (the engine scan is parallel
                              but the GIL serializes the calls).
  * arrow (vectorized) UDF -> observed NEGATIVE scaling (slower with more threads): per-chunk convert + GIL
                              contention. A regression here would deepen the negative slope.

Under the CI `-m informational` step these run in `simulation` (Callgrind), which SERIALIZES threads -- so the
wall-clock contention is NOT visible there; instead the deterministic instruction count captures the per-batch
Produce GIL calls and the UDF dispatch overhead. Never gated either way.

GOTCHA (verified locally, mirrors the suite's other "measure the right thing" traps): a SINGLE-BATCH arrow table
does NOT parallelize (one batch = one serial scan unit; flat across threads). The arrow scan bench MUST use a
MULTI-BATCH table (`from_batches` with a modest chunksize) or it silently measures a serial scan. A CPU-heavy
aggregate is also required: a cheap sum is memory-bandwidth-bound and will not parallelize, so there is nothing
to contend on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb.sqltypes import BIGINT

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")
import numpy as np  # noqa: E402  (after importorskip, matching the suite convention)
import pandas as pd  # noqa: E402

# informational: concurrency benchmarks are never gated (walltime-noisy; under Callgrind, thread-serialized).
pytestmark = pytest.mark.informational

N_SCAN = 1_000_000
BATCH = 20_000  # -> 50 record batches; MULTI-BATCH is required for the arrow scan to parallelize (see GOTCHA)
N_UDF_NATIVE = 200_000  # native UDF = one Python call per row; keep modest (Callgrind instruments every call)
N_UDF_ARROW = 1_000_000  # arrow UDF = one call per chunk (vectorized)
THREADS = [1, 4, 8]

# CPU-heavy aggregate so the parallel scan actually engages worker threads (a cheap sum is bandwidth-bound and
# would not parallelize -> no contention to measure). The binding signal is the per-batch Produce GIL handoff.
HEAVY = "sin(a) * cos(b) + sqrt(abs(a)) + ln(abs(a) + 1)"


@pytest.fixture(scope="module")
def arrow_multibatch() -> pa.Table:
    """Return a MULTI-batch arrow table (single-batch would scan serially -- see module GOTCHA)."""
    a = pa.array(np.arange(N_SCAN), type=pa.int64())
    b = pa.array(np.arange(N_SCAN, dtype="float64") * 1.5, type=pa.float64())
    return pa.Table.from_batches(pa.table({"a": a, "b": b}).to_batches(max_chunksize=BATCH))


@pytest.fixture(scope="module")
def pandas_frame() -> pd.DataFrame:
    """Return a numpy-backed pandas frame (its scan parallelizes across worker threads)."""
    return pd.DataFrame({"a": np.arange(N_SCAN), "b": np.arange(N_SCAN, dtype="float64") * 1.5})


# --------------------------------------------------------------------------- #
# Parallel SCAN: Python objects (arrow batches / pandas chunks) pulled through the binding by engine worker
# threads under a CPU-heavy aggregate. The scan Produce acquires/releases the GIL per batch across threads.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("threads", THREADS)
def test_scan_arrow_parallel(benchmark: BenchmarkFixture, arrow_multibatch: pa.Table, threads: int) -> None:
    """Benchmark a parallel aggregate pulling arrow batches across threads."""
    con = duckdb.connect(config={"threads": threads})
    try:
        con.register("t", arrow_multibatch)
        q = f"SELECT sum({HEAVY}) FROM t"
        con.execute(q).fetchall()  # warm
        benchmark(lambda: con.execute(q).fetchall())
    finally:
        con.close()


@pytest.mark.parametrize("threads", THREADS)
def test_scan_pandas_parallel(benchmark: BenchmarkFixture, pandas_frame: pd.DataFrame, threads: int) -> None:
    """Benchmark a parallel aggregate pulling pandas chunks across threads."""
    con = duckdb.connect(config={"threads": threads})
    try:
        con.register("t", pandas_frame)
        q = f"SELECT sum({HEAVY}) FROM t"
        con.execute(q).fetchall()  # warm
        benchmark(lambda: con.execute(q).fetchall())
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Parallel UDF: the engine scans a MATERIALIZED table (range() does not parallelize) and invokes a Python UDF
# from multiple worker threads. Native = per-row Python call under the GIL (GIL tax); arrow = per-chunk convert.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("threads", THREADS)
def test_udf_native_parallel(benchmark: BenchmarkFixture, threads: int) -> None:
    """Benchmark a native Python UDF invoked from parallel worker threads (GIL tax)."""
    con = duckdb.connect(config={"threads": threads})
    try:
        con.execute(f"CREATE TABLE t AS SELECT i AS a FROM range({N_UDF_NATIVE}) s(i)")  # materialized -> parallel scan
        con.create_function("pyf", lambda x: (x * 2 + 1) % 97, [BIGINT], BIGINT)
        con.execute("SELECT sum(pyf(a)) FROM t").fetchall()  # warm
        benchmark(lambda: con.execute("SELECT sum(pyf(a)) FROM t").fetchall())
    finally:
        con.close()


@pytest.mark.parametrize("threads", THREADS)
def test_udf_arrow_parallel(benchmark: BenchmarkFixture, threads: int) -> None:
    """Benchmark a vectorized arrow UDF invoked from parallel worker threads."""
    con = duckdb.connect(config={"threads": threads})
    try:
        con.execute(f"CREATE TABLE t AS SELECT i AS a FROM range({N_UDF_ARROW}) s(i)")  # materialized -> parallel scan
        con.create_function("af", lambda x: pc.add(x, 1), [BIGINT], BIGINT, type="arrow")
        con.execute("SELECT sum(af(a)) FROM t").fetchall()  # warm
        benchmark(lambda: con.execute("SELECT sum(af(a)) FROM t").fetchall())
    finally:
        con.close()
