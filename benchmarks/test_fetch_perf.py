"""CodSpeed benchmark: row fetch paths (fetchall, fetchone iteration, expression construction). Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_fetch_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Only walltime works locally (no Valgrind on macOS arm64); the deterministic instruction-count mode needs Linux (CI).
Walltime is noisy on sub-ms benchmarks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

# gate: OUT-row fetch fully materializes every row to Python -> binding-dominated, GIL-held; the engine side is
# a cheap range() scan. Deterministic under Callgrind -> instruction-count gate-able. (The small-N *_gate tests
# are the compile+fetch fixed-cost variants; see MEAS-1.) The `con` fixture + threads=1 live in conftest.py.
pytestmark = pytest.mark.gate

# env-gated row counts (INFRA-4): full N locally, shrunk under BENCH_SCALE in the CI Callgrind sweep. The 2048
# small-N *_gate probes are intentionally NOT scaled (they are the compile+fetch fixed-cost baseline).
N_ROW = scaled(200_000)  # per-row-object numeric fetch (BIGINT/INTEGER/DOUBLE/2col/null/decimal128)
N_STR = scaled(100_000)  # varchar/blob/mixed-wide/timestamptz + fetchone/fetchmany loops
N_NEST = scaled(50_000)  # heterogeneous scalar/list/struct row


def _bench_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.execute(query).fetchall()  # warm the engine before measuring
    benchmark(lambda: con.execute(query).fetchall())


def test_fetchall_int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single BIGINT column."""
    _bench_fetchall(benchmark, con, f"SELECT i::BIGINT AS a FROM range({N_ROW}) t(i)")


def test_fetchall_smallint(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single INTEGER column."""
    _bench_fetchall(benchmark, con, f"SELECT (i % 100)::INTEGER AS a FROM range({N_ROW}) t(i)")


def test_fetchall_double(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single DOUBLE column."""
    _bench_fetchall(benchmark, con, f"SELECT (i * 1.5)::DOUBLE AS a FROM range({N_ROW}) t(i)")


def test_fetchall_2int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of two BIGINT columns."""
    _bench_fetchall(benchmark, con, f"SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range({N_ROW}) t(i)")


def test_fetchall_str(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single VARCHAR column."""
    _bench_fetchall(benchmark, con, f"SELECT ('str_value_' || i) AS s FROM range({N_STR}) t(i)")


def test_fetchall_mixed(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a mixed scalar/list/struct row."""
    query = (
        "SELECT i::BIGINT AS bi, ('str_' || i) AS s, [i, i + 1, i + 2] AS lst, "
        f"{{'a': i, 'b': i + 1}} AS st FROM range({N_NEST}) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchone_iter(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark iterating a result one row at a time with fetchone."""
    query = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N_STR}) t(i)"

    def run() -> None:
        rel = con.execute(query)
        while rel.fetchone() is not None:
            pass

    benchmark(run)


# --------------------------------------------------------------------------- #
# small-N COMPILE+FETCH FIXED-COST variants: at range(2048) the measured region is dominated by SQL front-end
# compilation + the engine, NOT fetch. MEAS-1 walltime split (vs the range(2048) engine floor in
# test_engine_control_perf.py): ~40% fetch fixed-cost, ~60% compile+engine. They still catch a fixed-cost
# regression, but they are compile+fetch fixed-cost gates, not pure-fetch gates. Plus expensive scalar OUT-row
# types (timestamptz pytz-per-row, blob, null-heavy), a heterogeneous per-cell-dispatch row
# (hugeint+uuid+decimal128+varchar, distinct from the homogeneous columns), and the batched fetchmany loop.
# --------------------------------------------------------------------------- #


def test_fetchall_int_gate(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark the small-N BIGINT compile+fetch fixed-cost (MEAS-1: ~60% compile+engine, ~40% fetch)."""
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a FROM range(2048) t(i)")


def test_fetchall_2int_gate(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark the small-N two-BIGINT compile+fetch fixed-cost."""
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range(2048) t(i)")


def test_fetchall_null_heavy(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a half-NULL BIGINT column."""
    _bench_fetchall(benchmark, con, f"SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END FROM range({N_ROW}) t(i)")


def test_fetchall_timestamptz(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a TIMESTAMPTZ column."""
    _bench_fetchall(
        benchmark, con, f"SELECT (TIMESTAMPTZ '2020-01-01' + (i * INTERVAL 1 SECOND)) FROM range({N_STR}) t(i)"
    )


def test_fetchall_decimal128(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a 128-bit DECIMAL column."""
    _bench_fetchall(benchmark, con, f"SELECT ((i * 1.5)::DECIMAL(28, 6)) FROM range({N_ROW}) t(i)")


def test_fetchall_blob(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a BLOB column."""
    _bench_fetchall(benchmark, con, f"SELECT ('blob_value_' || i)::BLOB FROM range({N_STR}) t(i)")


def test_fetchall_mixed_wide(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a heterogeneous wide-type row."""
    # heterogeneous row -> per-cell type dispatch in the Fetchone column loop (distinct branch/cache profile
    # from the homogeneous single-type columns above)
    query = (
        "SELECT (i::HUGEINT * 1000000000000) AS h, gen_random_uuid() AS u, "
        f"((i * 1.5)::DECIMAL(28, 6)) AS d, ('string_' || i) AS s FROM range({N_STR}) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchmany_batched(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark draining a result with batched fetchmany."""
    query = f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({N_STR}) t(i)"

    def run() -> None:
        rel = con.execute(query)
        while True:
            rows = rel.fetchmany(10_000)
            if not rows:
                break

    benchmark(run)
