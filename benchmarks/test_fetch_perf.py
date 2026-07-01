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

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture


@pytest.fixture
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a fresh connection, closed on teardown."""
    c = duckdb.connect()
    yield c
    c.close()


def _bench_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.execute(query).fetchall()  # warm the engine before measuring
    benchmark(lambda: con.execute(query).fetchall())


def test_fetchall_int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single BIGINT column."""
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a FROM range(200000) t(i)")


def test_fetchall_smallint(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single INTEGER column."""
    _bench_fetchall(benchmark, con, "SELECT (i % 100)::INTEGER AS a FROM range(200000) t(i)")


def test_fetchall_double(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single DOUBLE column."""
    _bench_fetchall(benchmark, con, "SELECT (i * 1.5)::DOUBLE AS a FROM range(200000) t(i)")


def test_fetchall_2int(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of two BIGINT columns."""
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range(200000) t(i)")


def test_fetchall_str(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a single VARCHAR column."""
    _bench_fetchall(benchmark, con, "SELECT ('str_value_' || i) AS s FROM range(100000) t(i)")


def test_fetchall_mixed(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a mixed scalar/list/struct row."""
    query = (
        "SELECT i::BIGINT AS bi, ('str_' || i) AS s, [i, i + 1, i + 2] AS lst, "
        "{'a': i, 'b': i + 1} AS st FROM range(50000) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchone_iter(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark iterating a result one row at a time with fetchone."""
    query = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(100000) t(i)"

    def run() -> None:
        rel = con.execute(query)
        while rel.fetchone() is not None:
            pass

    benchmark(run)


# --------------------------------------------------------------------------- #
# ADDED: small-N instruction-count-gate variants (the narrow-numeric fixed-cost path, noise-free at range(2048)
# under simulation mode in CI), expensive scalar OUT-row types (timestamptz pytz-per-row, blob, null-heavy), a
# heterogeneous per-cell-dispatch row (hugeint+uuid+decimal128+varchar, distinct from homogeneous columns), and
# the batched fetchmany loop.
# --------------------------------------------------------------------------- #


def test_fetchall_int_gate(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark the small-N BIGINT instruction-count gate."""
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a FROM range(2048) t(i)")


def test_fetchall_2int_gate(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark the small-N two-BIGINT instruction-count gate."""
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range(2048) t(i)")


def test_fetchall_null_heavy(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a half-NULL BIGINT column."""
    _bench_fetchall(benchmark, con, "SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END FROM range(200000) t(i)")


def test_fetchall_timestamptz(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a TIMESTAMPTZ column."""
    _bench_fetchall(
        benchmark, con, "SELECT (TIMESTAMPTZ '2020-01-01' + (i * INTERVAL 1 SECOND)) FROM range(100000) t(i)"
    )


def test_fetchall_decimal128(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a 128-bit DECIMAL column."""
    _bench_fetchall(benchmark, con, "SELECT ((i * 1.5)::DECIMAL(28, 6)) FROM range(200000) t(i)")


def test_fetchall_blob(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a BLOB column."""
    _bench_fetchall(benchmark, con, "SELECT ('blob_value_' || i)::BLOB FROM range(100000) t(i)")


def test_fetchall_mixed_wide(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark fetchall of a heterogeneous wide-type row."""
    # heterogeneous row -> per-cell type dispatch in the Fetchone column loop (distinct branch/cache profile
    # from the homogeneous single-type columns above)
    query = (
        "SELECT (i::HUGEINT * 1000000000000) AS h, gen_random_uuid() AS u, "
        "((i * 1.5)::DECIMAL(28, 6)) AS d, ('string_' || i) AS s FROM range(100000) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchmany_batched(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Benchmark draining a result with batched fetchmany."""
    query = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(100000) t(i)"

    def run() -> None:
        rel = con.execute(query)
        while True:
            rows = rel.fetchmany(10_000)
            if not rows:
                break

    benchmark(run)


def test_expr_many(benchmark: BenchmarkFixture) -> None:
    """Benchmark building many column/constant expressions."""

    def run() -> int:
        out = []
        for i in range(2000):
            col = duckdb.ColumnExpression(f"col_{i}")
            const = duckdb.ConstantExpression(i)
            out.append(((col + const) * duckdb.ConstantExpression(2)).alias(f"a{i}"))
        return len(out)

    benchmark(run)
