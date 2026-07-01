"""CodSpeed benchmark: the result-cardinality (rows-to-Python) sweep. Standalone, not in CI.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_cardinality_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

Sweeps `SELECT * FROM src LIMIT n` (no ORDER BY) over a pre-materialized 3-column source: a plain LIMIT
early-stops the scan, so the per-row conversion dominates and the slope is monotone in n. A steeper slope on
one build is a per-row conversion regression. n=100 is the overhead regime, n=100_000 is throughput.
(An earlier ORDER BY version was dropped: the top-N sort swamped the signal.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

SRC_ROWS = 200_000
LIMITS = [100, 1_000, 10_000, 100_000]


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a connection over a once-materialized source table."""
    # Fixed source materialized ONCE (module-scoped): building it per test would add noise, and it must be
    # identical across the n sweep. `SELECT * FROM src LIMIT n` then reads only the first n rows.
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE src AS "
        f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b, ('s_' || i) AS s FROM range({SRC_ROWS}) t(i)"
    )
    yield c
    c.close()


def _query(n: int) -> str:
    # No ORDER BY: a plain LIMIT early-stops the scan at n rows -> engine cost cheap and monotone in n, so the
    # per-row binding conversion dominates the n-varying signal (unlike the old ORDER BY top-N sort).
    return f"SELECT a, b, s FROM src LIMIT {n}"


@pytest.mark.parametrize("n", LIMITS)
def test_limit_fetchall(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, n: int) -> None:
    """Benchmark fetchall over a LIMIT n sweep."""
    q = _query(n)
    con.execute(q).fetchall()  # warm
    benchmark(lambda: con.execute(q).fetchall())


@pytest.mark.parametrize("n", LIMITS)
def test_limit_df(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, n: int) -> None:
    """Benchmark df() over a LIMIT n sweep."""
    q = _query(n)
    con.sql(q).df()  # warm
    benchmark(lambda: con.sql(q).df())


@pytest.mark.parametrize("n", LIMITS)
def test_limit_to_arrow(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, n: int) -> None:
    """Benchmark to_arrow_table() over a LIMIT n sweep."""
    q = _query(n)
    con.sql(q).to_arrow_table()  # warm
    benchmark(lambda: con.sql(q).to_arrow_table())
