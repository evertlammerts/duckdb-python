"""Standalone CodSpeed benchmark: the RESULT-CARDINALITY (rows-to-Python) sweep. Run under each build:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_cardinality_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_cardinality_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

REDESIGN NOTE: the first version swept `ORDER BY a DESC LIMIT n` over a fixed source. That was wrong:
the engine's full top-N SORT (~3-14ms, itself variable) dominated and swamped the per-row conversion
signal, and the numbers came out non-monotone. This version pre-materializes the fixed source table ONCE
and sweeps `SELECT * FROM src LIMIT n` with NO ORDER BY: a plain LIMIT early-stops the scan at n rows, so
the engine cost is cheap and monotone in n, and the rows-to-Python CONVERSION is the dominant n-varying
cost. That gives a clean, monotone per-row slope; the A/B delta at each n isolates the binding, and a build
whose slope is steeper has a per-row conversion regression. n=100 is the overhead regime (the natural
instruction-count-gate point); n=100_000 is throughput.

3 columns (BIGINT, DOUBLE, VARCHAR) so per-row conversion is non-trivial. numpy/pandas/pyarrow are pinned to
the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import pytest

SRC_ROWS = 200_000
LIMITS = [100, 1_000, 10_000, 100_000]


@pytest.fixture(scope="module")
def con():
    # Fixed source materialized ONCE (module-scoped): building it per test would add noise, and it must be
    # identical across the n sweep. `SELECT * FROM src LIMIT n` then reads only the first n rows.
    c = duckdb.connect()
    c.execute(
        "CREATE TABLE src AS "
        f"SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b, ('s_' || i) AS s FROM range({SRC_ROWS}) t(i)"
    )
    yield c
    c.close()


def _query(n):
    # No ORDER BY: a plain LIMIT early-stops the scan at n rows -> engine cost cheap and monotone in n, so the
    # per-row binding conversion dominates the n-varying signal (unlike the old ORDER BY top-N sort).
    return f"SELECT a, b, s FROM src LIMIT {n}"


@pytest.mark.parametrize("n", LIMITS)
def test_limit_fetchall(benchmark, con, n):
    q = _query(n)
    con.execute(q).fetchall()  # warm
    benchmark(lambda: con.execute(q).fetchall())


@pytest.mark.parametrize("n", LIMITS)
def test_limit_df(benchmark, con, n):
    q = _query(n)
    con.sql(q).df()  # warm
    benchmark(lambda: con.sql(q).df())


@pytest.mark.parametrize("n", LIMITS)
def test_limit_to_arrow(benchmark, con, n):
    q = _query(n)
    con.sql(q).to_arrow_table()  # warm
    benchmark(lambda: con.sql(q).to_arrow_table())
