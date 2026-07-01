"""Standalone CodSpeed benchmark module: the RESULT-CARDINALITY (top-N) sweep — NOT integrated (not in
pyproject, not in CI, not committed). Run under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-codspeed/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  $M -m pytest benchmarks/test_cardinality_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider
  $C -m pytest benchmarks/test_cardinality_perf.py --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider

WHY THIS MODULE (adopted from iqmo-org/bareduckdb): hold the SOURCE fixed and sweep only the number of rows
materialized to Python via ORDER BY ... LIMIT n for n in {100, 1k, 10k, 100k}, through fetchall / df /
to_arrow_table. The engine cost (scan the fixed SRC + top-N heap) stays ~constant, so the walltime delta
across n is dominated by the per-row binding conversion -> a clean per-row slope. The n=100 end is the
noise-free overhead regime (the natural instruction-count-gate point); the n=100k end is throughput.

A clean monotone slope (and ~parity slope between the two builds) is the signal we report; a build whose slope
is steeper has a per-row conversion regression. Source held constant rules out scan-cost as the confound (a
cleaner axis than varying range(), which also changes scan cost).

numpy/pandas/pyarrow are pinned to the SAME versions in both .venv-release, so the A/B delta is purely the binding.
"""

import duckdb
import pytest

SRC = 200_000  # fixed source size -> constant engine scan + top-N across all n
LIMITS = [100, 1_000, 10_000, 100_000]

# 3 columns (BIGINT, DOUBLE, VARCHAR) so the per-row conversion is non-trivial; source is a fixed inline
# subquery (no table state) and ORDER BY forces a full scan + top-N of the same SRC rows every time.
_SRC_SUBQ = f"(SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b, ('s_' || i) AS s FROM range({SRC}) t(i))"


def _query(n):
    return f"SELECT a, b, s FROM {_SRC_SUBQ} ORDER BY a DESC LIMIT {n}"


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


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
