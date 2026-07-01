"""CodSpeed benchmark: pure-ENGINE control (no Python egress). Standalone, not in CI's binding gate.

A/B: run under each build, compare (data libs pinned identically, so the delta is the binding):
  cd /Users/evert/projects/duckdb-python/wt-codspeed
  for P in ../main/.venv-release/bin/python .venv-release/bin/python; do \
    $P -m pytest benchmarks/test_engine_control_perf.py \
    --codspeed --codspeed-mode=walltime -o addopts= -p no:cacheprovider; \
  done

These `SELECT sum(...) FROM range(N)` queries aggregate to a single scalar, so the fetchall of the result is
negligible: they measure SQL compile + the engine aggregate with (almost) ZERO per-row Python egress. They are
the "engine floor" reference for MEAS-1: comparing a produce/fetch/ingest benchmark against the matching-N floor
here quantifies how much of that benchmark's cost is the binding vs the engine. They are `informational` (they
measure the engine, not the binding, so they must never gate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

    import duckdb

# informational: pure-engine reference, never gated. `con` fixture + threads=1 live in conftest.py.
pytestmark = pytest.mark.informational

# Matched to the N of the fetch/produce/ingest/udf benchmarks so the floors line up for MEAS-1 subtraction and,
# at baseline regen, for the Option-B binding-fraction of the numeric-produce gates (see compare_baseline.py).
# CRITICAL: these floors go through scaled() with the SAME base N as the benchmarks they floor, so under
# BENCH_SCALE the floor and its benchmark stay at an identical N and the fraction stays valid. The 2048 small-N
# floor is NOT scaled (it is the fixed-cost baseline for the *_gate probes).
Q_1C_SMALL = "SELECT sum(i::BIGINT) FROM range(2048) t(i)"  # small-N gate floor (compile-dominated), NOT scaled
Q_1C_100K = f"SELECT sum(i::BIGINT) FROM range({scaled(100_000)}) t(i)"  # types-matrix numeric-df floor
Q_1C_200K = f"SELECT sum(i::BIGINT) FROM range({scaled(200_000)}) t(i)"  # fetch / native-UDF floor
# produce/ingest floor
Q_2C_500K = (
    f"SELECT sum(a), sum(b) FROM (SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({scaled(500_000)}) t(i))"
)


def _bench(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection, query: str) -> None:
    con.execute(query).fetchall()  # warm
    benchmark(lambda: con.execute(query).fetchall())


def test_engine_sum_1col_small(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Engine floor: compile + sum over range(2048), no egress."""
    _bench(benchmark, con, Q_1C_SMALL)


def test_engine_sum_1col_100k(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Engine floor: compile + sum over range(100k), no egress."""
    _bench(benchmark, con, Q_1C_100K)


def test_engine_sum_1col_200k(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Engine floor: compile + sum over range(200k), no egress."""
    _bench(benchmark, con, Q_1C_200K)


def test_engine_sum_2col_500k(benchmark: BenchmarkFixture, con: duckdb.DuckDBPyConnection) -> None:
    """Engine floor: compile + 2-col sum over range(500k), no egress."""
    _bench(benchmark, con, Q_2C_500K)
