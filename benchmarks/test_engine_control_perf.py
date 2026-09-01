"""Pure-engine floor (no Python egress): the binding-fraction reference. See benchmarks/README.md.

`SELECT sum(...) FROM range(N)` aggregates to one scalar, so the fetch is
negligible: these measure SQL compile plus the engine aggregate with ~zero
per-row egress. Comparing a produce/fetch bench against the matching-N floor
here quantifies how much of its cost is binding vs engine. Informational
(they measure the engine), never gated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

import duckdb

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

pytestmark = pytest.mark.informational

# N matched to the benches these floor, and routed through scaled() with the
# SAME base N, so a floor and its bench stay at an identical scaled N and the
# binding fraction stays valid. The 2048 small-N floor is NOT scaled.
Q_1C_SMALL = "SELECT sum(i::BIGINT) FROM range(2048) t(i)"
Q_1C_200K = f"SELECT sum(i::BIGINT) FROM range({scaled(200_000)}) t(i)"
Q_2C_500K = (
    f"SELECT sum(a), sum(b) FROM (SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range({scaled(500_000)}) t(i))"
)


def _bench(benchmark: BenchmarkFixture, con: duckdb.Connection, query: str) -> None:
    duckdb.sql(query).rows(con)  # warm the engine before measuring
    benchmark(lambda: duckdb.sql(query).rows(con))


def test_engine_sum_1col_small(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench(benchmark, con, Q_1C_SMALL)


def test_engine_sum_1col_200k(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench(benchmark, con, Q_1C_200K)


def test_engine_sum_2col_500k(benchmark: BenchmarkFixture, con: duckdb.Connection) -> None:
    _bench(benchmark, con, Q_2C_500K)
