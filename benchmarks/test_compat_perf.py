"""What `duckdb.compat` adds: each bench has a twin in test_fetch_perf.py. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _scale import scaled

from duckdb import compat

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pytest_codspeed import BenchmarkFixture

# gate: the wrapper is Python on top of the same fetch path the other benches measure.
pytestmark = pytest.mark.gate

N_ROW = scaled(200_000)
N_STR = scaled(100_000)


@pytest.fixture
def compat_con() -> Iterator[compat.CompatConnection]:
    connection = compat.connect(config={"threads": 1})
    yield connection
    connection.close()


def test_compat_execute_fetchall_int(benchmark: BenchmarkFixture, compat_con: compat.CompatConnection) -> None:
    query = f"SELECT i::BIGINT AS a FROM range({N_ROW}) t(i)"
    compat_con.execute(query).fetchall()  # warm the engine before measuring
    benchmark(lambda: compat_con.execute(query).fetchall())


def test_compat_execute_fetchall_str(benchmark: BenchmarkFixture, compat_con: compat.CompatConnection) -> None:
    query = f"SELECT ('str_value_' || i) AS s FROM range({N_STR}) t(i)"
    compat_con.execute(query).fetchall()  # warm
    benchmark(lambda: compat_con.execute(query).fetchall())


def test_compat_relation_chain_fetchall(benchmark: BenchmarkFixture, compat_con: compat.CompatConnection) -> None:
    def run() -> list[tuple[object, ...]]:
        relation = compat_con.sql(f"SELECT i::BIGINT AS a FROM range({N_ROW}) t(i)")
        return relation.filter("a % 2 = 0").project("a").fetchall()

    run()  # warm
    benchmark(run)
