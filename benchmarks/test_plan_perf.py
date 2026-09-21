"""Building a query and turning it into SQL text, with no database involved. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb.frame import col, fn, lit, when

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# gate: nothing here reaches DuckDB, so the measurement is this package alone and repeats exactly.
pytestmark = pytest.mark.gate

# Each call must take about a millisecond, or the timing harness's own overhead swamps it.
CHAIN_DEPTH = 256
EXPR_REPEATS = 100


def _build_chain() -> duckdb.frame.Frame:
    plan = duckdb.frame.table("t")
    for i in range(CHAIN_DEPTH):
        plan = plan.filter(col("a") > lit(i))
    return plan.select("a", "b")


def _build_expressions() -> duckdb.frame.Frame:
    plan = duckdb.frame.table("t")
    return plan.with_columns(
        total=col("a") + col("b") * lit(2),
        clipped=when(col("a") > lit(100)).then(lit(100)).otherwise(col("a")),
        label=fn("concat", lit("row_"), col("a")),
        frac=col("b") / (col("a") + lit(1)),
    ).select("total", "clipped", "label", "frac")


def test_plan_build_chain(benchmark: BenchmarkFixture) -> None:
    _build_chain()  # warm imports and caches before measuring
    benchmark(_build_chain)


def test_plan_render_chain(benchmark: BenchmarkFixture) -> None:
    plan = _build_chain()
    plan.render()  # warm
    benchmark(plan.render)


def test_plan_build_render_expressions(benchmark: BenchmarkFixture) -> None:
    def run() -> int:
        total = 0
        for _ in range(EXPR_REPEATS):
            total += len(_build_expressions().render())
        return total

    run()  # warm
    benchmark(run)
