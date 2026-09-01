"""Plan construction and rendering: pure client work, no engine. See benchmarks/README.md."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb import col, fn, lit, when

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture

# gate: frames and expressions never touch the engine here (render on a stub
# schema), so the measurement is 100% client binding and fully deterministic.
pytestmark = pytest.mark.gate

# Chunky units of work (~1ms+ per call): sub-ms callables sit inside the
# walltime harness's per-round loop overhead and read as noise locally.
CHAIN_DEPTH = 256
EXPR_REPEATS = 100


def _build_chain() -> duckdb.Frame:
    plan = duckdb.table("t")
    for i in range(CHAIN_DEPTH):
        plan = plan.filter(col("a") > lit(i))
    return plan.select("a", "b")


def _build_expressions() -> duckdb.Frame:
    plan = duckdb.table("t")
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
