"""Row-count scaling for the benchmarks: CI measures under a slow profiler, so `BENCH_SCALE` shrinks the sizes."""

from __future__ import annotations

import os

FLOOR = 20_000  # below this a bench stops being about the rows, so scaling never goes further


def bench_scale() -> int:
    """Return the divisor from `BENCH_SCALE` (>=1); 1 (no scaling) if unset/invalid."""
    v = os.environ.get("BENCH_SCALE")
    if not v:
        return 1
    try:
        return max(int(v), 1)
    except ValueError:
        return 1


def scaled(n: int) -> int:
    """Return `n`, or a smaller row count when `BENCH_SCALE` is set; only counts shrink, never the data pattern."""
    d = bench_scale()
    if d <= 1:
        return n
    return max(n // d, min(n, FLOOR))
