"""Env-gated row-count scaling for the benchmark suite (INFRA-4).

Callgrind is 20-50x, and the O(rows) / per-row-object benchmarks at full N make the CI sweep too slow. `scaled(n)`
shrinks those row counts ONLY when an explicit `BENCH_SCALE=<divisor>` env var is set (which the CI Callgrind
sweep sets). Unset -> full N, so LOCAL walltime A/B keeps the large N unchanged.

CRITICAL: a gate benchmark and the engine-control floor it is compared against (the FLOOR_MAP pairs in
compare_baseline.py) share the same base N, so routing BOTH through `scaled()` keeps them at an identical scaled
N -- the Option-B binding_fraction stays valid. Scaling ONLY reduces row counts; it must never change the data
patterns the benchmarks depend on (real NULLs, mixed ASCII+non-ASCII+null, LIMIT-no-ORDER-BY, warm-before-measure).

A floor keeps a scaled benchmark row-dominated (well above the range(2048) fixed-cost probes), so per-element
work still dominates and the fraction/signal stay meaningful. The small-N `*_gate` probes are NOT routed through
this (they are already fast and are the fixed-cost baseline).
"""

from __future__ import annotations

import os

FLOOR = 20_000  # a scaled bench never drops below this (stays row-dominated, ~10x the range(2048) probes)


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
    """Return `n` at full scale, or `max(n // BENCH_SCALE, min(n, FLOOR))` when scaling is enabled."""
    d = bench_scale()
    if d <= 1:
        return n
    return max(n // d, min(n, FLOOR))
