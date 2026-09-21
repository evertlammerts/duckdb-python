# Benchmark suite

CodSpeed micro-benchmarks for the binding hot paths: row fetch, columnar
egress and plan construction. The harness is ported from duckdb-python main's suite; CI is
[codspeed.yml](../.github/workflows/codspeed.yml), the comparison script is
[compare_baseline.py](compare_baseline.py).

## Markers

Every benchmark carries exactly one (registered in `conftest.py`):

- **gate**: binding-dominated, deterministic under Callgrind. A threshold
  breach is a binding regression.
- **informational**: engine-diluted (the `test_engine_control_perf.py`
  floors). Reported, never gated: they would false-positive on engine bumps.

## CI baseline flow

Nightly, the suite runs under `valgrind --tool=callgrind` on Linux and
`compare_baseline.py compare` diffs the instruction counts against the
committed `benchmarks/baseline.json` (report-only). Provenance is the
`engine.pin` SHA: when it differs from the baseline's, gate deltas are
reported but not enforced. To refresh: dispatch the workflow with
`regen=true`, download the `baseline` artifact, and commit it together with
`requirements-bench.txt` (regenerate the pins per the header in that file).

## Local A/B (walltime)

Only walltime runs locally (no Valgrind on macOS arm64). Two caveats:

- The report's "Time (best)" column is unreliable for sub-ms benchmarks;
  read `Run time / Iters` instead, or keep each benchmark's unit of work
  above ~1ms (the plan benches batch their work for exactly this reason).
- Compare builds on the same machine in the same sitting; absolute numbers
  are not portable.

```bash
BENCH_SCALE=10 .venv/bin/python -m pytest benchmarks/ --codspeed \
  --codspeed-mode=walltime -o addopts= -p no:cacheprovider
```

## Conventions

- READ aggregates real columns (`sum`), never `count(*)` (answered from
  metadata).
- Warm once before measuring.
- The `con` fixture pins `threads=1` so engine parallelism does not vary
  with the runner's core count.
- OUT null benches need REAL nulls (`CASE WHEN ... THEN NULL`), or the cheap
  all-valid path is measured instead.
- Ns route through `_scale.py`'s `scaled()`; a floor and the bench it floors
  use the SAME base N. Small fixed-cost probes (`range(2048)`) are not
  scaled.
