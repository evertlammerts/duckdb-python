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

Nightly, two jobs each hold one side fixed (`.github/workflows/codspeed.yml`):

- **Client delta**: the current code runs under `valgrind --tool=callgrind`
  on the baseline's own engine, and `compare_baseline.py compare --enforce`
  diffs the counts against the committed `benchmarks/baseline.json`. Every
  delta is the client's, so a gate regression fails the job. Skipped while no
  baseline is committed.
- **Engine delta**: when the newest published engine differs from the
  baseline's, the same code runs on both engines in one job; the difference
  is the engine's, reported whole and never gated. The new-engine run is
  uploaded as the `baseline` artifact: the candidate baseline. To adopt it,
  download it and commit it as `benchmarks/baseline.json`, together with
  `requirements-bench.txt` (regenerate the pins per the header in that file).
  When the old engine can no longer be fetched or built against the pinned
  C++ API, the job says so and still produces the candidate. Dispatching the
  workflow with `regen` runs this job on an unchanged engine, which is how
  the baseline is refreshed after an accepted client change, a gate rename,
  or a pins bump.

With no baseline committed, the first run takes the bootstrap path: the
client job is skipped and the engine job measures the new engine alone and
uploads the first candidate. A baseline records the engine version and commit
it was measured on, read from the running build itself; `compare` reads its
own the same way, and an enforced run fails rather than passes when the
engines differ or a gate benchmark did not run.

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
