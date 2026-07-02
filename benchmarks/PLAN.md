# Benchmark suite plan

Design rationale for the binding micro-benchmarks. The suite is implemented in `benchmarks/`; CI lives in
`../.github/workflows/codspeed.yml`; conventions, markers, and the two data-pattern traps are in
[README.md](README.md).

Priority: **P0** = known-regression or cutover-reworked path (narrow-numeric common case); **P1** = high-traffic
conversion or per-element Python work; **P2** = correctness-relevant, lower-traffic or engine-dominated.

## Scenarios

PRODUCE (duckdb to Python) is the highest regression risk: `Fetchone` builds a `TupleBuilder` per row and calls
`FromValue` per cell (O(rows x cols), the shape of the historical ~15% fetchall regression).

- **OUT-row** (`test_fetch_perf`, `test_types_roundtrip_perf`): fetchall / fetchone / fetchmany per type. P0
  narrow numeric; P1 varchar, list, struct, and the expensive per-row types (decimal `Decimal()`, timestamptz
  pytz, hugeint string round-trip, uuid). Small-N `*_gate` probes isolate the compile+fetch fixed cost.
- **OUT-col** (`test_produce_numpy_perf`): df() / fetchnumpy() reworked columnar path. P0 numeric no-null vs
  REAL-null (the masked_array branch); plus string, timestamp, and wide-internal (hugeint/uuid/decimal128).
- **OUT-arrow / polars** (`test_arrow_perf`): to_arrow_table / reader / pl(). Informational (engine-parallel,
  GIL-released).
- **Cardinality** (`test_cardinality_perf`): a LIMIT-n sweep giving a clean per-row conversion slope.

INGEST (Python to duckdb):

- **numpy / pandas** (`test_ingest_numpy_perf`, `test_pandas_perf`): numpy-backed scan (NaN-to-NULL, masked),
  object-string transcode ladder, arrow-backed zero-copy, and the per-bind PandasAnalyzer.
- **arrow** (`test_arrow_perf`): Table + RecordBatchReader + dictionary sweep.
- **native** (`test_ingest_native_perf`): values() list/tuple/dict per-cell TransformPythonValue, executemany.

UDF (`test_udf_perf`, zero coverage before this suite): native scalar per-row (P0, the biggest untested per-call
path) and vectorized arrow per-chunk.

## Type x direction matrix

Directions: IN-native (TransformPythonValue), IN-numpy (NumpyScan), OUT-row (FromValue), OUT-col (ArrayWrapper),
OUT-arrow.

| Type | IN-native | IN-numpy | OUT-row | OUT-col | OUT-arrow |
|------|-----------|----------|---------|---------|-----------|
| int32/int64 | P1 | **P0** | **P0** | **P0** | P1 |
| double | P1 | **P0** (NaN->NULL) | P0 | P0 | P1 |
| varchar | P1 | **P0** (PyUnicode) | P1 | P1 | P1 |
| bool | P2 | P1 | P2 | P1 | P2 |
| decimal64/128 | P2 | n/a | **P1** (Python Decimal) | P1 | P2 |
| date | P2 | P1 | P1 | P1 | P2 |
| timestamp(tz) | P1 | P1 | **P1** (pytz/row) | P1 | P1 |
| LIST/STRUCT | P2 | P2 | P1 (recursive) | P1 | P2 |
| hugeint/uuid | P2 | P2 | **P1** (round-trip) | P1 | P2 |
| blob/map | P2 | P2 | P2 | P2 | P2 |
| NULL-heavy | n/a | **P1** | P2 | **P0** (masked_array) | P1 |

## Mechanics

- **Walltime vs instruction-count.** Local A/B is walltime only (no Valgrind on macOS arm64). CI is
  instruction-count via self-hosted Callgrind (near-deterministic, PYTHONHASHSEED pinned), diffed against a
  committed baseline. Report-only until trusted.
- **Marker split + auto-move.** Every benchmark is `gate` or `informational` (see README). At baseline regen,
  each numeric-produce gate's binding fraction `= 1 - floor_Ir / bench_Ir` is computed against its engine floor
  (`test_engine_control_perf`); a gate below the ~25% cutoff is auto-moved to informational (a threshold on an
  engine-diluted total is not meaningful). OUT-row fetch and UDFs are ~all binding; numeric produce is a bulk
  memcpy of ~engine magnitude (auto-move candidate).
- **Guards.** compare_baseline.py warns and stops enforcing when BENCH_SCALE, the pin file, or the DuckDB
  submodule SHA differ from the baseline's (any of those makes the counts non-comparable).
- **Sustained-leak guard** (`tests/fast/test_binding_pressure_leak.py`): a plain RSS + object-count test for the
  object-pinning paths, since a per-call refcount imbalance is invisible to a steady-state benchmark.
- **Memory mode** (a second Callgrind sweep for O(rows) produce peak-RSS) is designed but deferred; the
  `test_mem_df_with_nulls` tracemalloc guard is the local stand-in.

## Cross-check vs iqmo-org/bareduckdb

Their suite is a SQL-file-driven A/B comparing two clients (production `duckdb` vs the C-API prototype), arrow-in
/ arrow-out only, no fetchall/df/numpy/native/UDF coverage. So our binding suite is far broader; their genuine
deltas concentrate in PRODUCE/types. Actionable additions they suggest:

- **hugeint / uuid in the produce matrix** (they select both): OUT-row does a per-value string round-trip, distinct
  from narrow int. Now in `test_produce_numpy_perf` / `test_fetch_perf`.
- **int128-internal decimal** (`DECIMAL(28,x)`) alongside the int64-internal one: hits a wider cast path. Added.
- **heterogeneous mixed-type row**: exercises per-cell type dispatch in the Fetchone loop, unlike homogeneous
  columns. Added as `test_fetchall_mixed_wide`.
- **long varchar (>64 char)** alongside the short string: shifts string copy / transcode toward copy-bound. Added
  as `varchar_long` in the matrix.
- **result-cardinality (top-N) sweep**: holds engine work ~constant while sweeping rows-to-Python. Adopted as
  `test_cardinality_perf` (plain LIMIT, no ORDER BY; the sort swamped the signal).
- **peak-memory guard** on the O(rows) produce paths: a conversion regression is often memory-shaped. Partially
  covered by the tracemalloc guard; full coverage waits on memory mode.

Out of scope (theirs, not adopted): pure-engine filter/group/window workloads; 100M+ row scale (IO/engine
dominated); the free-threading category (unsupported by this client). Do NOT adopt their no-warmup single-run
methodology (charges import-cache population into the measurement).
