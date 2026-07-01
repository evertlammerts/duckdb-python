# CodSpeed Benchmark Suite Plan — duckdb-python binding hot paths

Grounded in the binding source on `perf/codspeed` (`src/`). File:line citations are to this tree.

## 0. Conventions (from the existing 3 modules, keep these)

- Function-scoped `con` fixture; module-scoped input-data fixtures.
- READ = `SELECT sum(col) / sum(length(col))` (never `count(*)`, which is answered from metadata).
- WRITE = eager materialize or fully drain the lazy reader.
- Warm the engine once (`con.execute(query).fetchall()`) before `benchmark(...)` so first-call import-cache population is not charged to the measured region.
- Pin numpy/pandas/pyarrow/polars identically across A/B so deltas are pure binding cost.

Ranking: **P0** = on a known regression path or the cutover-reworked code (narrow-numeric common case); **P1** = high-traffic conversion / per-element Python work; **P2** = correctness-relevant, lower traffic or engine-dominated.

## (a) Prioritized scenarios

### PRODUCE (duckdb -> external) — highest regression risk

Row path: `DuckDBPyResult::Fetchone` (`src/pyresult.cpp:126-151`) builds a `PyUtil::TupleBuilder` (`src/include/duckdb_python/pyutil.hpp:101-125`) per row and calls `PythonObject::FromValue` (`src/native/python_objects.cpp:474`) per cell. O(rows x cols). This is the shape of the historical ~15% fetchall regression.

| # | Scenario | SQL / setup | Measures | Pri |
|---|----------|-------------|----------|-----|
| P0-1 | fetchall int64 1col | `SELECT i::BIGINT a FROM range(1_000_000)` | TupleBuilder + FromValue int (`python_objects.cpp:489`) | P0 |
| P0-2 | fetchall int 2-4col | `SELECT i::BIGINT,(i+1)::BIGINT,(i*2)::INTEGER FROM range(1_000_000)` | TupleBuilder scaling w/ col count | P0 |
| P0-3 | fetchall double | `SELECT (i*1.5)::DOUBLE FROM range(1_000_000)` | FromValue double | P0 |
| P0-4 | fetchall varchar | `SELECT ('str_value_'||i) FROM range(500_000)` | FromValue VARCHAR string copy (`python_objects.cpp:515`) | P1 |
| P0-5 | fetchone loop (overhead) | `SELECT i::BIGINT,(i*1.5)::DOUBLE FROM range(100_000)` | per-call Fetchone + chunk-boundary FetchNext + GIL cycle | P0 |
| P0-6 | fetchmany batched | as P0-5, `fetchmany(10_000)` loop | Fetchmany loop | P1 |
| P1-7 | **df() numeric (reworked)** | `SELECT i::BIGINT,(i*1.5)::DOUBLE FROM range(1_000_000)` | FetchNumpyInternal -> ArrayWrapper ConvertColumnRegular, `HAS_NULLS=false/PANDAS=true` branch (`array_wrapper.cpp:415-425`) | P0 |
| P1-8 | **df() numeric WITH NULLS** | `SELECT CASE WHEN i%10=0 THEN NULL ELSE i::BIGINT END FROM range(1_000_000)` | `HAS_NULLS=true` + masked_array build (`array_wrapper.cpp:743-757`) + masked->pd.NA rewrite (`pyresult.cpp:362-393`) | P0 |
| P1-9 | fetchnumpy numeric | as P1-7 | FetchNumpyInternal without the DataFrame wrap | P1 |
| P1-10 | df() varchar | `SELECT ('str_value_'||i) FROM range(500_000)` | StringConvert PyUnicode_FromStringAndSize per row (`array_wrapper.cpp:164-181`) | P1 |
| P1-11 | df() timestamp | `SELECT TIMESTAMP '2020-01-01'+(i*INTERVAL 1 SECOND) FROM range(1_000_000)` | TimestampConvertNano + ConvertDateTimeTypes (`pyresult.cpp:299`) | P1 |
| P1-13 | to_record_batch_reader drained | `range(1_000_000)`, `to_record_batch_reader(100_000)` | lazy stream (`pyresult.cpp:573`), iterate + sum num_rows | P1 |
| P2-15 | torch()/tf() numeric | `range(500_000)` | FetchNumpyInternal + per-col from_numpy (`pyresult.cpp:405-421`) | P2 |
| P2-16 | fetch_df_chunk | large query, loop `fetch_df_chunk()` | FetchDFChunk per chunk (`pyresult.cpp:400`) | P2 |
| P1-17 | fetchall LIST<int> | `SELECT [i,i+1,i+2] FROM range(200_000)` | FromValue LIST recursion (`python_objects.cpp:651`) | P1 |
| P1-18 | fetchall STRUCT | `SELECT {'a':i,'b':i+1} FROM range(200_000)` | FromStruct dict build (`python_objects.cpp:390-414`) | P1 |
| P1-20 | fetchall DECIMAL | `SELECT (i::DECIMAL(18,3))/1000 FROM range(200_000)` | Python `Decimal()(val.ToString())` per row (`python_objects.cpp:507`) | P1 |
| P1-21 | fetchall TIMESTAMPTZ | `SELECT (TIMESTAMPTZ '2020-01-01'+(i*INTERVAL 1 SECOND)) FROM range(100_000)` | pytz localize+astimezone per row (`python_objects.cpp:567-573`) | P1 |
| P2-22 | fetchall NULL-heavy | `SELECT CASE WHEN i%2=0 THEN NULL ELSE i::BIGINT END FROM range(1_000_000)` | validity branch + nb::none (`pyresult.cpp:142`) | P2 |
| P2-23 | fetchall BLOB | `SELECT ('blob_'||i)::BLOB FROM range(200_000)` | nb::bytes (`python_objects.cpp:517`) | P2 |

### INGEST (external -> duckdb)

| # | Scenario | Setup | Path | Pri |
|---|----------|-------|------|-----|
| I0-1 | **pandas numpy int64/double** | DataFrame 1M | NumpyScan::Scan ScanNumpyMasked zero-copy when stride==sizeof(T); double NaN->NULL loop (`numpy_scan.cpp:76-112,236-246`) reworked | P0 |
| I0-2 | **pandas numpy object-string** | `pd.array(strings,dtype=object)` 500k | NumpyScan STRING/OBJECT: per-row isinstance, PyUnicodeIsCompactASCII zero-copy vs DecodePythonUnicode transcode (`numpy_scan.cpp:353-452`) reworked | P0 |
| I1-3 | pandas object bind-time analyzer | object col 100k+ | Pandas::Bind -> PandasAnalyzer::Analyze samples rows GetItemType ladder (`analyzer.cpp:356-460`). Per-BIND overhead, independent of rows (count(*) ok here) | P1 |
| I1-4 | pandas arrow-backed | pd.ArrowDtype 1M | ToArrowTable -> arrow scan (`pyconnection.cpp:1799`) | P1 |
| I0-5 | arrow Table | 1M | CreateArrowScan PythonTableArrowArrayStreamFactory near-zero-copy (`python_replacement_scan.cpp:55-83`) | P1 |
| I1-6 | arrow RecordBatchReader | from_batches | same factory, streaming (distinct from Table) | P1 |
| I1-7 | polars DataFrame | 1M | entry.to_arrow() one-time + arrow scan (`replacement_scan.cpp:150-156`) | P2 |
| I1-8 | numpy ndarray + dict-of-arrays | np.arange | replacement scan -> pandas_scan (`replacement_scan.cpp:163-200`) | P2 |
| I1-9 | **native values() list-of-tuples** | `con.values([(i,i*1.5,'s') for i in range(100_000)])` | Values -> TransformPythonValue per cell, GetPythonObjectType ladder (`python_conversion.cpp:402-454,1075`) | P1 |
| I1-10 | native list-of-dicts | list of dicts | TransformDictionaryToStruct recursion (`python_conversion.cpp:119`) | P2 |
| I1-11 | executemany params | INSERT ?,?  100k sets | ExecuteMany loop, TransformPythonValue per set (`pyconnection.cpp:500-544`) | P2 |
| I2-12 | read_parquet/csv/json | a file | arg marshal -> TableFunction under GIL-release; engine-dominated | P2 |

### UDF (`src/python_udf.cpp`) — zero coverage today

| # | Scenario | Setup | Path | Pri |
|---|----------|-------|------|-----|
| U0-1 | **scalar native 1 int arg** | `def f(x):return x+1`, `SELECT sum(f(i::BIGINT)) FROM range(1_000_000)` | per-row TupleBuilder args + PyObject_CallObject + TransformPythonObject result (`python_udf.cpp:320-384`) | P0 |
| U0-2 | scalar native 2-3 args | `def f(a,b):return a+b` 2 cols 1M | arg-tuple scaling | P1 |
| U1-3 | scalar native string | `def f(s):return s.upper()` 500k | VARCHAR in + string out | P1 |
| U1-4 | scalar native NULL inputs | 50% NULL, DEFAULT handling | SetNull short-circuit (`python_udf.cpp:340-350`) | P1 |
| U1-6 | **vectorized arrow UDF** | `type='arrow'` pc.add 1M | ConvertDataChunkToPyArrowTable + call + ConvertArrowTableToVector cast (`python_udf.cpp:33-144,225`) | P0 |
| U2-7 | vectorized NULL slicing | DEFAULT + nulls | selvec compaction/reconstruction (`python_udf.cpp:197-305`) | P2 |

## (b) Type x direction matrix

Directions: IN-native (TransformPythonValue), IN-numpy (NumpyScan), OUT-row (FromValue), OUT-col (ArrayWrapper), OUT-arrow.

| Type | IN-native | IN-numpy | OUT-row | OUT-col | OUT-arrow |
|------|-----------|----------|---------|---------|-----------|
| int32/int64 | P1 | **P0** | **P0** | **P0** | P1 |
| double | P1 | **P0** (NaN->NULL) | P0 | P0 | P1 |
| varchar | P1 | **P0** (PyUnicode) | P1 | P1 | P1 |
| bool | P2 | P1 | P2 | P1 | P2 |
| decimal | P2 | n/a | **P1** (Python Decimal) | P1 | P2 |
| date | P2 | P1 | P1 | P1 | P2 |
| timestamp | P1 | **P1** | P1 | P1 | P1 |
| timestamptz | P2 | P1 | **P1** (pytz/row) | P1 | P2 |
| time/interval | P2 | P1 | P1 | P1 | P2 |
| LIST/ARRAY | P2 | P2 | P1 (recursive) | P1 | P2 |
| STRUCT | P2 | P2 | P1 (recursive) | P1 | P2 |
| MAP | P2 | P2 | P2 | P2 | P2 |
| blob | P2 | P2 | P2 | P2 | P2 |
| NULL-heavy | - | **P1** | P2 | **P0** (masked_array) | P1 |
| enum/category | - | P1 | P1 | P1 | P2 |

Minimum viable to ship: int64, double, varchar, timestamp, decimal, LIST, STRUCT, NULL-heavy in OUT-row and OUT-col; int64/double/varchar in IN-numpy.

## (c) Gaps vs the existing 3 modules

Covered well: OUT-row narrow numeric, OUT-arrow/polars numeric+string, pandas IN/OUT numpy-vs-arrow numeric+string, fetchone-loop numeric.

Missing:
1. **PRODUCE columnar reworked path under-covered** — df() only 500k, only numeric/string, never with NULLS (the masked-array branch is exactly what changed). Add df-with-nulls, fetchnumpy, df-timestamp.
2. **UDFs: zero coverage** — whole subsystem (python_udf.cpp), native per-row is the single biggest untested per-call-overhead path. Add U0-1/U0-2/U1-3/4/U1-6.
3. **Native Python ingest: zero coverage** — values()/list-of-tuples/list-of-dicts/executemany via TransformPythonValue. Add I1-9/10/11.
4. **Expensive scalar OUT-row types untested** — decimal, timestamptz, interval, isolated LIST/STRUCT/MAP. Add P1-17..21.
5. **Object-column bind-time analyzer untested** — PandasAnalyzer sampling, per-bind cost. Add I1-3.
6. **Size regimes thin** — add 1M throughput AND 1-row overhead variants.
7. **Arrow ingest only pa.table** — add RecordBatchReader, polars, numpy-ndarray ingest.
8. **NULL-heavy IN-numpy untested** (ScanNumpyMasked + ApplyMask).

## (d) Suite organization + CodSpeed mechanics

```
benchmarks/
  test_fetch_perf.py            # EXISTING — OUT-row. Add: nested, decimal, timestamptz, null-heavy, 1M+1-row
  test_arrow_perf.py            # EXISTING — add RecordBatchReader ingest, materialized vs stream
  test_pandas_perf.py           # EXISTING — add df()-with-nulls, datetime, fetchnumpy, analyzer bind
  test_produce_numpy_perf.py    # NEW — df()/fetchnumpy/fetch_df_chunk reworked columnar, per-type, null vs no-null
  test_ingest_native_perf.py    # NEW — values()/list-of-tuples/list-of-dicts/executemany
  test_ingest_numpy_perf.py     # NEW — numpy ndarray / object-string scan / analyzer bind
  test_udf_perf.py              # NEW — scalar native + vectorized arrow UDFs
  test_types_roundtrip_perf.py  # NEW — type x direction matrix sweep, parametrized
```
One module per binding subsystem so a CodSpeed report points at one src/ area. torch/tf go in produce_numpy (wrap FetchNumpyInternal); polars stays in arrow (wraps FetchArrowTable).

### Walltime vs instruction-count

- **Local A/B (macOS arm64): walltime only** (no Valgrind), `--codspeed-mode=walltime`.
- **CI gate: instruction-count / simulation (Linux + Callgrind)**, deterministic — gate PRs with this.

Instruction-count is ideal AND should gate the GIL-held single-threaded overhead paths: fetchone loop, fetchall/fetchmany, native UDF per-call, native values() ingest, analyzer bind, all per-element converters (FromValue, TransformPythonValue, NumpyScan object/string, ArrayWrapper fill). The historical fetchall regression would be caught cleanly here.

Noisy under instruction-count — keep walltime-only, informational, do NOT hard-gate:
- to_arrow_table / pl() on materialized results: PromoteMaterializedToArrow re-runs the query parallel with GIL released (`pyresult.cpp:450-477`).
- Large 1M+ SELECT sum() ingest reads: engine parallel aggregate dominates.
- read_csv/parquet/json: engine + I/O dominated.
- GIL-per-chunk streaming (FetchNextRaw, to_record_batch_reader drain).

Gate tactic: pair each large-throughput scenario with a small/1-row variant (e.g. fetchall range(1_000_000) walltime + fetchall range(2048) instruction-count gate) so binding fixed-cost is measured noise-free.

### Two code-grounded gotchas
- **OUT-col null benchmarks need REAL DuckDB nulls** (`CASE WHEN ... THEN NULL`): the masked-array branch only triggers on an actually-invalid validity bit (`array_wrapper.cpp:396-404,736`); a no-null column silently takes the cheap `std::move` path and measures the wrong thing.
- **IN-numpy string benchmarks need mixed ASCII + non-ASCII + a NaN/pd.NA/None sentinel**: the scan zero-copies compact-ASCII (`numpy_scan.cpp:416-418`) but transcodes otherwise (`numpy_scan.cpp:429-446`); ASCII-only misses the transcode + null-detection ladder.

## (e) Cross-check vs iqmo-org/bareduckdb

Source read live from `iqmo-org/bareduckdb` `main`, subdir `benchmark/` (GitHub API + raw files).

### What their suite covers / how it is organized

A **SQL-file-driven A/B harness comparing two clients** — production `duckdb` vs `bareduckdb` (the C-API / free-threading prototype) — not a binding micro-bench.

- `benchmark.py` orchestrates: discovers `cases/**/*.sql`, picks the matching `data/DATA*` dir, and runs each `(sql x parquet-file x db_mode)` as a fresh `uv run run_case.py` **subprocess**. `DBMODES=[duckdb, bareduckdb_capsule, bareduckdb_arrow]`; active `READ_MODES=[arrow_table]` (parquet/arrow_reader present but off).
- `run_case.py` per case: fresh `connect()`, `pyarrow.parquet.read_table(file)` + `conn.register(name, table)`, then `conn.sql(query).to_arrow_table()`, timed with `time.perf_counter()` and peak RSS via `resource.getrusage`. **No warmup, single run, result discarded.** Universal ingest = register(arrow table); universal produce = `to_arrow_table()`.
- `data/`: `DATA_RANGE` = single BIGINT `range(N)` at 5M / 100M; `DATA_CATEGORY_DATE_PRICE` = (VARCHAR category, DATE, BIGINT price) cross-join at 36M / 3.6B.
- `cases/`: `types/` (decimal `DECIMAL(28,12)`, hugeint `HUGEINT`, mixed_types `HUGEINT+uuid()+DECIMAL(28,6)+VARCHAR` in one row, timestamp `TIMESTAMP+INTERVAL`, varchar_long ~100-char), `limit/` (LIMIT 100 / 1k / 10k / 100k top-N — a result-cardinality sweep), `filter/`, `groups/`, `window/`, `threading/` (parallel group/window/self-join/registered-arrow-scan), plus a separate `stats/` harness.

Their INGEST is arrow-only and their PRODUCE is arrow-only; they have **no** fetchall/fetchone, df()/numpy, pandas/numpy/native/polars ingest, or UDF coverage — so our binding suite is far broader on binding-specific surfaces. Their genuine deltas are concentrated in the PRODUCE/types dimension and in engine/threading workloads.

### DELTA — actionable additions/changes

- **[BINDING] Add HUGEINT to the produce matrix (currently absent).** `types/hugeint.sql`, `mixed_types.sql`. OUT-row `FromValue` HUGEINT does `PyLong_FromString(val.GetValue<string>())` — a per-value string round-trip (`python_objects.cpp:500`), unlike narrow int; OUT-col casts hugeint->double (`array_wrapper.cpp:662`); OUT-arrow is a distinct decimal128/int128 export. Scenario: `SELECT i::HUGEINT FROM range(1_000_000)` through fetchall / df / to_arrow_table. Add a `hugeint` row to the type x direction matrix.
- **[BINDING] Add UUID to the produce matrix (absent).** `mixed_types.sql` selects `uuid()`. OUT-row builds a Python `uuid.UUID` per row (`python_objects.cpp:708-711`); OUT-col uses `UUIDConvert` (`array_wrapper.cpp:230-244`). Scenario: `SELECT gen_random_uuid() FROM range(200_000)` through fetchall / df / to_arrow_table. Add a `uuid` row to the matrix.
- **[BINDING] Add a 128-bit-internal DECIMAL variant.** Our P1-20 uses `DECIMAL(18,3)` (int64 internal); bareduckdb uses `DECIMAL(28,12)` / `(28,6)` (int128 internal), hitting `ConvertDecimalInternal<hugeint_t>` (`array_wrapper.cpp:571`) and the wider `PyDecimalCastSwitch`/`Decimal()` round-trip. Run both an int64-internal and an int128-internal decimal.
- **[BINDING] Add a heterogeneous mixed-type row (new scenario).** `SELECT i::HUGEINT, gen_random_uuid(), (i*1.5)::DECIMAL(28,6), ('string_'||i) FROM range(200_000)` through fetchall and df. Exercises per-cell type dispatch in the `Fetchone` column loop (`pyresult.cpp:140-148`) — a different branch/cache profile than our homogeneous columns (P0-1..3 are single-type).
- **[BINDING] Add a long-varchar (>64 char) variant** alongside the short `'str_value_'||i`. `'...'||repeat('data ',10)||i::VARCHAR` (~100 chars). Short strings are copy-cheap/overhead-bound; long strings shift OUT-row/OUT-col string copy and the IN-numpy `DecodePythonUnicode` transcode (`numpy_scan.cpp:429-446`) toward copy-bound. Apply to OUT-row, OUT-col, IN-numpy varchar scenarios.
- **[BINDING] Adopt their result-cardinality (top-N) sweep as a produce axis.** `SELECT * FROM <fixed source> ORDER BY k DESC LIMIT n` for n in {100, 1k, 10k, 100k}, fetched via fetchall / df / to_arrow_table with the source held constant. Holds engine work ~constant while sweeping rows-materialized-to-Python → a clean per-row conversion slope, and the small-n end is an ideal noise-free instruction-count gate (overhead regime). Cleaner than varying `range()` (which also changes scan cost).
- **[BINDING] Broaden the OUT-arrow column of the matrix.** Their entire produce path is `to_arrow_table`, and they push hugeint / decimal128 / uuid / timestamp / long-varchar / mixed-row through it — exactly the arrow-export converters (ArrowConverter/appender for int128/uuid/decimal128) our OUT-arrow column currently leaves at P1/P2 numeric+string. Add these types to OUT-arrow.
- **[BINDING, hard to gate] registered-arrow-scan under parallelism.** `threading/registered_arrow_scan.sql` pulls batches from `PythonTableArrowArrayStreamFactory::Produce` (binding code in `arrow/arrow_array_stream.cpp`) across engine threads holding/releasing the GIL — a real binding-contention risk. Keep as walltime-informational only; too noisy for an instruction-count gate.
- **[ENGINE] `filter` / `groups` / `window` / `self_join` pure-engine workloads** — out of scope for a binding gate; the binding only wraps them with register + to_arrow_table, and their consume (a small aggregate) is trivial so the measurement is ~pure engine. Note, do not add to the binding suite.
- **[ENGINE] 100M / 3.6B-row scale** — too slow / IO+engine-dominated / walltime-noisy for a codspeed gate; keep our regimes <= ~1M.
- **[ENGINE] threading / free-threading category** — the production client does not support free-threading (CLAUDE.md); deprioritize for this suite.

### Methodology notes for our codspeed mechanics

- **Adopt: result-cardinality (LIMIT) axis** (above) — a clean per-row conversion-cost slope and a natural small/large pairing for the instruction-count-gate-vs-walltime split already in (d).
- **Consider adopting: a peak-memory guard** for the O(rows) produce paths. bareduckdb tracks `getrusage` max RSS; codspeed walltime tracks neither memory nor allocations. A conversion regression is often memory-shaped (cf. the recorded fetchall +8% list->tuple edge-copy; the df() masked_array branch) — add a separate `getrusage`/memray delta assertion on `fetchall` and `df()`-with-nulls as a secondary signal, since a pure-timing gate can miss it.
- **Do NOT adopt their anti-patterns:** no-warmup + single subprocess run charges one-time import-cache population into the measurement and yields no statistics — bad for steady-state binding isolation. Our warmup + codspeed repeated rounds are correct; keep them.
- **Consistent with us:** their full-consume is eager `to_arrow_table()` and never `count(*)` — matches our discipline. Caveat: for their aggregate cases the arrow output is tiny, so the consume is trivial and the run is engine-only; our produce benchmarks must keep the materialization the heavy part (large output / top-N with large LIMIT).
