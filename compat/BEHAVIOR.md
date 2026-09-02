# Behavior changes against the shipping client

## Values and conversion

- A temporal beyond Python's datetime range raises `ConversionError`
  naming the value; the old client silently degraded it to its text
  form. Applies in `duckdb.compat` too.
  (xfail: `test_fetch_dict_coverage` cases 18/19/21/22/24)
- A MAP whose keys Python cannot hash comes back as `(key, value)`
  pairs; the old client returned a `{'key': [...], 'value': [...]}`
  dict of lists. (xfail: `test_fetch_dict_key_not_hashable`)
- Aware values carry stdlib `datetime.timezone` info, never pytz; and
  TIMESTAMPTZ is UTC-anchored until the seam exposes the session
  TimeZone, where the old client converted into the session zone.
  (divergence visible in `suite/api/test_cursor.py::test_cursor_timezone`)
- Parameter names must be strings on the native seam; the old client
  stringified any key. The leniency lives in `duckdb.compat` alone.
  (`tests/test_parameters.py`, `tests/test_compat.py`)

## Expressions and query building

- A bare string operand is a value: `col("state") == "active"` compares
  against the text; the old client read it as a reference to a column
  named `active`. (`tests/test_expr.py::TestStringOperandsAreLiterals`)
- `aggregate()` returns its group keys, and `select()` is pure
  projection that never groups. (`tests/test_frame.py`)
- Duplicate output names and join clashes are refused at build with a
  message; the old client let the engine silently bind the first.
  (`tests/test_frame.py::TestJoinRefusesToDuplicateAName`)

## Connections and transactions

- `sql()` over a held result follows the old client halfway: a held
  result nothing has fetched from is silently discarded, as the old
  client did; one that has been fetched from raises the engine's
  live-result refusal, where the old client kept the remainder because
  it had materialized the whole result at execute. Same for a
  relation's held result. Both raise loudly, never lose data silently;
  both resolve when statement classification moves from a probe
  execution to the parser (`sql_statement_type` on the seam).
- A row-producing CALL materializes once at `sql()`; the old client's
  relation re-ran it per fetch.

- `duckdb.connect()` carries no execute/fetch/cursor vocabulary; that
  contract lives in `duckdb.dbapi` (strict PEP 249) and `duckdb.compat`
  (the old shape). What `connect()` returns in 2.0 is an open roadmap
  decision, priced by the facade measurement.
- dbapi cursors share their connection's transaction, as PEP 249
  requires; the old client's `cursor()` was an independent duplicate.
  `duckdb.compat` keeps the old semantics, including closing cursors
  with their parent. (`tests/test_dbapi.py`, `tests/test_compat.py`)
- Opening one database file twice natively is refused by the engine's
  Environment guard; the old client shared one instance per path.
  `duckdb.compat` reproduces the shared instance, the config-mismatch
  refusal in the old words, and shared `:memory:name`.
  (`tests/test_connection.py`, `tests/test_compat.py`)
- `rowcount` is real on dbapi cursors and `run()`; the old client
  always reported -1, which `duckdb.compat` reproduces.
  (`tests/test_dbapi.py`, `tests/test_compat.py`)

## Errors and interruption

- Ctrl-C during a query raises `KeyboardInterrupt`; the old client
  converted it to `RuntimeError`. (`tests/test_connection.py`;
  divergence visible in `suite/api/test_query_interrupt.py`)
- Exception classes follow the engine's error-code space in
  `duckdb.exceptions`; the old names exist as aliases in
  `duckdb.compat`, with `ConnectionException` mapping to
  `InterfaceError`. (`tests/test_exceptions.py`, `tests/test_compat.py`)
- `interrupt()` on a closed connection raises `InterfaceError`; the old
  client raised `ConnectionException`. (`tests/test_connection.py`)

## Module surface

- `duckdb.sql()` returns a plan bound to no connection; the old
  module-level `sql`/`execute`/`query` ran on an implicit default
  connection, reproduced only in `duckdb.compat`.
  (`tests/test_frame.py`, `tests/test_compat.py`)
- `CompatRelation.types`/`dtypes` return type text; the old client
  returned `DuckDBPyType` objects, which compared equal to their text.
- A shorthand aggregate operand that fails to parse is quoted as an
  identifier, as the old client's fallback did; an operand like `a b`
  that the old parser accidentally read as `a AS b` (and then failed on)
  is quoted here instead and works.
- A `--` comment inside a shorthand aggregate operand is not stripped:
  the old client's parser round-trip silently discarded it, here the
  operand falls back to its quoted form and fails loudly as an unknown
  column. Block comments and `--` inside string literals are unaffected.
  (xfail: `test_comment_is_harmless`)

## Pending

- Multi-statement `execute`: the old client ran `"a; b; c"` and
  returned the last result; the seam refuses it until `ParseSQL` is
  bound. (`suite/test_multi_statement.py` fails as behavior)
- The implicit `FROM df` replacement scan of local dataframes and
  relations: M2 data-in. (`suite/test_metatransaction.py` and
  `suite/relational_api/test_rapi_query.py::test_replacement_scan_recursion`
  fail as behavior; `from_df` is missing surface)
- The old Expression-object API (`ColumnExpression`,
  `ConstantExpression`, ...): a future compat step.
  (`suite/relational_api/test_joins.py` fails at import)
