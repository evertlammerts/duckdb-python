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
- A Python float is DOUBLE: `lit(1.5)` renders `1.5::DOUBLE` and comes
  back as a float, and `[1.5]` or `{"a": 1.5}` types as `DOUBLE[]` or
  `STRUCT(a DOUBLE)` both in `types(con)` and at execution. The old client
  bound every float as DOUBLE too; neo's own earlier alphas wrote an inline
  float bare, which the engine read as `DECIMAL(2,1)` and returned as a
  `Decimal`. (`tests/test_names.py::TestFloatsAreDoubles`)

## Expressions and query building

- A bare string operand is a value: `col("state") == "active"` compares
  against the text; the old client read it as a reference to a column
  named `active`. (`tests/test_expr.py::TestStringOperandsAreLiterals`)
- `aggregate()` returns its group keys, and `select()` is pure
  projection that never groups. (`tests/test_frame.py`)
- Duplicate output names and join clashes are refused at build with a
  message; the old client let the engine silently bind the first.
  (`tests/test_frame.py::TestJoinRefusesToDuplicateAName`)
- `intersect()` and `except_()` on a `CompatRelation` render
  `INTERSECT ALL` and `EXCEPT ALL`, as the old client's did; the main
  API's `intersect()` and `except_()` drop duplicates.

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

## Functions

- `create_function` on a name already registered replaces the function;
  the old client refused with "already created" and required
  `remove_function` first. (`tests/test_udf.py`, `tests/test_compat.py`)
- `remove_function` raises `NotSupportedError`: the engine keeps a
  registered function until the database closes. The message points at
  re-registration, which replaces. (`tests/test_compat.py`)
- A function with neither type annotations nor explicit types is
  refused with directions; the old client fell back to ANY parameters,
  which the seam does not support yet. (`tests/test_compat.py`)
- A `KeyboardInterrupt` raised inside a running function surfaces as
  the query's `InvalidInputError`, not as `KeyboardInterrupt`: the
  engine owns the call and sees only its failure.

## Names

- A string that names a column, table, macro or function is one name,
  never split on dots: `col("user.id")` is the column called `user.id`,
  `table("a.b")` the table called `a.b`. A qualified name is a tuple,
  `table(("main", "orders"))`, at any depth. The old client and its
  Expression API split every dotted string. (`tests/test_names.py`,
  `TestAStringIsOneName`, `TestQualifiedNamesAreTuples`)
- A join condition is a function of the two sides,
  `on=lambda left, right: left["id"] == right["order_id"]`; the old
  `ColumnExpression("l.id")` spelling names a column called `l.id`. A
  USING join takes names: `on=[col("id")]` is refused, with a message
  pointing at the callable.
  (`tests/test_names.py::TestJoinSidesAreTheConditionsArguments`)
- Names compare as the engine compares them, case-insensitively in
  ASCII, quoted or not: `select("TOTAL")` finds `total`, and two columns
  whose names differ only in case are one name twice, refused where
  duplicates are refused (`values()`, a join without `suffix`, `rename`).
  (`tests/test_names.py::TestNamesCompareAsTheEngineDoes`)
- A struct field, map entry or list element is a bracket on the
  expression, `col("st")["a"]`; the old dotted `st.a` is a column name.
  (`tests/test_names.py::TestBracketsReachIntoValues`)
- A function name is written as the engine writes an identifier: bare
  when it is a plain identifier that is not one of the engine's keywords,
  quoted otherwise, never split. `sum("x")` and `read_csv(...)` are bare;
  `"range"(10)` and `"filter"(...)` are quoted, as the old client's
  `sql_query()` wrote them. The keyword list is generated from the
  engine's `duckdb_keywords()` by `scripts/gen_keywords.py` and checked
  by the suite and CI, so an engine bump that changes it fails until it
  is regenerated. COALESCE is syntax, not a function: `coalesce(...)`
  renders `COALESCE(...)`, and `fn("coalesce", ...)` fails because the
  engine has no function of that name. `create_function()` takes a bare
  string name; the engine registers a Python function by its bare name.
  (`tests/test_names.py::TestQualifiedNamesAreTuples`)
- `drop()` and `rename()` of a name holding a dot or a quote: the
  engine's `EXCLUDE` re-parses such a quoted name as a qualified path and
  errors, and its `RENAME` does the same and then silently renames
  nothing. `drop` renders `COLUMNS(lambda c: lower(c) NOT IN (...))`,
  order kept; `rename` lists the columns when it has the shape, in place,
  and without a connection raises `NeedsConnection`, as a suffixed join
  does, since there is no blind form that keeps the order. Both refuse a
  name the input does not have when the plan is resolved, `drop: no column
  'nope' in the input`, where the engine's `RENAME` stays silent even for a
  plain name.
  `star(exclude=[...])` is the engine's `EXCLUDE` as is.
  (`tests/test_names.py::TestAStringIsOneName`)
- `duckdb.compat` keeps the old split: `ColumnExpression("tbl_a.b")` is
  table `tbl_a` column `b`, `ColumnExpression("a", "b")` gives parts,
  `con.table("main.orders")` is qualified, with the engine's
  qualified-name grammar and messages. One difference: the old
  `ColumnExpression` silently dropped the middle parts of a name with
  four or more components; the face keeps them all. `ColumnExpression("")`
  is refused with `InvalidInputException`, where the old client reached an
  internal error. A malformed dotted name such as `a..b` raises an
  old-client error class, `InvalidInputException` from the verbs that
  build a plan and the engine's `ParserException` from the text verbs,
  never a bare `ValueError`. `update()` says the closed-connection words
  before looking at a name, like its siblings. The old client's bundled
  engine parsed names without the `""` escape and with one error message,
  so `"x""y"` was `xy` there and is `x"y` here; the face follows the
  engine it ships with.
  (`tests/test_names.py::TestCompatKeepsTheOldSplit`)

## Pending

- Multi-statement `execute`: the old client ran `"a; b; c"` and
  returned the last result; the seam refuses it until `ParseSQL` is
  bound. (`suite/test_multi_statement.py` fails as behavior)
- The implicit `FROM df` replacement scan of local dataframes and
  relations: M2 data-in. (`suite/test_metatransaction.py` and
  `suite/relational_api/test_rapi_query.py::test_replacement_scan_recursion`
  fail as behavior; `from_df` is missing surface)
- Arrow UDFs (`create_function(type='arrow')`): the Arrow wave.
- ANY-typed UDF parameters: needs the facade's bind callback to
  capture the bound argument types.
