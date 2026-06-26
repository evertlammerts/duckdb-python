"""Tests for the MaterializedRelation re-feed Arrow design.

The redesign promotes an already-executed result back into a relation and re-feeds
it through the engine, so there is a single uniform Arrow path.

Routing is by *result type*, and a lazy result is never materialized:

* ``MaterializedQueryResult`` (from ``rel.execute()``) — a ColumnDataCollection in
  memory — is the only result re-fed back through the engine: ``to_arrow_table`` /
  ``pl`` run it through a ``PhysicalArrowCollector`` (eager + parallel); the lazy
  surfaces re-run it as a ``StreamQueryResult``. Both run on the user's own context,
  which the produced stream co-owns (so it survives ``del conn``).
* ``StreamQueryResult`` (from ``con.execute()``, or a fresh streaming relation)
  already has a live context, so it is converted directly: ``to_arrow_table`` pulls
  the stream serially; ``__arrow_c_stream__`` / ``to_arrow_reader`` wrap it directly
  in core's ``ResultArrowArrayStreamWrapper`` — never copied. Such a reader/capsule
  shares the connection's single active-stream slot (consume before reusing the
  connection) and survives ``del conn``.

The headline correctness win — GEOMETRY conversion after the originating connection
closes — is exercised directly (see ``TestGeometryAfterClose``): DuckDB's built-in
``GEOMETRY`` type maps to the ``geoarrow.wkb`` Arrow extension type, whose conversion
needs a live ``ClientContext`` / type-extension registry. The lazy paths run on a
context the ``StreamQueryResult`` co-owns, so the conversion succeeds even after the
originating connection is destroyed.
"""

import gc

import pytest

import duckdb

pa = pytest.importorskip("pyarrow")


# A spread of types that exercise the re-feed: ints, floats, strings, bools,
# nullables, nested struct/list/map, and an enum-backed categorical.
RICH_SQL = """
    SELECT
        i AS int_col,
        i::DOUBLE AS double_col,
        'row_' || i::VARCHAR AS str_col,
        i % 2 = 0 AS bool_col,
        CASE WHEN i % 3 = 0 THEN NULL ELSE i END AS nullable_col,
        {'x': i, 'y': i::VARCHAR} AS struct_col,
        [i, i + 1, i + 2] AS list_col,
        MAP {i::VARCHAR: i * 10} AS map_col,
    FROM range(1000) t(i)
"""


class TestEagerToArrowTablePromotion:
    """Eager + parallel promotion of already-executed results matches the fresh path."""

    def test_cursor_fetch_arrow_table_matches_fresh(self):
        conn = duckdb.connect()
        expected = conn.sql(RICH_SQL).to_arrow_table()
        # con.execute(...) materializes; fetch goes through the promotion path.
        actual = conn.execute(RICH_SQL).to_arrow_table()
        assert actual.equals(expected)

    def test_preexecuted_relation_to_arrow_table_matches_fresh(self):
        conn = duckdb.connect()
        expected = conn.sql(RICH_SQL).to_arrow_table()
        rel = conn.sql(RICH_SQL)
        rel.execute()  # forces a MaterializedQueryResult
        actual = rel.to_arrow_table()
        assert actual.equals(expected)

    def test_cursor_pl_matches_fresh(self):
        pl = pytest.importorskip("polars")
        conn = duckdb.connect()
        sql = "SELECT i AS a, i::VARCHAR AS b FROM range(500) t(i)"
        expected = conn.sql(sql).pl()
        actual = conn.execute(sql).pl()
        assert expected.equals(actual)
        assert isinstance(actual, pl.DataFrame)

    def test_cursor_fetch_arrow_table_empty(self):
        conn = duckdb.connect()
        sql = "SELECT i AS a, i::VARCHAR AS b FROM range(10) t(i) WHERE i < 0"
        expected = conn.sql(sql).to_arrow_table()
        actual = conn.execute(sql).to_arrow_table()
        assert actual.num_rows == 0
        assert actual.schema.equals(expected.schema)


class TestLazyCapsuleStreaming:
    """``__arrow_c_stream__`` is now a lazy streaming object."""

    def test_fresh_capsule_matches_to_arrow_table(self):
        conn = duckdb.connect()
        expected = conn.sql(RICH_SQL).to_arrow_table()
        actual = pa.table(conn.sql(RICH_SQL))
        assert actual.equals(expected)

    def test_preexecuted_capsule_matches_to_arrow_table(self):
        conn = duckdb.connect()
        expected = conn.sql(RICH_SQL).to_arrow_table()
        rel = conn.sql(RICH_SQL)
        rel.execute()  # MaterializedQueryResult -> re-fed through the engine as a stream
        actual = pa.table(rel)
        assert actual.equals(expected)

    def test_fresh_capsule_survives_del_conn(self):
        """Fresh capsule survives ``del conn``: the StreamQueryResult owns the context (#492)."""
        conn = duckdb.connect()
        capsule = conn.sql(RICH_SQL).__arrow_c_stream__()  # noqa: F841
        del conn
        gc.collect()
        out = duckdb.connect().sql("SELECT * FROM capsule").to_arrow_table()
        assert out.num_rows == 1000

    def test_preexecuted_capsule_survives_del_conn(self):
        """Pre-executed (materialized) capsule survives ``del conn``: the re-fed stream owns the context."""
        conn = duckdb.connect()
        rel = conn.sql(RICH_SQL)
        rel.execute()
        capsule = rel.__arrow_c_stream__()  # noqa: F841
        del rel, conn
        gc.collect()
        out = duckdb.connect().sql("SELECT * FROM capsule").to_arrow_table()
        assert out.num_rows == 1000

    def test_fresh_capsule_consume_then_reuse_same_connection(self):
        """Streaming-object contract: consume the fresh capsule before reusing the connection."""
        conn = duckdb.connect()
        sql = "SELECT i FROM range(100) t(i)"
        c1 = conn.sql(sql).__arrow_c_stream__()
        first = pa.RecordBatchReader._import_from_c_capsule(c1).read_all()
        assert first.num_rows == 100
        # After fully consuming the first stream, the slot is free for the next.
        c2 = conn.sql(sql).__arrow_c_stream__()
        second = pa.RecordBatchReader._import_from_c_capsule(c2).read_all()
        assert second.num_rows == 100


class TestCursorRecordBatchReaderStreaming:
    """Cursor reader over a stream: not materialized, survives del conn, shares the active-stream slot."""

    def test_reader_consume_then_reuse_connection(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(3000)")
        reader = conn.execute("SELECT a FROM t").to_arrow_reader(1024)
        tbl = reader.read_all()  # consume before reusing the connection
        assert tbl.num_rows == 3000
        assert tbl.column("a").to_pylist() == list(range(3000))
        # the connection is free to use again
        assert conn.execute("SELECT 42").fetchone()[0] == 42

    def test_reader_survives_del_conn(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(3000)")
        reader = conn.execute("SELECT a FROM t").to_arrow_reader(1024)
        del conn
        gc.collect()
        tbl = reader.read_all()
        assert tbl.num_rows == 3000

    def test_cursor_reader_exact_batch_sizes(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT range AS a FROM range(3000)")
        reader = conn.execute("SELECT a FROM t").to_arrow_reader(1024)
        assert reader.read_next_batch().num_rows == 1024
        assert reader.read_next_batch().num_rows == 1024
        assert reader.read_next_batch().num_rows == 952
        with pytest.raises(StopIteration):
            reader.read_next_batch()


class TestDuplicateColumnNames:
    """Duplicate output column names must survive the re-feed (promotion restores them)."""

    DUP_SQL = "SELECT i AS a, i + 1 AS a, i + 2 AS a FROM range(50) t(i)"

    def test_cursor_fetch_arrow_table_duplicate_columns(self):
        conn = duckdb.connect()
        tbl = conn.execute(self.DUP_SQL).to_arrow_table()
        assert tbl.num_rows == 50
        assert tbl.num_columns == 3
        # Re-feeding through a MaterializedRelation re-binds (which would dedup the
        # names); the promotion restores the original names, so pyarrow still sees
        # the duplicate 'a' columns exactly as the un-promoted result would.
        assert tbl.column_names == ["a", "a", "a"]
        assert tbl.column(0).to_pylist() == list(range(50))
        assert tbl.column(1).to_pylist() == list(range(1, 51))
        assert tbl.column(2).to_pylist() == list(range(2, 52))

    def test_preexecuted_capsule_duplicate_columns(self):
        conn = duckdb.connect()
        rel = conn.sql(self.DUP_SQL)
        rel.execute()
        tbl = pa.table(rel)
        assert tbl.num_rows == 50
        assert tbl.num_columns == 3

    def test_cursor_record_batch_duplicate_columns(self):
        conn = duckdb.connect()
        reader = conn.execute(self.DUP_SQL).to_arrow_reader()
        tbl = reader.read_all()
        assert tbl.num_rows == 50
        assert tbl.num_columns == 3

    def test_cursor_pl_duplicate_columns_dedups(self):
        pl = pytest.importorskip("polars")
        conn = duckdb.connect()
        df = conn.execute(self.DUP_SQL).pl()
        # polars requires unique column names; the dedup must still apply.
        assert isinstance(df, pl.DataFrame)
        assert len(set(df.columns)) == 3


class TestEdgeCaseShapes:
    # Note: a 0-column result is not constructible via SQL ("SELECT FROM ..." is a
    # parser error), so the "0 columns" risk from CONTEXT.md is not reachable here.

    def test_single_row_single_column(self):
        conn = duckdb.connect()
        tbl = conn.execute("SELECT 42 AS x").to_arrow_table()
        assert tbl.num_rows == 1
        assert tbl.column("x").to_pylist() == [42]

    def test_enum_categorical_roundtrip(self):
        conn = duckdb.connect()
        conn.execute("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        conn.execute("CREATE TABLE t AS SELECT 'happy'::mood AS m FROM range(100)")
        expected = conn.sql("SELECT m FROM t").to_arrow_table()
        actual = conn.execute("SELECT m FROM t").to_arrow_table()
        assert actual.equals(expected)
        # ENUM should map to a dictionary-encoded Arrow column.
        assert pa.types.is_dictionary(actual.schema.field("m").type)

    def test_enum_via_cursor_stream(self):
        conn = duckdb.connect()
        conn.execute("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        conn.execute("CREATE TABLE t AS SELECT 'ok'::mood AS m FROM range(100)")
        reader = conn.execute("SELECT m FROM t").to_arrow_reader()
        tbl = reader.read_all()
        assert tbl.num_rows == 100
        assert pa.types.is_dictionary(tbl.schema.field("m").type)


class TestConfigFidelity:
    """Re-fed / directly-wrapped results reproduce the user's Arrow output config (e.g. TimeZone)."""

    def test_timezone_preserved_through_cursor_stream_reader(self):
        conn = duckdb.connect()
        conn.execute("SET TimeZone = 'America/New_York'")
        reader = conn.execute("SELECT TIMESTAMPTZ '2021-06-01 12:00:00' AS ts").to_arrow_reader()
        tbl = reader.read_all()
        assert tbl.schema.field("ts").type.tz == "America/New_York"

    def test_timezone_preserved_through_preexec_capsule(self):
        conn = duckdb.connect()
        conn.execute("SET TimeZone = 'Asia/Kathmandu'")
        rel = conn.sql("SELECT TIMESTAMPTZ '2021-06-01 12:00:00' AS ts")
        rel.execute()
        tbl = pa.table(rel)
        assert tbl.schema.field("ts").type.tz == "Asia/Kathmandu"

    def test_large_buffer_size_preserved_through_cursor_stream(self):
        # arrow_large_buffer_size promotes string/blob/list offsets to 64-bit.
        conn = duckdb.connect()
        conn.execute("SET arrow_large_buffer_size = true")
        reader = conn.execute("SELECT 'hello' AS s FROM range(10)").to_arrow_reader()
        tbl = reader.read_all()
        assert tbl.schema.field("s").type == pa.large_string()


class TestLazyStreamMechanism:
    """Mechanism behind the GEOMETRY-after-close win: a lazy re-fed stream is correct after del conn."""

    def test_refed_stream_data_correct_after_del_conn(self):
        conn = duckdb.connect()
        conn.execute("CREATE TABLE t AS SELECT i, i::VARCHAR AS s FROM range(2000) t(i)")
        reader = conn.execute("SELECT * FROM t ORDER BY i").to_arrow_reader(512)
        del conn
        gc.collect()
        tbl = reader.read_all()
        assert tbl.num_rows == 2000
        assert tbl.column("i").to_pylist() == list(range(2000))


class TestGeometryAfterClose:
    """The headline correctness win, now directly testable.

    DuckDB's built-in GEOMETRY type maps to the ``geoarrow.wkb`` Arrow extension type,
    whose conversion requires a live ClientContext / type-extension registry. The lazy
    capsule/reader run on a context the StreamQueryResult co-owns, so the conversion
    (schema + WKB data) succeeds AFTER the originating connection is destroyed. Before
    this design, the conversion ran against a dangling context (use-after-free / wrong
    schema). GEOMETRY is a core type here (no spatial extension needed).
    """

    GEOM_SQL = "SELECT id, ('POINT(' || id || ' ' || id || ')')::GEOMETRY AS g FROM range(8) t(id)"

    def _expected(self):
        # deterministic query -> compute the reference on an independent connection
        return duckdb.connect().sql(self.GEOM_SQL).to_arrow_table()

    @staticmethod
    def _assert_geoarrow(tbl) -> None:
        field = tbl.schema.field("g")
        assert field.metadata is not None
        assert field.metadata[b"ARROW:extension:name"] == b"geoarrow.wkb"

    def test_geometry_to_arrow_is_geoarrow_wkb(self):
        # sanity: the build really has built-in GEOMETRY mapping to geoarrow.wkb
        tbl = self._expected()
        self._assert_geoarrow(tbl)
        assert tbl.num_rows == 8

    def test_fresh_capsule_geometry_after_del_conn(self):
        expected = self._expected()
        conn = duckdb.connect()
        capsule = conn.sql(self.GEOM_SQL).__arrow_c_stream__()
        del conn
        gc.collect()
        actual = pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
        self._assert_geoarrow(actual)
        assert actual.column("g").to_pylist() == expected.column("g").to_pylist()

    def test_preexecuted_capsule_geometry_after_del_conn(self):
        # MaterializedQueryResult -> re-fed as a stream; conversion runs after del conn
        expected = self._expected()
        conn = duckdb.connect()
        rel = conn.sql(self.GEOM_SQL)
        rel.execute()
        capsule = rel.__arrow_c_stream__()
        del rel, conn
        gc.collect()
        actual = pa.RecordBatchReader._import_from_c_capsule(capsule).read_all()
        self._assert_geoarrow(actual)
        assert actual.column("g").to_pylist() == expected.column("g").to_pylist()

    def test_cursor_reader_geometry_after_del_conn(self):
        # con.execute() stream wrapped directly; conversion runs after del conn
        expected = self._expected()
        conn = duckdb.connect()
        reader = conn.execute(self.GEOM_SQL).to_arrow_reader()
        del conn
        gc.collect()
        actual = reader.read_all()
        self._assert_geoarrow(actual)
        assert actual.column("g").to_pylist() == expected.column("g").to_pylist()

    def test_preexecuted_to_arrow_table_geometry(self):
        # eager parallel re-feed (SelectStatement + PhysicalArrowCollector) of geometry
        expected = self._expected()
        conn = duckdb.connect()
        rel = conn.sql(self.GEOM_SQL)
        rel.execute()
        actual = rel.to_arrow_table()
        self._assert_geoarrow(actual)
        assert actual.column("g").to_pylist() == expected.column("g").to_pylist()
