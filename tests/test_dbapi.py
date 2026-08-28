"""PEP 249 conformance."""

from __future__ import annotations

import datetime
from collections.abc import Iterator

import pytest

from duckdb import dbapi


@pytest.fixture
def con() -> Iterator[dbapi.Connection]:
    connection = dbapi.connect()
    yield connection
    connection.close()


class TestModuleInterface:
    def test_globals(self) -> None:
        assert dbapi.apilevel == "2.0"
        assert dbapi.threadsafety == 1
        assert dbapi.paramstyle == "qmark"

    @pytest.mark.parametrize(
        "name",
        [
            "Warning",
            "Error",
            "InterfaceError",
            "DatabaseError",
            "DataError",
            "OperationalError",
            "IntegrityError",
            "InternalError",
            "ProgrammingError",
            "NotSupportedError",
        ],
    )
    def test_exception_is_exported(self, name: str) -> None:
        assert issubclass(getattr(dbapi, name), Exception)

    @pytest.mark.parametrize(
        ("constructor", "expected"),
        [
            ("DateFromTicks", datetime.date(1970, 1, 1)),
            ("TimeFromTicks", datetime.time(0, 0)),
        ],
    )
    def test_ticks_constructors(self, constructor: str, expected: object) -> None:
        # date.fromtimestamp takes no keyword arguments, unlike datetime's,
        # so the obvious implementation raises TypeError at runtime.
        assert getattr(dbapi, constructor)(0) == expected

    def test_binary_constructor(self) -> None:
        assert dbapi.Binary(bytearray(b"ab")) == b"ab"


class TestTypeObjects:
    """The DB-API type objects compare equal to the type codes in description."""

    def test_string_matches_varchar(self) -> None:
        assert dbapi.STRING == "VARCHAR"

    def test_number_matches_integers_and_decimal(self) -> None:
        assert dbapi.NUMBER == "BIGINT"
        # Parameterised types carry their arguments in the code.
        assert dbapi.NUMBER == "DECIMAL(18,3)"

    def test_datetime_matches_timestamps(self) -> None:
        assert dbapi.DATETIME == "TIMESTAMP WITH TIME ZONE"

    def test_sets_do_not_overlap(self) -> None:
        assert dbapi.STRING != "BIGINT"
        assert dbapi.NUMBER != "VARCHAR"

    def test_rowid_matches_nothing(self) -> None:
        # DuckDB has no row identifier type; the object exists because PEP 249
        # lists it, not because anything is ever equal to it.
        assert dbapi.ROWID != "BIGINT"

    def test_description_type_codes_compare(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT 'x' AS s, 1 AS n, TIMESTAMP '2026-01-01' AS t")
        assert cur.description is not None
        codes = [column[1] for column in cur.description]
        assert [codes[0] == dbapi.STRING, codes[1] == dbapi.NUMBER, codes[2] == dbapi.DATETIME] == [
            True,
            True,
            True,
        ]


class TestCursor:
    def test_description_is_none_without_a_result(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        assert cur.description is None

    def test_description_shape(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT 1 AS a")
        assert cur.description is not None
        assert len(cur.description[0]) == 7  # PEP 249 mandates seven fields
        assert cur.description[0][0] == "a"

    def test_fetchone_then_exhaustion(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT i FROM range(3) t(i)")
        assert [cur.fetchone(), cur.fetchone(), cur.fetchone()] == [(0,), (1,), (2,)]
        assert cur.fetchone() is None

    def test_fetchmany_uses_arraysize(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT i FROM range(5) t(i)")
        assert cur.fetchmany() == [(0,)]  # arraysize defaults to 1
        cur.arraysize = 3
        assert cur.fetchmany() == [(1,), (2,), (3,)]
        assert cur.fetchmany(10) == [(4,)]

    def test_fetch_crosses_chunk_boundaries(self, con: dbapi.Connection) -> None:
        # Chunks are 2048 rows, so this spans several and would break a
        # converter that forgot where it stopped inside one.
        cur = con.cursor()
        cur.execute("SELECT i FROM range(5000) t(i)")
        seen: list[tuple[int, ...]] = []
        while batch := cur.fetchmany(700):
            seen.extend(batch)
        assert seen == [(i,) for i in range(5000)]

    def test_mixed_fetches_share_one_position(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT i FROM range(5) t(i)")
        assert cur.fetchone() == (0,)
        assert cur.fetchmany(2) == [(1,), (2,)]
        assert cur.fetchall() == [(3,), (4,)]

    def test_iteration(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT i FROM range(3) t(i)")
        assert list(cur) == [(0,), (1,), (2,)]

    def test_executemany(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.executemany("INSERT INTO t VALUES (?)", [[1], [2], [3]])
        cur.execute("SELECT sum(v) FROM t")
        assert cur.fetchone() == (6,)

    def test_qmark_parameters(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT ? + ?", [2, 3])
        assert cur.fetchone() == (5,)

    def test_fetch_without_execute_is_an_interface_error(self, con: dbapi.Connection) -> None:
        # PEP 249: fetching before executing is an InterfaceError, because the
        # fault is in how the API was used, not in the database.
        cur = con.cursor()
        with pytest.raises(dbapi.InterfaceError):
            cur.fetchone()

    def test_use_after_close(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.close()
        with pytest.raises(dbapi.InterfaceError):
            cur.execute("SELECT 1")

    def test_close_is_idempotent(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.close()
        cur.close()

    def test_negative_fetchmany_is_rejected(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT 1")
        with pytest.raises(dbapi.ProgrammingError):
            cur.fetchmany(-1)

    def test_setinputsizes_and_setoutputsize_are_accepted(self, con: dbapi.Connection) -> None:
        # PEP 249 requires them to exist; DuckDB needs neither.
        cur = con.cursor()
        cur.setinputsizes([])
        cur.setoutputsize(0)

    def test_context_manager(self, con: dbapi.Connection) -> None:
        with con.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone() == (1,)
        with pytest.raises(dbapi.InterfaceError):
            cur.execute("SELECT 1")


class TestTransactions:
    """Cursors share their connection's transaction, as PEP 249 requires.

    The previous client gave every cursor its own engine connection, so two
    cursors could not see each other's uncommitted work.
    """

    def test_cursors_share_the_transaction(self, con: dbapi.Connection) -> None:
        writer, reader = con.cursor(), con.cursor()
        writer.execute("CREATE TABLE t (v INTEGER)")
        writer.execute("INSERT INTO t VALUES (1)")
        reader.execute("SELECT count(*) FROM t")
        assert reader.fetchone() == (1,), "a sibling cursor cannot see uncommitted work"

    def test_rollback_discards(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        con.commit()
        cur.execute("INSERT INTO t VALUES (1)")
        con.rollback()
        cur.execute("SELECT count(*) FROM t")
        assert cur.fetchone() == (0,)

    def test_commit_persists(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.execute("INSERT INTO t VALUES (1)")
        con.commit()
        con.rollback()  # nothing left to undo
        cur.execute("SELECT count(*) FROM t")
        assert cur.fetchone() == (1,)

    def test_commit_without_a_transaction_is_a_no_op(self, con: dbapi.Connection) -> None:
        con.commit()
        con.commit()

    def test_autocommit_skips_the_transaction(self) -> None:
        with dbapi.connect(autocommit=True) as con:
            cur = con.cursor()
            cur.execute("CREATE TABLE t (v INTEGER)")
            cur.execute("INSERT INTO t VALUES (1)")
            con.rollback()  # no transaction was open, so nothing is undone
            cur.execute("SELECT count(*) FROM t")
            assert cur.fetchone() == (1,)

    def test_context_manager_commits_on_success(self) -> None:
        con = dbapi.connect()
        with con:
            con.cursor().execute("CREATE TABLE t (v INTEGER)")
        cur = con.cursor()
        cur.execute("SELECT count(*) FROM t")
        assert cur.fetchone() == (0,)
        con.close()

    def test_context_manager_rolls_back_on_error(self) -> None:
        con = dbapi.connect()
        con.cursor().execute("CREATE TABLE t (v INTEGER)")
        con.commit()

        def fail_inside_the_block() -> None:
            with con:
                con.cursor().execute("INSERT INTO t VALUES (1)")
                message = "deliberate"
                raise RuntimeError(message)

        with pytest.raises(RuntimeError):
            fail_inside_the_block()
        cur = con.cursor()
        cur.execute("SELECT count(*) FROM t")
        assert cur.fetchone() == (0,)
        con.close()


class TestConnection:
    def test_close_is_idempotent(self) -> None:
        con = dbapi.connect()
        con.close()
        con.close()

    def test_use_after_close(self) -> None:
        con = dbapi.connect()
        con.close()
        with pytest.raises(dbapi.InterfaceError):
            con.cursor()

    def test_open_time_options(self) -> None:
        with dbapi.connect(threads="2") as con:
            cur = con.cursor()
            cur.execute("SELECT current_setting('threads')")
            assert cur.fetchone() == (2,)  # the setting comes back typed

    def test_errors_are_typed(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        with pytest.raises(dbapi.DatabaseError):
            cur.execute("SELECT * FROM no_such_table")


class TestRowcount:
    """Real DML counts, where the previous client always reported -1."""

    def test_insert_reports_rows_written(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.execute("INSERT INTO t VALUES (1), (2), (3)")
        assert cur.rowcount == 3

    def test_update_and_delete_report_rows_touched(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.execute("INSERT INTO t VALUES (1), (2), (3)")
        cur.execute("UPDATE t SET v = v + 1 WHERE v > 1")
        assert cur.rowcount == 2
        cur.execute("DELETE FROM t WHERE v > 2")
        assert cur.rowcount == 2

    def test_select_reports_minus_one(self, con: dbapi.Connection) -> None:
        # The result is streamed, so the count is not known up front.
        cur = con.cursor()
        cur.execute("SELECT 1")
        assert cur.rowcount == -1


class TestResultExclusivity:
    """The engine allows one live result per connection; cursors share one."""

    def test_a_sibling_execute_invalidates_an_unread_result(self, con: dbapi.Connection) -> None:
        first, second = con.cursor(), con.cursor()
        first.execute("SELECT i FROM range(100) t(i)")
        assert first.fetchone() == (0,)
        second.execute("SELECT 1")
        # PEP 249 permits this, and it is what ODBC does without MARS.
        with pytest.raises(dbapi.InterfaceError):
            first.fetchone()

    def test_re_executing_releases_the_previous_result(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT i FROM range(100) t(i)")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT 42")
        assert cur.fetchone() == (42,)


class TestFailedExecute:
    """A statement that raises leaves no metadata behind.

    `description` documents itself as the last query's columns. After a failed
    execute there is no last query, so keeping the previous one's values would
    describe a statement that never ran.
    """

    def test_description_is_cleared(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT 1 AS a")
        assert cur.description is not None
        with pytest.raises(dbapi.DatabaseError):
            cur.execute("SELECT * FROM no_such_table")
        assert cur.description is None

    def test_rowcount_is_cleared(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.execute("INSERT INTO t VALUES (1), (2)")
        assert cur.rowcount == 2
        with pytest.raises(dbapi.DatabaseError):
            cur.execute("INSERT INTO no_such_table VALUES (1)")
        assert cur.rowcount == -1

    def test_a_syntax_error_clears_it_too(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("SELECT 1 AS a")
        with pytest.raises(dbapi.DatabaseError):
            cur.execute("SELECT FROM WHERE")
        assert cur.description is None


class TestExecutemanyRowcount:
    """rowcount is the total across parameter sets, as sqlite3 reports."""

    def test_insert_totals_across_sets(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.executemany("INSERT INTO t VALUES (?)", [[1], [2], [3]])
        assert cur.rowcount == 3

    def test_multi_row_sets_total(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.executemany("INSERT INTO t VALUES (?), (?)", [[1, 2], [3, 4]])
        assert cur.rowcount == 4

    def test_empty_sequence_leaves_it_undefined(self, con: dbapi.Connection) -> None:
        cur = con.cursor()
        cur.execute("CREATE TABLE t (v INTEGER)")
        cur.executemany("INSERT INTO t VALUES (?)", [])
        assert cur.rowcount == -1

    def test_row_producing_statement_stays_undefined(self, con: dbapi.Connection) -> None:
        # PEP 249 leaves rowcount undefined when the statement produces rows.
        cur = con.cursor()
        cur.executemany("SELECT ?", [[1], [2]])
        assert cur.rowcount == -1
