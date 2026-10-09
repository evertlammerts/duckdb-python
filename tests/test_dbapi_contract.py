"""The agreed execute() contract: settling, supersede-with-drain, busy refusals, complete and abort.

Ported from the old client's contract tests in cursor vocabulary. Not portable to this face and so
absent here: scripts (execute parses exactly one statement), UDFs, relations, and polars sources.
Divergence kept deliberately: register() is a registry write here, no statement runs, so it stays
free while a result is open.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from duckdb import dbapi, exceptions

ROWS = 1_000_000
VECTOR = 2048
#: The engine surfaces a stream's first batch only once its streaming buffer is full, so settling runs
#: one buffer, not one vector; the fixtures shrink the buffer to 64KB (8192 BIGINTs) to pin the bound.
SETTLED = 4 * VECTOR


def _bounded(connection: dbapi.Connection) -> None:
    cursor = connection.cursor()
    # One thread, and a small streaming buffer, bound how far a statement runs before it is read.
    cursor.execute("SET threads = 1")
    cursor.execute("SET max_streaming_buffer_size = '64KB'")
    cursor.close()


@pytest.fixture
def con() -> Iterator[dbapi.Connection]:
    connection = dbapi.connect(autocommit=True)
    _bounded(connection)
    yield connection
    connection.close()


@pytest.fixture
def transactional() -> Iterator[dbapi.Connection]:
    connection = dbapi.connect()
    _bounded(connection)
    connection.commit()
    yield connection
    connection.close()


@pytest.fixture
def sequenced(con: dbapi.Connection) -> dbapi.Cursor:
    cursor = con.cursor()
    cursor.execute("CREATE SEQUENCE s")
    return cursor


def currval(con: dbapi.Connection) -> int:
    cursor = con.cursor()
    cursor.execute("SELECT currval('s')")
    row = cursor.fetchone()
    assert row is not None
    return int(row[0])


class TestExecuteSettles:
    def test_an_error_in_the_first_chunk_raises_from_execute(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        with pytest.raises(exceptions.InvalidInputError, match="boom"):
            cursor.execute("SELECT error('boom')")
        assert cursor.description is None
        assert cursor.rowcount == -1

    def test_a_parameterized_error_raises_from_execute(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        with pytest.raises(exceptions.InvalidInputError, match="boom"):
            cursor.execute("SELECT error(?)", ["boom"])

    def test_the_cursor_recovers_after_a_failed_execute(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        with pytest.raises(exceptions.InvalidInputError):
            cursor.execute("SELECT error('boom')")
        cursor.execute("SELECT 42")
        assert cursor.fetchall() == [(42,)]

    def test_the_first_chunk_runs_and_the_rest_waits(self, sequenced: dbapi.Cursor, con: dbapi.Connection) -> None:
        sequenced.execute(f"SELECT nextval('s') FROM range({ROWS})")
        # The superseding read cancels the SELECT where it settled, keeping its side effects.
        ran = currval(con)
        assert 0 < ran <= SETTLED

    def test_a_superseded_select_runs_no_further(self, sequenced: dbapi.Cursor, con: dbapi.Connection) -> None:
        sequenced.execute(f"SELECT nextval('s') FROM range({ROWS})")
        ran = currval(con)
        time.sleep(0.3)
        assert currval(con) == ran

    def test_settling_loses_no_rows(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        assert cursor.description is not None
        assert cursor.description[0][0] == "range"
        rows = cursor.fetchall()
        assert len(rows) == ROWS
        assert rows[:3] == [(0,), (1,), (2,)]
        assert rows[-1] == (ROWS - 1,)

    def test_an_empty_result_settles_with_its_schema(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute("SELECT 1 AS a WHERE false")
        assert cursor.description is not None
        assert cursor.description[0][0] == "a"
        assert cursor.fetchall() == []

    def test_an_error_after_the_first_chunk_waits_for_the_fetch(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT CASE WHEN i = {ROWS - 1} THEN error('late') ELSE i END FROM range({ROWS}) t(i)")
        with pytest.raises(exceptions.InvalidInputError, match="late"):
            cursor.fetchall()

    def test_an_interrupt_while_settling_raises_from_execute(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        timer = threading.Timer(0.3, con.interrupt)
        timer.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                # An aggregate emits its only row at the end and needs no memory while it runs.
                cursor.execute("SELECT sum(a.range * b.range) FROM range(200000) a, range(200000) b")
        finally:
            timer.cancel()
        cursor.execute("SELECT 42")
        assert cursor.fetchall() == [(42,)]

    def test_a_failed_select_fails_the_transaction(self, transactional: dbapi.Connection) -> None:
        cursor = transactional.cursor()
        cursor.execute("CREATE TABLE t AS SELECT 1 AS a")
        with pytest.raises(exceptions.InvalidInputError):
            cursor.execute("SELECT error('boom')")
        with pytest.raises(exceptions.TransactionError):
            cursor.execute("SELECT 1")
        transactional.rollback()
        cursor.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 't'")
        assert cursor.fetchone() == (0,)

    def test_a_write_that_fails_late_raises_from_execute_and_leaves_no_trace(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"CREATE TABLE t AS SELECT range AS i FROM range({ROWS})")
        late = f"CASE WHEN i = {ROWS - 1} THEN error('boom')::BIGINT ELSE i + 1 END"
        with pytest.raises(exceptions.InvalidInputError, match="boom"):
            cursor.execute(f"UPDATE t SET i = {late}")
        cursor.execute("SELECT count(*), sum(i) FROM t")
        assert cursor.fetchone() == (ROWS, ROWS * (ROWS - 1) // 2)


class TestReturning:
    def test_returning_completes_in_execute(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute("CREATE TABLE t AS SELECT range AS i FROM range(5000)")
        cursor.execute("DELETE FROM t WHERE i < 2500 RETURNING i")
        # Reading the count on a second cursor supersedes the RETURNING result; the write must survive.
        other = con.cursor()
        other.execute("SELECT count(*) FROM t")
        assert other.fetchone() == (2500,)

    def test_returning_rows_are_fetchable(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute("CREATE TABLE t AS SELECT range AS i FROM range(100)")
        cursor.execute("INSERT INTO t SELECT range FROM range(100, 200) RETURNING i")
        assert sorted(cursor.fetchall()) == [(i,) for i in range(100, 200)]

    def test_unread_returning_survives_the_next_statement(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute("CREATE TABLE t AS SELECT range AS i FROM range(5000)")
        cursor.execute("DELETE FROM t RETURNING i")
        cursor.execute("SELECT count(*) FROM t")
        assert cursor.fetchone() == (0,)


class TestSupersede:
    def test_a_new_statement_replaces_the_open_result(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        assert cursor.fetchone() == (0,)
        cursor.execute("SELECT 42")
        assert cursor.fetchall() == [(42,)]

    def test_a_superseded_cursors_fetch_raises(self, con: dbapi.Connection) -> None:
        first = con.cursor()
        first.execute(f"SELECT * FROM range({ROWS})")
        second = con.cursor()
        second.execute("SELECT 42")
        with pytest.raises(exceptions.InterfaceError):
            first.fetchone()
        assert second.fetchall() == [(42,)]

    def test_a_superseded_read_only_select_leaves_the_transaction_valid(self, transactional: dbapi.Connection) -> None:
        cursor = transactional.cursor()
        cursor.execute("CREATE TABLE t AS SELECT 1 AS a")
        cursor.execute(f"SELECT * FROM range({ROWS})")
        assert cursor.fetchone() == (0,)
        cursor.execute("INSERT INTO t VALUES (2)")
        transactional.commit()
        cursor.execute("SELECT a FROM t ORDER BY a")
        assert cursor.fetchall() == [(1,), (2,)]

    def test_a_superseded_returning_write_is_drained_not_cancelled(self, transactional: dbapi.Connection) -> None:
        cursor = transactional.cursor()
        cursor.execute("CREATE TABLE t (i BIGINT)")
        cursor.execute("INSERT INTO t SELECT range FROM range(5000) RETURNING i")
        assert cursor.fetchone() == (0,)
        # A drained supersede keeps the transaction valid; a cancel would fail it.
        cursor.execute("SELECT count(*) FROM t")
        assert cursor.fetchone() == (5000,)
        transactional.commit()

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="a write hidden in a SELECT reads as read-only until the engine exposes statement properties",
    )
    def test_a_superseded_select_that_writes_is_drained(self, sequenced: dbapi.Cursor, con: dbapi.Connection) -> None:
        sequenced.execute(f"SELECT nextval('s') FROM range({ROWS})")
        assert currval(con) == ROWS

    def test_commit_while_a_read_only_result_is_open(self, transactional: dbapi.Connection) -> None:
        cursor = transactional.cursor()
        cursor.execute("CREATE TABLE t AS SELECT 1 AS a")
        cursor.execute(f"SELECT * FROM range({ROWS})")
        assert cursor.fetchone() == (0,)
        transactional.commit()
        cursor.execute("SELECT a FROM t")
        assert cursor.fetchall() == [(1,)]


class TestCompleteAndAbort:
    def test_complete_runs_the_rest_of_the_result(self, sequenced: dbapi.Cursor, con: dbapi.Connection) -> None:
        sequenced.execute(f"SELECT nextval('s') FROM range({ROWS})")
        sequenced.complete()
        assert currval(con) == ROWS

    def test_complete_without_a_result_is_a_no_op(self, con: dbapi.Connection) -> None:
        con.cursor().complete()

    def test_complete_keeps_a_transaction_valid_after_a_write(self, transactional: dbapi.Connection) -> None:
        cursor = transactional.cursor()
        cursor.execute("CREATE TABLE t (i BIGINT)")
        cursor.execute("INSERT INTO t SELECT range FROM range(5000) RETURNING i")
        cursor.complete()
        transactional.commit()
        cursor.execute("SELECT count(*) FROM t")
        assert cursor.fetchone() == (5000,)

    def test_abort_stops_the_result_where_it_settled(self, sequenced: dbapi.Cursor, con: dbapi.Connection) -> None:
        sequenced.execute(f"SELECT nextval('s') FROM range({ROWS})")
        sequenced.abort()
        ran = currval(con)
        assert 0 < ran <= SETTLED

    def test_abort_frees_the_cursor_for_the_next_statement(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        cursor.abort()
        with pytest.raises(exceptions.InterfaceError):
            cursor.fetchone()
        cursor.execute("SELECT 42")
        assert cursor.fetchall() == [(42,)]


class TestDisposalNeverCutsAWrite:
    def test_executemany_with_returning_applies_every_set(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute("CREATE TABLE t (lo BIGINT, hi BIGINT)")
        cursor.executemany(
            "INSERT INTO t SELECT ?, range FROM range(?, ? + 5000) RETURNING lo", [(1, 0, 0), (2, 5000, 5000)]
        )
        other = con.cursor()
        other.execute("SELECT count(*) FROM t")
        assert other.fetchone() == (10000,)

    def test_cursor_close_runs_out_a_pending_write(self, transactional: dbapi.Connection) -> None:
        cursor = transactional.cursor()
        cursor.execute("CREATE TABLE t (i BIGINT)")
        cursor.execute("INSERT INTO t SELECT range FROM range(5000) RETURNING i")
        cursor.close()
        transactional.commit()
        other = transactional.cursor()
        other.execute("SELECT count(*) FROM t")
        assert other.fetchone() == (5000,)

    def test_complete_and_abort_on_a_closed_cursor_are_no_ops(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        cursor.complete()
        cursor.abort()


class TestBusyRefusal:
    def test_close_refused_mid_drive_leaves_the_connection_open(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        with con._driving(), pytest.raises(dbapi.ProgrammingError, match="open result"):
            con.close()
        cursor.execute("SELECT 42")
        assert cursor.fetchall() == [(42,)]

    def test_cursor_close_refused_mid_drive_leaves_the_cursor_open(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        with con._driving(), pytest.raises(dbapi.ProgrammingError, match="open result"):
            cursor.close()
        assert cursor.fetchone() == (0,)
        cursor.close()

    def test_a_driving_connection_refuses_a_statement(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        with con._driving(), pytest.raises(dbapi.ProgrammingError, match="open result"):
            cursor.execute("SELECT 1")

    def test_a_driving_connection_refuses_fetches_and_disposal(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        with con._driving():
            with pytest.raises(dbapi.ProgrammingError, match="open result"):
                cursor.fetchone()
            with pytest.raises(dbapi.ProgrammingError, match="open result"):
                cursor.complete()
            with pytest.raises(dbapi.ProgrammingError, match="open result"):
                cursor.abort()
        assert cursor.fetchone() == (0,)

    def test_interrupt_bypasses_the_busy_claim(self, con: dbapi.Connection) -> None:
        with con._driving():
            con.interrupt()

    def test_a_concurrent_statement_never_hangs(self, con: dbapi.Connection) -> None:
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        outcomes: list[object] = []

        def contend() -> None:
            other = con.cursor()
            try:
                other.execute("SELECT 42")
                outcomes.append(other.fetchall())
            except dbapi.ProgrammingError as error:
                outcomes.append(error)

        thread = threading.Thread(target=contend)
        thread.start()
        # The contender either supersedes first (this fetch then raises) or is refused mid-drive.
        try:
            rows = cursor.fetchall()
            assert len(rows) == ROWS
        except dbapi.InterfaceError:
            pass
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert outcomes
        assert outcomes[0] == [(42,)] or isinstance(outcomes[0], dbapi.ProgrammingError)


class TestRegisterDivergence:
    def test_register_is_free_while_a_result_is_open(self, con: dbapi.Connection) -> None:
        np = pytest.importorskip("numpy")
        cursor = con.cursor()
        cursor.execute(f"SELECT * FROM range({ROWS})")
        con.register("v", {"a": np.arange(3)})
        # The open result reads on, untouched by the registration.
        assert cursor.fetchone() == (0,)
        other = con.cursor()
        other.execute("SELECT count(*) FROM v")
        assert other.fetchone() == (3,)
