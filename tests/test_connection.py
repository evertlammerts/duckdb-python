"""Connection behaviour: options, interruption, and statement handling."""

from __future__ import annotations

import decimal
import gc
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import duckdb
from duckdb import _duckdb, exceptions

from ._support import ctrl_c_after


def test_settings_round_trip() -> None:
    con = _duckdb.Database(":memory:").connect()
    con.set_option("threads", "3")
    assert con.get_option("threads") == "3"


def test_unknown_setting_is_reported() -> None:
    con = _duckdb.Database(":memory:").connect()
    with pytest.raises(exceptions.Error):
        con.get_option("no_such_setting_at_all")


def test_database_accepts_open_time_options() -> None:
    # Some settings can only be chosen before the database exists, so this is not the same as set_option.
    con = _duckdb.Database(":memory:", [("threads", "2")]).connect()
    assert con.get_option("threads") == "2"


def test_connect_accepts_a_path(tmp_path: Path) -> None:
    # A Path used to reach DuckDB unconverted and be refused as the wrong argument type.
    with duckdb.frame.connect(tmp_path / "by_path.db") as con:
        con.run("CREATE TABLE t AS SELECT 1 AS v")
        assert duckdb.frame.sql("SELECT v FROM t").rows(con) == [(1,)]
    assert (tmp_path / "by_path.db").exists()


def test_interrupt_cancels_a_running_query() -> None:
    con = _duckdb.Database(":memory:").connect()
    failure: list[BaseException] = []

    def run() -> None:
        try:
            # Large enough to still be running when the interrupt lands.
            con.execute("SELECT count(*) FROM range(100_000_000_000)").fetch_all()
        except BaseException as error:
            failure.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    # Interrupting before the query starts does nothing, and the test would hang instead of failing.
    time.sleep(0.5)
    con.interrupt()
    worker.join(timeout=30)

    assert not worker.is_alive(), "interrupt did not stop the query"
    assert failure, "interrupted query returned normally"
    assert isinstance(failure[0], exceptions.Error)


def test_empty_statement_with_parameters_is_refused() -> None:
    con = _duckdb.Database(":memory:").connect()
    with pytest.raises(exceptions.InvalidInputError, match="no statement"):
        con.execute("   ", [1])


def test_comment_only_statement_with_parameters_is_refused() -> None:
    con = _duckdb.Database(":memory:").connect()
    with pytest.raises(exceptions.InvalidInputError, match="no statement"):
        con.execute("-- nothing here", [1])


class TestOneEnvironment:
    """One shared Environment opens every database, so a second attempt to open the same file is noticed."""

    def test_opening_the_same_file_twice_is_refused(self, tmp_path: Path) -> None:
        path = str(tmp_path / "same.db")
        first = _duckdb.Database(path).connect()
        first.execute("CREATE TABLE t (v INTEGER)").drain()
        with pytest.raises(exceptions.Error, match=r"(?i)in use|conflict"):
            _duckdb.Database(path)

    def test_reopening_after_release_works(self, tmp_path: Path) -> None:
        # The guard is about concurrent use, not a permanent claim.
        path = str(tmp_path / "reopen.db")
        first = _duckdb.Database(path)
        first.connect().execute("CREATE TABLE t (v INTEGER)").drain()
        del first
        gc.collect()
        second = _duckdb.Database(path).connect()
        assert second.execute("SELECT count(*) FROM t").fetch_all() == [(0,)]

    def test_memory_databases_are_independent(self) -> None:
        # ":memory:" names no file, so each is its own database and the guard does not apply.
        one = _duckdb.Database(":memory:").connect()
        two = _duckdb.Database(":memory:").connect()
        one.execute("CREATE TABLE only_in_one (v INTEGER)").drain()
        with pytest.raises(exceptions.CatalogError):
            two.execute("SELECT * FROM only_in_one").fetch_all()

    def test_two_connections_to_one_database_see_each_other(self) -> None:
        database = _duckdb.Database(":memory:")
        writer, reader = database.connect(), database.connect()
        writer.execute("CREATE TABLE t (v INTEGER)").drain()
        writer.execute("INSERT INTO t VALUES (1)").drain()
        assert reader.execute("SELECT count(*) FROM t").fetch_all() == [(1,)]


class TestPublicInterrupt:
    """`Connection.interrupt()` from another thread, and Ctrl-C in this one."""

    def test_interrupt_cancels_from_another_thread(self) -> None:
        con = duckdb.frame.connect()
        # One interrupt at a fixed delay can land before the query starts and leave it running, so repeat.
        stop = threading.Event()

        def keep_interrupting() -> None:
            while not stop.is_set():
                con.interrupt()
                time.sleep(0.05)

        worker = threading.Thread(target=keep_interrupting)
        worker.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                duckdb.frame.sql("SELECT count(*) FROM range(100_000_000_000)").rows(con)
        finally:
            stop.set()
            worker.join()
        assert duckdb.frame.sql("SELECT 1").rows(con) == [(1,)]

    def test_interrupting_an_idle_connection_is_a_no_op(self) -> None:
        con = duckdb.frame.connect()
        con.interrupt()
        assert duckdb.frame.sql("SELECT 1").rows(con) == [(1,)]

    def test_interrupt_on_a_closed_connection_is_refused(self) -> None:
        con = duckdb.frame.connect()
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            con.interrupt()

    def test_a_keyboard_interrupt_stops_a_drain(self) -> None:
        # run() gives the interpreter back a few times a second to check for signals, so Ctrl-C lands promptly.
        con = duckdb.frame.connect()
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt), ctrl_c_after(0.3, con.interrupt):
            con.run("SELECT * FROM range(20_000_000_000)")
        assert time.monotonic() - started < 5, "the interrupt did not land between step batches"
        assert duckdb.frame.sql("SELECT 1").rows(con) == [(1,)]

    def test_a_keyboard_interrupt_stops_a_fetch(self) -> None:
        # Minutes of work for a few thousand rows: a fetch that never checks for Ctrl-C runs into the rescue and
        # fails the bound, where a quick query would finish inside it, and no rows pile up while it runs.
        # On the engine's connection, so the result is closed after the timing: closing a result whose query is
        # still running waits for the task running it unless the query is stopped first.
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(100_000_000_000) t(i) WHERE i % 50_000_000 = 0")
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt), ctrl_c_after(0.3, con.interrupt):
                result.fetch_all()
            assert time.monotonic() - started < 3, "the interrupt did not land while the fetch ran"
        finally:
            con.interrupt()
            result.close()
        assert con.execute("SELECT 1").fetch_all() == [(1,)]


def abort_inside(
    connection: duckdb.frame.Connection, work: Callable[[], object], error: type[BaseException] = RuntimeError
) -> None:
    """Run `work` in a transaction and then fail it, so rollback paths can be asserted."""
    with connection.transaction():
        work()
        message = "abort"
        raise error(message)


class TestTransaction:
    """`Connection.transaction()`: COMMIT on success, ROLLBACK on any error."""

    def test_commit_on_success(self) -> None:
        con = duckdb.frame.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with con.transaction():
            con.run("INSERT INTO t VALUES (1), (2)")
        assert duckdb.frame.table("t").count(con) == 2

    def test_rollback_on_error(self) -> None:
        con = duckdb.frame.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with pytest.raises(RuntimeError, match="abort"):
            abort_inside(con, lambda: con.run("INSERT INTO t VALUES (1)"))
        assert duckdb.frame.table("t").count(con) == 0

    def test_rollback_undoes_ddl(self) -> None:
        con = duckdb.frame.connect()
        with pytest.raises(RuntimeError):
            abort_inside(con, lambda: con.run("CREATE TABLE gone (v INTEGER)"))
        with pytest.raises(exceptions.CatalogError):
            duckdb.frame.table("gone").count(con)

    def test_a_keyboard_interrupt_rolls_back(self) -> None:
        # BaseException, not Exception: a Ctrl-C mid-block must not leave the transaction open.
        con = duckdb.frame.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with pytest.raises(KeyboardInterrupt):
            abort_inside(con, lambda: con.run("INSERT INTO t VALUES (1)"), error=KeyboardInterrupt)
        assert duckdb.frame.table("t").count(con) == 0

    def test_plans_share_the_transaction(self) -> None:
        con = duckdb.frame.connect()
        con.run("CREATE TABLE src AS SELECT 1 AS v")
        with pytest.raises(RuntimeError):
            abort_inside(con, lambda: duckdb.frame.table("src").create(con, "copy"))
        with pytest.raises(exceptions.CatalogError):
            duckdb.frame.table("copy").count(con)

    def test_uncommitted_work_is_invisible_to_a_sibling(self) -> None:
        con = duckdb.frame.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        sibling = con.duplicate()
        with con.transaction():
            con.run("INSERT INTO t VALUES (1)")
            assert duckdb.frame.table("t").count(sibling) == 0
        assert duckdb.frame.table("t").count(sibling) == 1

    def test_nesting_is_refused_in_the_engines_words(self) -> None:
        con = duckdb.frame.connect()
        refused = pytest.raises(exceptions.TransactionError, match="within a transaction")
        with refused, con.transaction(), con.transaction():
            pass
        assert duckdb.frame.sql("SELECT 1").rows(con) == [(1,)]

    def test_a_failed_rollback_does_not_hide_the_error(self) -> None:
        # Closing mid-block makes the rollback itself fail; the block's own error must still be the one that surfaces.
        con = duckdb.frame.connect()
        with pytest.raises(RuntimeError, match="abort"):
            abort_inside(con, con.close)


#: Every call that uses the connection beyond parsing and binding, as source over `con`; the parameter and the
#: default are values that convert through the connection, and the DECIMAL type is one its parser reads.
CONNECTION_CALLS = {
    "execute": "con.execute('SELECT 1')",
    "execute with a parameter": "con.execute('SELECT $1', [decimal.Decimal('1.5')])",
    "execute_arrow": "con.execute_arrow('SELECT 1')",
    "create_scalar_function": (
        "con.create_scalar_function('g', abs, [('x', _duckdb.ParameterKind.STANDARD, 'DECIMAL(4, 1)', True, "
        "decimal.Decimal('1.5'))], 'DECIMAL(4, 1)', _duckdb.FunctionNullHandling.DEFAULT, "
        "_duckdb.FunctionStability.CONSISTENT)"
    ),
    "register_object": "con.register_object('o', object(), False, False)",
    "unregister_object": "con.unregister_object('o')",
    "get_option": "con.get_option('threads')",
    "set_option": "con.set_option('threads', '2')",
}

#: Calls that parse or bind, which the engine answers beside an open result.
ANSWERED_WHILE_HELD = {"classify": "con.classify('SELECT 1')", "bind": "con.bind('SELECT 1')"}

#: Every call; none may use a connection whose query is running.
RUNNING_CALLS = {**CONNECTION_CALLS, **ANSWERED_WHILE_HELD}

#: What a call on a connection whose query is running is refused with, wherever the call comes from.
RUNNING = "Resource In Use Error: connection has a live result; it cannot be used while its query runs"


def use(con: _duckdb.Connection, call: str) -> None:
    exec(RUNNING_CALLS[call], {"con": con, "decimal": decimal, "_duckdb": _duckdb})


def run_alone(script: str) -> str:
    """Run `script` in a child process and return what it printed; a call that waits on the engine fails the test."""
    try:
        done = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)], capture_output=True, text=True, check=False, timeout=20
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the call waited on the connection instead of being refused")
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


class TestALiveResultHoldsTheConnection:
    """A result holds its connection until it is read to its end, fails, or is closed, as the engine's one does."""

    @pytest.mark.parametrize("call", list(CONNECTION_CALLS))
    def test_every_use_is_refused_until_the_result_is_closed(self, call: str) -> None:
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(100000) t(i)")
        result.fetch_rows(1)
        with pytest.raises(exceptions.OperationalError, match="live result; finish reading it or close it"):
            use(con, call)
        result.close()
        use(con, call)

    def test_parsing_is_still_answered(self) -> None:
        # The DB-API decides how to dispose of an open result by what the next statement is.
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(100000) t(i)")
        result.fetch_rows(1)
        assert con.classify("ROLLBACK") == ("transaction", "ROLLBACK")
        assert len(result.fetch_all()) == 99999

    def test_binding_is_still_answered(self) -> None:
        # The engine binds beside an open result, which is how a plan learns its columns next to an open reader.
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(100000) t(i)")
        result.fetch_rows(1)
        assert con.bind("SELECT $1::INTEGER AS a") == ([("a", "INTEGER")], [("1", "INTEGER")])
        assert len(result.fetch_all()) == 99999

    def test_a_result_read_to_its_end_holds_nothing(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(5000) t(i)")
        assert len(result.fetch_all()) == 5000
        assert con.execute("SELECT 42").fetch_all() == [(42,)]

    def test_a_drained_result_holds_nothing(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("CREATE TABLE t AS SELECT 1 AS a")
        result.drain()
        assert con.execute("SELECT a FROM t").fetch_all() == [(1,)]

    def test_a_result_that_failed_holds_nothing(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT CASE WHEN i = 4000 THEN error('late') ELSE i END FROM range(5000) t(i)")
        with pytest.raises(exceptions.InvalidInputError, match="late"):
            result.fetch_all()
        assert con.execute("SELECT 42").fetch_all() == [(42,)]

    def test_a_cancelled_result_holds_nothing(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT count(*) FROM range(100_000_000_000)")
        stop = threading.Event()

        def keep_interrupting() -> None:
            while not stop.is_set():
                con.interrupt()
                time.sleep(0.05)

        worker = threading.Thread(target=keep_interrupting)
        worker.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                result.fetch_all()
        finally:
            stop.set()
            worker.join()
        assert con.execute("SELECT 42").fetch_all() == [(42,)]

    def test_a_result_collected_unread_holds_nothing(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        con.execute("SELECT i FROM range(100000) t(i)").fetch_rows(1)
        gc.collect()
        assert con.execute("SELECT 42").fetch_all() == [(42,)]

    def test_an_open_arrow_stream_holds_the_connection(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        stream = con.execute_arrow("SELECT i FROM range(100000) t(i)")
        with pytest.raises(exceptions.OperationalError, match="live result; finish reading it or close it"):
            con.execute("SELECT 1")
        stream.close()

    def test_a_closed_arrow_stream_holds_nothing(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        stream = con.execute_arrow("SELECT i FROM range(100000) t(i)")
        stream.close()
        assert con.execute("SELECT 42").fetch_all() == [(42,)]

    def test_interrupt_and_close_are_never_refused(self) -> None:
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(100000) t(i)")
        result.fetch_rows(1)
        con.interrupt()
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            con.execute("SELECT 1")
        result.close()


REENTRY = """
import decimal
from duckdb import _duckdb, exceptions
con = _duckdb.Database(':memory:').connect()

def inner(x):
    try:
        {call}
    except exceptions.OperationalError as error:
        return 'refused: ' + str(error)
    return 'ran'

con.create_scalar_function(
    'inner', inner, [('x', _duckdb.ParameterKind.STANDARD, 'BIGINT', False, None)], 'VARCHAR',
    _duckdb.FunctionNullHandling.DEFAULT, _duckdb.FunctionStability.CONSISTENT)
print(con.execute({query!r}).fetch_all()[0][0])
"""

CROSS_THREAD = """
import decimal
import threading
from duckdb import _duckdb, exceptions
con = _duckdb.Database(':memory:').connect()
inside, done = threading.Event(), threading.Event()

def gate(x):
    inside.set()
    done.wait(30)
    return x

con.create_scalar_function(
    'gate', gate, [('x', _duckdb.ParameterKind.STANDARD, 'BIGINT', False, None)], 'BIGINT',
    _duckdb.FunctionNullHandling.DEFAULT, _duckdb.FunctionStability.VOLATILE)
reader = threading.Thread(target=lambda: con.execute('SELECT gate(i) FROM range(1) t(i)').fetch_all())
reader.start()
inside.wait(30)
try:
    {call}
    print('ran')
except exceptions.OperationalError as error:
    print('refused: ' + str(error))
finally:
    done.set()
    reader.join()
"""


ON_THE_WORKERS = """
import decimal
import threading
from duckdb import _duckdb, exceptions
con = _duckdb.Database(':memory:').connect()
con.execute('SET threads = 4').drain()
seen = []
ready = threading.Event()

def probe(x):
    if not ready.is_set():
        try:
            {call}
            seen.append('ran')
        except exceptions.OperationalError as error:
            seen.append('refused: ' + str(error))
        ready.set()
    return x

con.create_scalar_function(
    'probe', probe, [('x', _duckdb.ParameterKind.STANDARD, 'BIGINT', False, None)], 'BIGINT',
    _duckdb.FunctionNullHandling.DEFAULT, _duckdb.FunctionStability.VOLATILE)
result = con.execute('SELECT sum(probe(i)) FROM range(2000000) t(i)')
ready.wait(15)
print(seen[0] if seen else 'never called before the read')
con.interrupt()
result.close()
"""


class TestARunningQueryHoldsTheConnection:
    """A call on a connection whose query is running is refused at once, never left to wait for that query."""

    @pytest.mark.parametrize("call", list(RUNNING_CALLS))
    @pytest.mark.parametrize(
        "query", ["SELECT inner(1)", "SELECT inner(i) FROM range(1) t(i)"], ids=["while planning", "while reading"]
    )
    def test_a_python_function_inside_the_query_is_refused(self, call: str, query: str) -> None:
        printed = run_alone(REENTRY.format(call=RUNNING_CALLS[call], query=query))
        assert printed.startswith(f"refused: {RUNNING}"), printed

    @pytest.mark.parametrize("call", list(RUNNING_CALLS))
    def test_another_thread_is_refused(self, call: str) -> None:
        printed = run_alone(CROSS_THREAD.format(call=RUNNING_CALLS[call]))
        assert printed.startswith(f"refused: {RUNNING}"), printed

    @pytest.mark.parametrize("call", list(RUNNING_CALLS))
    def test_a_python_function_running_ahead_of_any_read_is_refused(self, call: str) -> None:
        # The workers run the query before anything reads it, with no call of ours in progress.
        printed = run_alone(ON_THE_WORKERS.format(call=RUNNING_CALLS[call]))
        assert printed.startswith(f"refused: {RUNNING}"), printed

    def test_a_connection_handed_between_threads_is_used_one_at_a_time(self) -> None:
        # Only use at the same time is refused: a result opened here can be read elsewhere, and the connection
        # used there and back here, each call starting after the one before it ended.
        con = _duckdb.Database(":memory:").connect()
        result = con.execute("SELECT i FROM range(5000) t(i)")
        outcomes: list[object] = []

        def elsewhere() -> None:
            try:
                outcomes.append(len(result.fetch_all()))
                outcomes.append(con.execute("SELECT 42").fetch_all())
                outcomes.append(con.bind("SELECT 1 AS a")[0])
                con.register_object("o", object(), False, False)
                outcomes.append(con.unregister_object("o"))
            except BaseException as error:
                outcomes.append(error)

        worker = threading.Thread(target=elsewhere)
        worker.start()
        worker.join(timeout=30)
        assert not worker.is_alive()
        assert outcomes == [5000, [(42,)], [("a", "INTEGER")], True]
        assert con.execute("SELECT 7").fetch_all() == [(7,)]
