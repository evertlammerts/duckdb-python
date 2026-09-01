"""The connection surface: options, interruption, and statement handling."""

from __future__ import annotations

import _thread
import gc
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import duckdb
from duckdb import _duckdb, exceptions


def test_settings_round_trip() -> None:
    con = _duckdb.Database(":memory:").connect()
    con.set_option("threads", "3")
    assert con.get_option("threads") == "3"


def test_unknown_setting_is_reported() -> None:
    con = _duckdb.Database(":memory:").connect()
    with pytest.raises(exceptions.Error):
        con.get_option("no_such_setting_at_all")


def test_database_accepts_open_time_options() -> None:
    # Some settings can only be chosen before the database exists, so the
    # options path is not interchangeable with set_option afterwards.
    con = _duckdb.Database(":memory:", [("threads", "2")]).connect()
    assert con.get_option("threads") == "2"


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
    # Give the query time to actually start; interrupting before it does is a
    # no-op and the test would hang rather than fail.
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
    """All databases are opened through a single Environment.

    The Environment exists to notice a second attempt to open a database
    already open in this process. Giving each Database its own would remove
    that guard silently, which is what an earlier version did.
    """

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
        # ":memory:" names no file, so each is its own database and the guard
        # does not apply.
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
        con = duckdb.connect()
        # Interrupt repeatedly until the query dies: one shot at a fixed delay
        # can fire before execution starts, land on nothing, and leave the
        # query running unbounded.
        stop = threading.Event()

        def keep_interrupting() -> None:
            while not stop.is_set():
                con.interrupt()
                time.sleep(0.05)

        worker = threading.Thread(target=keep_interrupting)
        worker.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                duckdb.sql("SELECT count(*) FROM range(100_000_000_000)").rows(con)
        finally:
            stop.set()
            worker.join()
        assert duckdb.sql("SELECT 1").rows(con) == [(1,)]

    def test_interrupting_an_idle_connection_is_a_no_op(self) -> None:
        con = duckdb.connect()
        con.interrupt()
        assert duckdb.sql("SELECT 1").rows(con) == [(1,)]

    def test_interrupt_on_a_closed_connection_is_refused(self) -> None:
        con = duckdb.connect()
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            con.interrupt()

    @pytest.mark.skipif(sys.platform == "win32", reason="simulated Ctrl-C delivery differs on Windows")
    def test_a_keyboard_interrupt_stops_a_drain(self) -> None:
        # run() steps the result with the GIL released, retaking it a few
        # times a second for the signal check, so the Ctrl-C lands promptly
        # however long the statement would run.
        con = duckdb.connect()
        interrupter = threading.Timer(0.3, _thread.interrupt_main)
        # The rescue bounds a lost Ctrl-C: the query dies as InterruptError,
        # failing this test promptly instead of hanging the whole suite.
        rescue = threading.Timer(10, con.interrupt)
        interrupter.start()
        rescue.start()
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                con.run("SELECT * FROM range(20_000_000_000)")
        finally:
            interrupter.cancel()
            rescue.cancel()
        assert time.monotonic() - started < 5, "the interrupt did not land between step batches"
        assert duckdb.sql("SELECT 1").rows(con) == [(1,)]

    @pytest.mark.skipif(sys.platform == "win32", reason="simulated Ctrl-C delivery differs on Windows")
    def test_a_keyboard_interrupt_stops_a_streaming_fetch(self) -> None:
        # Chunks flow and the signal check runs once per chunk; a query whose
        # one chunk arrives only at the end waits on the engine's own step
        # granularity instead, measured at seconds, not tested here.
        con = duckdb.connect()
        interrupter = threading.Timer(0.3, _thread.interrupt_main)
        # Same rescue as the drain test: a lost Ctrl-C fails fast, never hangs.
        rescue = threading.Timer(10, con.interrupt)
        interrupter.start()
        rescue.start()
        started = time.monotonic()
        try:
            with pytest.raises(KeyboardInterrupt):
                duckdb.sql("SELECT i, md5(i::VARCHAR) FROM range(8_000_000) t(i)").rows(con)
        finally:
            interrupter.cancel()
            rescue.cancel()
        assert time.monotonic() - started < 3, "the interrupt did not land between chunks"
        assert duckdb.sql("SELECT 1").rows(con) == [(1,)]


def abort_inside(
    connection: duckdb.Connection, work: Callable[[], object], error: type[BaseException] = RuntimeError
) -> None:
    """Run `work` in a transaction and then fail it, so rollback paths can be asserted."""
    with connection.transaction():
        work()
        message = "abort"
        raise error(message)


class TestTransaction:
    """`Connection.transaction()`: COMMIT on success, ROLLBACK on any error."""

    def test_commit_on_success(self) -> None:
        con = duckdb.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with con.transaction():
            con.run("INSERT INTO t VALUES (1), (2)")
        assert duckdb.table("t").count(con) == 2

    def test_rollback_on_error(self) -> None:
        con = duckdb.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with pytest.raises(RuntimeError, match="abort"):
            abort_inside(con, lambda: con.run("INSERT INTO t VALUES (1)"))
        assert duckdb.table("t").count(con) == 0

    def test_rollback_undoes_ddl(self) -> None:
        con = duckdb.connect()
        with pytest.raises(RuntimeError):
            abort_inside(con, lambda: con.run("CREATE TABLE gone (v INTEGER)"))
        with pytest.raises(exceptions.CatalogError):
            duckdb.table("gone").count(con)

    def test_a_keyboard_interrupt_rolls_back(self) -> None:
        # BaseException, not Exception: a Ctrl-C mid-block must not leave the
        # transaction open.
        con = duckdb.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with pytest.raises(KeyboardInterrupt):
            abort_inside(con, lambda: con.run("INSERT INTO t VALUES (1)"), error=KeyboardInterrupt)
        assert duckdb.table("t").count(con) == 0

    def test_plans_share_the_transaction(self) -> None:
        con = duckdb.connect()
        con.run("CREATE TABLE src AS SELECT 1 AS v")
        with pytest.raises(RuntimeError):
            abort_inside(con, lambda: duckdb.table("src").create(con, "copy"))
        with pytest.raises(exceptions.CatalogError):
            duckdb.table("copy").count(con)

    def test_uncommitted_work_is_invisible_to_a_sibling(self) -> None:
        con = duckdb.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        sibling = con.duplicate()
        with con.transaction():
            con.run("INSERT INTO t VALUES (1)")
            assert duckdb.table("t").count(sibling) == 0
        assert duckdb.table("t").count(sibling) == 1

    def test_nesting_is_refused_in_the_engines_words(self) -> None:
        con = duckdb.connect()
        refused = pytest.raises(exceptions.TransactionError, match="within a transaction")
        with refused, con.transaction(), con.transaction():
            pass
        assert duckdb.sql("SELECT 1").rows(con) == [(1,)]

    def test_a_failed_rollback_does_not_hide_the_error(self) -> None:
        # Closing mid-block makes the rollback itself fail; the block's own
        # error must still be the one that surfaces.
        con = duckdb.connect()
        with pytest.raises(RuntimeError, match="abort"):
            abort_inside(con, con.close)
