"""The connection surface: options, interruption, and statement handling."""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

import pytest

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
