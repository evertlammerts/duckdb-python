"""The connection surface: options, interruption, and statement handling."""

from __future__ import annotations

import threading
import time

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
