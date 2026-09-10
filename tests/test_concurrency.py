"""Cross-thread use of one connection, whose single live result its cursors share."""

from __future__ import annotations

import contextlib
import threading

import pytest

from duckdb import _duckdb, dbapi, exceptions

# Big enough that the reader is still inside DuckDB when the other thread acts, small enough to stay quick.
LONG_QUERY = "SELECT i FROM range(4_000_000) t(i)"
ATTEMPTS = 25


def test_closing_a_result_while_another_thread_reads_it() -> None:
    """Closing must not free the result while another thread reads it; each side holds its own reference."""
    for _ in range(ATTEMPTS):
        con = _duckdb.Database(":memory:").connect()
        result = con.execute(LONG_QUERY)
        failures: list[BaseException] = []

        def read(result: _duckdb.Result = result, failures: list[BaseException] = failures) -> None:
            try:
                result.fetch_all()
            except exceptions.Error:
                pass  # a closed result is a fine outcome
            except BaseException as error:
                failures.append(error)

        reader = threading.Thread(target=read)
        reader.start()
        result.close()
        reader.join(timeout=60)

        # Only a crash or a hang can be caught here: the dangling read could not be made to fault on macOS.
        assert not reader.is_alive(), "reader hung after close"
        assert not failures, f"unexpected failure: {failures[0]!r}"


def test_sibling_cursor_execute_while_a_cursor_is_reading() -> None:
    """One thread is inside fetchall() when another executes on a sibling cursor and releases its result."""
    for _ in range(ATTEMPTS):
        con = dbapi.connect()
        reader_cursor, writer_cursor = con.cursor(), con.cursor()
        reader_cursor.execute(LONG_QUERY)
        failures: list[BaseException] = []

        def read(cursor: dbapi.Cursor = reader_cursor, failures: list[BaseException] = failures) -> None:
            try:
                cursor.fetchall()
            except (exceptions.Error, dbapi.InterfaceError):
                pass
            except BaseException as error:
                failures.append(error)

        reader = threading.Thread(target=read)
        reader.start()
        # Either outcome is fine; the point is that nothing crashes.
        with contextlib.suppress(exceptions.Error):
            writer_cursor.execute("SELECT 1")
        reader.join(timeout=60)

        assert not reader.is_alive(), "reader hung"
        assert not failures, f"unexpected failure: {failures[0]!r}"
        con.close()


def test_using_a_closed_result_reports_rather_than_crashes() -> None:
    con = _duckdb.Database(":memory:").connect()
    result = con.execute("SELECT i FROM range(10) t(i)")
    result.close()
    with pytest.raises(exceptions.InterfaceError, match="closed"):
        result.fetch_rows(1)
    with pytest.raises(exceptions.InterfaceError, match="closed"):
        result.fetch_all()
