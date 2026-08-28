"""Cross-thread use of a shared connection.

The engine allows one live result per connection and cursors share one, so
these paths are reachable through the documented API rather than by sharing a
Result object by hand.
"""

from __future__ import annotations

import contextlib
import threading

import pytest

from duckdb import _duckdb, dbapi, exceptions

# Big enough that the reader is still inside the engine when the other thread
# acts, small enough not to slow the suite down.
LONG_QUERY = "SELECT i FROM range(4_000_000) t(i)"
ATTEMPTS = 25


def test_closing_a_result_while_another_thread_reads_it() -> None:
    """Closing must not free the result out from under a stepping thread.

    Both `close()` and the fetch path release the GIL, so the GIL provides no
    mutual exclusion. The result is held by shared_ptr and the reader takes its
    own reference before dropping the GIL, so a concurrent close cannot free it
    underneath.

    Honest limitation: this is a smoke test, not a demonstration of the
    original defect. The dangling reference could not be made to fault on
    macOS, including under AddressSanitizer, because destroying the engine
    result blocks until the engine settles and that masks the window. It is
    here to catch a regression that crashes or hangs.
    """
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

        assert not reader.is_alive(), "reader hung after close"
        assert not failures, f"unexpected failure: {failures[0]!r}"


def test_sibling_cursor_execute_while_a_cursor_is_reading() -> None:
    """The review's scenario: two cursors, one connection, one result slot.

    Thread A is inside fetchall(); thread B executes on a sibling cursor, which
    releases A's result. That must fail cleanly, never crash.
    """
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
        # The engine may still consider the first result live; either outcome
        # is fine, the point is that nothing crashes.
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
