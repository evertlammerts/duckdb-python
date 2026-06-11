"""Tests for the PythonLogStorage composite log storage (issue #480).

PythonLogStorage forwards engine log entries to Python's `logging` module AND keeps them
queryable via `SELECT * FROM duckdb_logs`. It is registered on the first connection to each
DatabaseInstance and routes WARNING+ entries to logging.getLogger("duckdb").

Forwarding to `logging` is ASYNCHRONOUS (a background thread drains a queue), because the engine
calls the log-write path while holding LogManager::lock and acquiring the GIL there would
deadlock. So any assertion about the `logging` channel must first call `_drain()` to wait for the
forwarder to catch up. The `duckdb_logs` table channel is synchronous and needs no drain.
"""

import logging

import _duckdb
import pytest

import duckdb

DEPRECATION_FRAGMENT = "Deprecated lambda arrow"


def _drain():
    """Block until the async forwarder has delivered every queued entry to `logging`."""
    _duckdb._drain_log_forwarding()


def _trigger_deprecation_warning(con):
    """Run a query that reliably emits a single engine DUCKDB_LOG_WARNING, then drain.

    The deprecated arrow (->) lambda form warns only when lambda_syntax is DEFAULT. DEFAULT
    is the current engine default but is slated to change, so we pin it to keep this exercising
    the warning path across submodule bumps.

    We drain the async forwarder before returning so the entry is delivered to `logging` while
    the caller's caplog handler is still attached — callers may read records after their
    `with caplog` block exits.
    """
    con.execute("SET lambda_syntax='DEFAULT'")
    con.execute("SELECT list_transform([1, 2, 3], x -> x + 1)").fetchall()
    _drain()


def _deprecation_records(caplog):
    return [r for r in caplog.records if "deprecated" in r.getMessage().lower()]


def _duckdb_logs_deprecation_count(con):
    return con.execute(f"SELECT count(*) FROM duckdb_logs WHERE message LIKE '%{DEPRECATION_FRAGMENT}%'").fetchone()[0]


# ---------------------------------------------------------------------------
# Channel 1: forwarding to Python's logging module
# ---------------------------------------------------------------------------


def test_warning_routed_to_python_logging(caplog):
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        _trigger_deprecation_warning(con)
    records = _deprecation_records(caplog)
    assert records, "expected a deprecation warning routed to the 'duckdb' logger"
    assert all(r.name == "duckdb" for r in records)
    assert all(r.levelno == logging.WARNING for r in records)


def test_warning_not_emitted_for_clean_queries(caplog):
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        con.execute("SELECT 1 + 1").fetchone()
    # Assert absence of the deprecation warning specifically rather than requiring zero records
    # total — an incidental connect-time warning (e.g. the macOS Rosetta notice on some
    # hardware) would otherwise make this flaky.
    assert not _deprecation_records(caplog)


def test_module_level_default_connection_forwards(caplog):
    # The most common Jupyter/script pattern: no explicit connect(). The default connection is
    # created lazily via Connect(), so it must register the storage too.
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        duckdb.execute("SET lambda_syntax='DEFAULT'")
        duckdb.sql("SELECT list_transform([1, 2, 3], x -> x + 1)").fetchall()
        _drain()  # this test triggers directly rather than via _trigger_deprecation_warning
    assert _deprecation_records(caplog), "default connection should route warnings to logging"


# ---------------------------------------------------------------------------
# Channel 2: duckdb_logs stays queryable (the regression this design fixes)
# ---------------------------------------------------------------------------


def test_duckdb_logs_still_populated():
    con = duckdb.connect()
    _trigger_deprecation_warning(con)
    assert _duckdb_logs_deprecation_count(con) >= 1, "SELECT * FROM duckdb_logs must still surface engine warnings"


def test_single_warning_visible_immediately():
    # Guards against regressing to a batched (buffer_size=2048) storage where a lone warning
    # would never flush. With buffer_size=1 it must appear after a single triggering query.
    con = duckdb.connect()
    assert _duckdb_logs_deprecation_count(con) == 0
    _trigger_deprecation_warning(con)
    assert _duckdb_logs_deprecation_count(con) >= 1


def test_duckdb_logs_schema_and_content():
    con = duckdb.connect()
    _trigger_deprecation_warning(con)
    row = con.execute(
        "SELECT log_level, message, type, timestamp "
        f"FROM duckdb_logs WHERE message LIKE '%{DEPRECATION_FRAGMENT}%' LIMIT 1"
    ).fetchone()
    assert row is not None
    log_level, message, log_type, timestamp = row
    assert log_level == "WARNING"
    assert DEPRECATION_FRAGMENT in message
    # `type` is a VARCHAR but may be empty for some engine warnings (the deprecation notice
    # carries no log type), so only assert the schema, not non-emptiness.
    assert isinstance(log_type, str)
    assert timestamp is not None


# ---------------------------------------------------------------------------
# Both channels together
# ---------------------------------------------------------------------------


def test_both_channels_receive_the_same_entry(caplog):
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        _trigger_deprecation_warning(con)
        table_rows = con.execute(
            f"SELECT message FROM duckdb_logs WHERE message LIKE '%{DEPRECATION_FRAGMENT}%'"
        ).fetchall()
    logging_records = _deprecation_records(caplog)
    assert logging_records, "logging channel missing the entry"
    assert table_rows, "duckdb_logs channel missing the entry"
    # The message content must agree across both channels.
    assert any(DEPRECATION_FRAGMENT in r.getMessage() for r in logging_records)
    assert all(DEPRECATION_FRAGMENT in row[0] for row in table_rows)


def test_repeated_warnings_accumulate_in_both_channels(caplog):
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        _trigger_deprecation_warning(con)
        after_first = _duckdb_logs_deprecation_count(con)
        records_after_first = len(_deprecation_records(caplog))
        _trigger_deprecation_warning(con)
        after_second = _duckdb_logs_deprecation_count(con)
        records_after_second = len(_deprecation_records(caplog))
    # No deduplication: a second occurrence is recorded again in both channels.
    assert after_second > after_first
    assert records_after_second > records_after_first


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_raising_handler_does_not_fail_query_and_row_persists():
    # A user logging handler that raises must not disrupt anything. Forwarding runs on a
    # background thread (decoupled from the query path), so the exception is swallowed there by
    # the forwarder's catch(...) — the query can never see it. The entry is also stored BEFORE
    # being queued, so the duckdb_logs row must still be present. We drain while the handler is
    # attached so it actually fires and exercises the C++ exception safety net.
    class BoomHandler(logging.Handler):
        def emit(self, record):
            # Intentionally raise (bare raise keeps ruff's EM101/TRY003 happy).
            raise RuntimeError

    logger = logging.getLogger("duckdb")
    handler = BoomHandler()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        con = duckdb.connect()
        con.execute("SET lambda_syntax='DEFAULT'")
        result = con.execute("SELECT list_transform([1, 2, 3], x -> x + 1)").fetchall()
        _drain()  # force the raising handler to fire on the forwarder thread
        assert result == [([2, 3, 4],)]
        assert _duckdb_logs_deprecation_count(con) >= 1
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@pytest.mark.timeout(60)
def test_concurrent_warning_queries_do_not_deadlock():
    # Regression guard. Forwarding used to acquire the GIL from inside FlushChunk, which runs
    # under LogManager::lock. With two threads each running a warning-emitting query, one thread
    # would hold that lock and block on the GIL while another held the GIL and blocked on the
    # lock (via LogManager::CreateLogger) — a hard deadlock. Forwarding is now async, so this
    # must complete quickly. pytest-timeout (configured for the suite) fails the test if it hangs.
    from concurrent.futures import ThreadPoolExecutor

    def hammer(con):
        cur = con.cursor()
        cur.execute("SET lambda_syntax='DEFAULT'")
        for _ in range(20):
            cur.execute("SELECT list_transform([1, 2, 3], x -> x + 1)").fetchall()

    con = duckdb.connect()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(hammer, con) for _ in range(4)]
        for future in futures:
            future.result()
    _drain()


def test_default_storage_configuration():
    con = duckdb.connect()
    assert con.execute("SELECT current_setting('logging_storage')").fetchone()[0] == "python_log_storage"
    assert con.execute("SELECT current_setting('enable_logging')").fetchone()[0]
    assert con.execute("SELECT current_setting('logging_level')").fetchone()[0] == "WARNING"


def test_switching_to_memory_storage_disables_forwarding(caplog):
    # The user escape hatch: SET logging_storage='memory' detaches our storage. Forwarding to
    # logging stops, but the table path still works (now via the engine's in-memory storage).
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        con.execute("SET logging_storage='memory'")
        _trigger_deprecation_warning(con)
        table_count = _duckdb_logs_deprecation_count(con)
    assert table_count >= 1, "memory storage should still populate duckdb_logs"
    assert not _deprecation_records(caplog), "forwarding should stop once our storage is detached"


def test_separate_databases_are_independent(caplog):
    # Logging is per-DatabaseInstance; each fresh database registers its own storage and keeps
    # its own duckdb_logs, while both forward to the shared process-wide 'duckdb' logger.
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con_a = duckdb.connect()
        con_b = duckdb.connect()
        _trigger_deprecation_warning(con_a)
        assert _duckdb_logs_deprecation_count(con_a) >= 1
        assert _duckdb_logs_deprecation_count(con_b) == 0, "con_b has its own, untouched storage"
        _trigger_deprecation_warning(con_b)
        assert _duckdb_logs_deprecation_count(con_b) >= 1
    assert len(_deprecation_records(caplog)) >= 2


def test_cursor_shares_storage(caplog):
    # A cursor shares the parent's DatabaseInstance, so warnings it triggers land in the same
    # duckdb_logs and route to logging.
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        cur = con.cursor()
        _trigger_deprecation_warning(cur)
        assert _duckdb_logs_deprecation_count(con) >= 1
        assert _duckdb_logs_deprecation_count(cur) >= 1
    assert _deprecation_records(caplog)
