import logging

import duckdb


def test_warning_routed_to_python_logging(caplog):
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        # Pin lambda_syntax to DEFAULT so the deprecated arrow (->) form reliably emits a
        # DUCKDB_LOG_WARNING. DEFAULT is the current engine default, but it is explicitly
        # slated to change ("before DuckDB's next release"); pinning keeps this test
        # exercising the warning path across future submodule bumps.
        con.execute("SET lambda_syntax='DEFAULT'")
        con.execute("SELECT list_transform([1, 2, 3], x -> x + 1)")
    deprecation_records = [r for r in caplog.records if "deprecated" in r.message.lower()]
    assert deprecation_records, "expected a deprecation warning routed to the 'duckdb' logger"
    assert all(r.name == "duckdb" for r in deprecation_records)
    assert all(r.levelno == logging.WARNING for r in deprecation_records)


def test_warning_not_emitted_for_clean_queries(caplog):
    with caplog.at_level(logging.WARNING, logger="duckdb"):
        con = duckdb.connect()
        con.execute("SELECT 1 + 1").fetchone()
    # Assert the absence of the deprecation warning specifically rather than requiring zero
    # records total — an incidental connect-time warning (e.g. the macOS Rosetta notice on
    # some hardware) would otherwise make this flaky.
    assert not [r for r in caplog.records if "deprecated" in r.message.lower()]
