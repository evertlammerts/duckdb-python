"""How this package covers each feature DuckDB's own tests cover, and a check that neither side drifts."""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest

#: A method reaches this feature and a named test class proves it.
VERB_TESTED = "verb-tested"
#: A method reaches this feature and nothing here proves it yet.
VERB_UNTESTED = "verb-untested"
#: Reachable only inside an expression, through sql_expr().
SQL_EXPR = "sql_expr"
#: A statement rather than a query, so sql() and run() carry it and no method builds it.
BRIDGE = "bridge"
#: DuckDB's own behaviour, not something this package presents as an API.
ENGINE = "engine"

#: Directory -> (kind, reason, proof), where the proof names a test class and only a verb-tested entry has one.
FEATURES: dict[str, tuple[str, str, str | None]] = {
    "aggregate": (VERB_TESTED, "aggregate(), group_by().agg(), the generated methods", "test_frame.TestAggregation"),
    "alter": (BRIDGE, "ALTER is DDL; run() carries it", None),
    "aoc24": (ENGINE, "an advent-of-code query workload exercising the engine, not a SQL surface", None),
    "append": (ENGINE, "the C++ appender, not a SQL surface", None),
    "attach": (BRIDGE, "ATTACH is a statement; run() carries it", None),
    "binder": (ENGINE, "name resolution rules; the layer defers to the binder by design", None),
    "cast": (VERB_TESTED, "Expr.cast()", "test_expr.TestPredicatesAndFunctions"),
    "catalog": (BRIDGE, "catalog DDL and lookups; run() and sql() carry them", None),
    "collate": (SQL_EXPR, "COLLATE is expression syntax the verbs do not spell", None),
    "conjunction": (VERB_TESTED, "& | ~ on expressions", "test_expr.TestOperatorsEvaluate"),
    "connect": (ENGINE, "connection strings and modes; connect() passes them through", None),
    "constraints": (BRIDGE, "constraints are DDL", None),
    "copy": (
        VERB_TESTED,
        "copy_to() and to_parquet/to_csv/to_json are COPY TO; COPY FROM is a statement",
        "test_frame.TestCopyIsSql",
    ),
    "copy_database": (BRIDGE, "COPY FROM DATABASE is a statement", None),
    "create": (VERB_TESTED, "create() renders CREATE TABLE AS", "test_frame.TestSinks"),
    "cte": (VERB_TESTED, "every step is a CTE; the graph renders one per step", "test_frame.TestGraph"),
    "delete": (BRIDGE, "DELETE is a statement, not a query; run() carries it", None),
    "detailed_profiler": (ENGINE, "profiler output format", None),
    "error": (ENGINE, "error message wording; the layer never rewrites engine messages", None),
    "explain": (VERB_TESTED, "explain()", "test_frame.TestInspection"),
    "export": (BRIDGE, "EXPORT DATABASE is a statement", None),
    "extensions": (ENGINE, "extension loading; the bundle decides what is built in", None),
    "filter": (VERB_TESTED, "filter()", "test_frame.TestRows"),
    "function": (
        VERB_TESTED,
        ".str()/.dt()/.list()/.json() and fn(), spec-tested against the catalog",
        "test_frame.TestFunctionNamespaces",
    ),
    "generated_columns": (BRIDGE, "generated columns are DDL", None),
    "index": (BRIDGE, "indexes are DDL", None),
    "insert": (
        VERB_TESTED,
        "insert_into() renders INSERT ... SELECT; INSERT VALUES is a statement",
        "test_frame.TestSinks",
    ),
    "join": (VERB_TESTED, "join() with every kind in _JOIN_KINDS", "test_frame.TestJoins"),
    "json": (VERB_TESTED, "the .json() function namespace", "test_frame.TestFunctionNamespaces"),
    "multi_file": (BRIDGE, "file lists with per-file open options; the file readers take the list", None),
    "keywords": (ENGINE, "parser keyword handling; identifiers are always quoted here", None),
    "limit": (VERB_TESTED, "limit(), head(), offset()", "test_frame.TestRows"),
    "logging": (ENGINE, "engine logging", None),
    "merge": (BRIDGE, "MERGE INTO is a statement, not a query; run() carries it", None),
    "optimizer": (ENGINE, "no client-side optimizer by design", None),
    "order": (VERB_TESTED, "sort() with direction and nulls placement", "test_frame.TestRows"),
    "ordinality": (BRIDGE, "WITH ORDINALITY is FROM-clause syntax", None),
    "outofcore": (ENGINE, "spilling", None),
    "overflow": (ENGINE, "arithmetic overflow semantics", None),
    "parallelism": (ENGINE, "thread scheduling", None),
    "parser": (ENGINE, "parser behaviour", None),
    "peg_parser": (ENGINE, "the alternative parser", None),
    "pg_catalog": (BRIDGE, "pg_* views are read through sql()", None),
    "pivot": (
        BRIDGE,
        "PIVOT cannot be bound in advance nor see a sibling CTE, so no verb builds on it; sql() carries it",
        None,
    ),
    "pragma": (BRIDGE, "PRAGMA is a statement", None),
    "prepared": (VERB_TESTED, "param() and parameters=", "test_frame.TestParametersAreSupplied"),
    "progress_bar": (ENGINE, "progress reporting", None),
    "projection": (VERB_TESTED, "select(), with_columns(), star()", "test_frame.TestProjection"),
    "returning": (BRIDGE, "RETURNING rides DML statements", None),
    "sample": (VERB_TESTED, "sample()", "test_frame.TestSample"),
    "secrets": (BRIDGE, "CREATE SECRET is a statement", None),
    "select": (VERB_TESTED, "column references and select lists", "test_frame.TestProjection"),
    "setops": (VERB_TESTED, "union(), union_by_name(), intersect(), except_()", "test_frame.TestSetOperations"),
    "settings": (BRIDGE, "SET is a statement; run() carries it and forgets stub answers", None),
    "show_select": (VERB_TESTED, "describe() renders SUMMARIZE; DESCRIBE rides sql()", "test_frame.TestInspection"),
    "storage": (ENGINE, "storage format", None),
    "storage_version": (ENGINE, "storage format versions", None),
    "subquery": (
        VERB_TESTED,
        "scalar() and isin(plan), uncorrelated; correlated is the bridge",
        "test_frame.TestSubqueries",
    ),
    "table_function": (
        VERB_TESTED,
        "table_function(), and read_csv/read_parquet/read_json as sources",
        "test_frame.TestTableFunctionSources",
    ),
    "timezone": (SQL_EXPR, "AT TIME ZONE is expression syntax; .dt reaches the functions", None),
    "topn": (ENGINE, "the top-N optimisation; sort().limit() is what triggers it", None),
    "tpcds": (BRIDGE, "the dsdgen extension", None),
    "tpch": (
        VERB_TESTED,
        "all 22 queries expressed with verbs and checked row for row",
        "test_tpch.TestExpressedDirectly",
    ),
    "transactions": (BRIDGE, "BEGIN/COMMIT/ROLLBACK are statements; the DB-API face owns them", None),
    "trigger": (BRIDGE, "triggers are DDL; run() carries them", None),
    "types": (VERB_TESTED, "the value bridge, both directions", "test_expr.TestRichLiteralTypes"),
    "udf_function": (ENGINE, "Python functions need engine surface that lands in wave 2", None),
    "update": (BRIDGE, "UPDATE is a statement, not a query; run() carries it", None),
    "upsert": (BRIDGE, "INSERT ... ON CONFLICT is a statement", None),
    "variant": (ENGINE, "the VARIANT type's semantics; values and cast() reach it and flow through untouched", None),
    "vacuum": (BRIDGE, "VACUUM and ANALYZE are statements", None),
    "variables": (BRIDGE, "SET VARIABLE is a statement", None),
    "vector_types": (ENGINE, "vector layout", None),
    "window": (VERB_TESTED, "over() with frames, the ranking constructors", "test_frame.TestWindowFrames"),
}

KINDS = {VERB_TESTED, VERB_UNTESTED, SQL_EXPR, BRIDGE, ENGINE}


def pinned_engine() -> str:
    """The engine commit in engine.pin: the first line that is not a comment."""
    for line in (Path(__file__).resolve().parents[1] / "engine.pin").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            return line.strip()
    msg = "engine.pin names no commit"
    raise AssertionError(msg)


def corpus() -> Path | None:
    """DuckDB's own test/sql directory, from a checkout at the pinned engine commit, if one is at hand.

    A corpus from another commit tests the engine against expectations it was never built to meet, so an explicit
    DUCKDB_SOURCE at the wrong commit fails and the fallback checkout is only used when it matches.
    """
    explicit = os.environ.get("DUCKDB_SOURCE")
    candidates = [Path(explicit)] if explicit else []
    candidates.append(Path(__file__).resolve().parents[2] / "main" / "external" / "duckdb")
    pinned = pinned_engine()
    for root in candidates:
        if not (root / "test" / "sql").is_dir():
            continue
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if head == pinned:
            return root / "test" / "sql"
        if explicit and root == Path(explicit):
            pytest.fail(f"DUCKDB_SOURCE is at {head[:10]} but engine.pin names {pinned[:10]}; the corpus must match")
    return None


def test_every_entry_is_well_formed() -> None:
    for name, (kind, reason, proof) in FEATURES.items():
        assert kind in KINDS, name
        assert reason, name
        assert (proof is not None) == (kind == VERB_TESTED), f"{name}: a proof exactly when verb-tested"


def test_every_verb_tested_claim_names_a_test_class_that_exists() -> None:
    for name, (kind, _, proof) in FEATURES.items():
        if kind != VERB_TESTED:
            continue
        assert proof is not None
        module_name, class_name = proof.split(".")
        module = importlib.import_module(f"tests.{module_name}")
        assert hasattr(module, class_name), f"{name} claims {proof}, which does not exist"


def test_the_corpus_and_the_table_agree() -> None:
    root = corpus()
    if root is None:
        pytest.skip("no duckdb checkout at the pinned engine commit: set DUCKDB_SOURCE")
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    unclassified = sorted(directories - set(FEATURES))
    assert not unclassified, f"the engine tests features this table does not classify: {unclassified}"
    stale = sorted(set(FEATURES) - directories)
    assert not stale, f"classified, but no longer in the engine's tree: {stale}"


def test_the_shape_of_the_answer() -> None:
    """A record of how the 74 features split: 25 / 0 / 2 / 25 / 22."""
    counts = {kind: sum(1 for k, _, _ in FEATURES.values() if k == kind) for kind in KINDS}
    assert counts[VERB_TESTED] == 25
    assert counts[VERB_UNTESTED] == 0, "a verb reaches a feature nothing proves; write the test"
