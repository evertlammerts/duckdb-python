"""Where the layer stands against the engine's own feature taxonomy.

DuckDB's `test/sql/` directory is one entry per feature the engine tests.
Each is classified here, with a reason, into one of five kinds:

    verb-tested   the verbs reach it and a named test class proves it
    verb-untested the verbs reach it and nothing here proves it yet
    sql_expr      reachable inside an expression through `sql_expr()`
    bridge        a statement: `sql()` or `run()` carry it, no verb does
    engine        engine behaviour, not a SQL surface this layer presents

The test refuses drift in both directions: a directory the engine adds is
unclassified until someone classifies it, and a directory that disappears
must leave the table. Every verb-tested claim names the class that proves
it, and that class has to exist.

The corpus is the duckdb source tree, found through DUCKDB_SOURCE or the
sibling checkout the main worktree carries. Without one the corpus checks
skip; the internal checks run regardless.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

VERB_TESTED = "verb-tested"
VERB_UNTESTED = "verb-untested"
SQL_EXPR = "sql_expr"
BRIDGE = "bridge"
ENGINE = "engine"

#: directory -> (kind, reason, proof). The proof is `module.Class` for a
#: verb-tested entry and None otherwise.
FEATURES: dict[str, tuple[str, str, str | None]] = {
    "aggregate": (VERB_TESTED, "aggregate(), group_by().agg(), the generated methods", "test_frame.TestAggregation"),
    "alter": (BRIDGE, "ALTER is DDL; run() carries it", None),
    "append": (ENGINE, "the C++ appender, not a SQL surface", None),
    "attach": (BRIDGE, "ATTACH is a statement; run() carries it", None),
    "binder": (ENGINE, "name resolution rules; the layer defers to the binder by design", None),
    "cast": (VERB_TESTED, "Expr.cast()", "test_expr.TestPredicatesAndFunctions"),
    "catalog": (BRIDGE, "catalog DDL and lookups; run() and sql() carry them", None),
    "collate": (SQL_EXPR, "COLLATE is expression syntax the verbs do not spell", None),
    "conjunction": (VERB_TESTED, "& | ~ on expressions", "test_expr.TestOperatorsEvaluate"),
    "constraints": (BRIDGE, "constraints are DDL", None),
    "copy": (VERB_TESTED, "to_parquet() and to_csv() render COPY TO; COPY FROM is a statement", "test_frame.TestSinks"),
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
        ".str/.dt/.list and fn(), spec-tested against the catalog",
        "test_frame.TestExpressionNamespaces",
    ),
    "generated_columns": (BRIDGE, "generated columns are DDL", None),
    "index": (BRIDGE, "indexes are DDL", None),
    "insert": (
        VERB_TESTED,
        "insert_into() renders INSERT ... SELECT; INSERT VALUES is a statement",
        "test_frame.TestSinks",
    ),
    "join": (VERB_TESTED, "join() with every kind in _JOIN_KINDS", "test_frame.TestJoins"),
    "json": (SQL_EXPR, "json functions through fn() or sql_expr(); no .json namespace yet", None),
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
    "table_function": (BRIDGE, "table functions in FROM ride sql()", None),
    "timezone": (SQL_EXPR, "AT TIME ZONE is expression syntax; .dt reaches the functions", None),
    "topn": (ENGINE, "the top-N optimisation; sort().limit() is what triggers it", None),
    "tpcds": (BRIDGE, "the dsdgen extension", None),
    "tpch": (
        VERB_TESTED,
        "all 22 queries expressed with verbs and checked row for row",
        "test_tpch.TestExpressedDirectly",
    ),
    "transactions": (BRIDGE, "BEGIN/COMMIT/ROLLBACK are statements; the DB-API face owns them", None),
    "types": (VERB_TESTED, "the value bridge, both directions", "test_expr.TestRichLiteralTypes"),
    "udf_function": (ENGINE, "Python functions need engine surface that lands in wave 2", None),
    "update": (BRIDGE, "UPDATE is a statement, not a query; run() carries it", None),
    "upsert": (BRIDGE, "INSERT ... ON CONFLICT is a statement", None),
    "vacuum": (BRIDGE, "VACUUM and ANALYZE are statements", None),
    "variables": (BRIDGE, "SET VARIABLE is a statement", None),
    "vector_types": (ENGINE, "vector layout", None),
    "window": (VERB_TESTED, "over() with frames, the ranking constructors", "test_frame.TestWindowFrames"),
}

KINDS = {VERB_TESTED, VERB_UNTESTED, SQL_EXPR, BRIDGE, ENGINE}


def corpus() -> Path | None:
    """The engine's test/sql directory, if a duckdb checkout is at hand."""
    candidates = [Path(os.environ["DUCKDB_SOURCE"])] if "DUCKDB_SOURCE" in os.environ else []
    candidates.append(Path(__file__).resolve().parents[2] / "main" / "external" / "duckdb")
    for root in candidates:
        if (root / "test" / "sql").is_dir():
            return root / "test" / "sql"
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
        pytest.skip("no duckdb checkout: set DUCKDB_SOURCE")
    directories = {p.name for p in root.iterdir() if p.is_dir()}
    unclassified = sorted(directories - set(FEATURES))
    assert not unclassified, f"the engine tests features this table does not classify: {unclassified}"
    stale = sorted(set(FEATURES) - directories)
    assert not stale, f"classified, but no longer in the engine's tree: {stale}"


def test_the_shape_of_the_answer() -> None:
    """A record more than a gate: how the 72 features split. 23 / 0 / 3 / 24 / 19."""
    counts = {kind: sum(1 for k, _, _ in FEATURES.values() if k == kind) for kind in KINDS}
    assert counts[VERB_TESTED] == 23
    assert counts[VERB_UNTESTED] == 0, "a verb reaches a feature nothing proves; write the test"
