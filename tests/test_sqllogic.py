"""The engine's own sqllogictest corpus, run through `sql()` and `run()`.

Anything the engine accepts must pass through this layer unchanged. The
engine's `test/sql/` files are the fullest statement of what it accepts, so
the SELECT-shaped directories are run here file by file, each on a fresh
database, and held to the outcomes the files record.

Two measurements per directory. *Carried*: every statement ran with the
outcome its file expects; a shortfall is a bridge bug, or version skew
between the corpus and the engine. *Matched*: every compared query returned
the rows its file expects; a shortfall is either a real difference or a
formatting rule the runner does not know yet, and each is listed.

Needs a duckdb checkout (DUCKDB_SOURCE, or the sibling the main worktree
carries) and takes a minute, so it runs only when asked: `pytest -m corpus`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import duckdb

from ._sqllogic import as_text, close_enough, error_matches, placeholders, run_file
from .test_feature_coverage import corpus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DIRECTORIES = ["aggregate", "join", "subquery", "cte", "window", "setops", "order", "cast", "filter", "projection"]

#: What each directory reached on 2026-08-30, as a floor. A drop is a
#: regression; a rise is a reason to raise the floor.
FLOORS = {
    "aggregate": (0.99, 0.96),
    "join": (0.99, 0.98),
    "subquery": (0.98, 0.99),
    "cte": (0.99, 0.98),
    "window": (0.99, 0.98),
    "setops": (0.99, 0.97),
    "order": (0.99, 0.99),
    "cast": (0.99, 0.99),
    "filter": (0.96, 0.99),
    "projection": (0.99, 0.99),
}


@pytest.mark.corpus
@pytest.mark.parametrize("directory", DIRECTORIES)
def test_the_engine_corpus_rides_the_bridge(directory: str, tmp_path: Path) -> None:
    root = corpus()
    if root is None:
        pytest.skip("no duckdb checkout: set DUCKDB_SOURCE")
    outcomes = [run_file(path, tmp_path, root.parent.parent) for path in sorted((root / directory).rglob("*.test"))]
    ran = [o for o in outcomes if o.skipped is None]
    assert ran, f"nothing in {directory} could run"
    total = sum(o.statements + o.queries for o in ran)
    carried = sum(o.carried for o in ran)
    compared = sum(o.compared for o in ran)
    matched = sum(o.matched for o in ran)
    not_carried = [f"{o.path.name}: {item}" for o in ran for item in o.not_carried]
    mismatched = [f"{o.path.name}: {item}" for o in ran for item in o.mismatched]
    carried_floor, matched_floor = FLOORS[directory]
    report = (
        f"{directory}: {len(ran)}/{len(outcomes)} files ran, carried {carried}/{total}, "
        f"matched {matched}/{compared}\nnot carried:\n  "
        + "\n  ".join(not_carried[:12])
        + "\nmismatched:\n  "
        + "\n  ".join(mismatched[:12])
    )
    assert carried >= carried_floor * total, report
    assert matched >= matched_floor * compared, report


# The engine's own VARCHAR cast is the reference for how a value prints.
PRINTED = [
    "[MAP {1: 'a'}, MAP {2: 'b', 3: 'c'}]",
    "{'m': MAP {1: 'a'}}",
    "MAP {[1, 2]: 'a'}",
    "MAP {'x y': 'a,b'}",
    "{'k': 'a,b', 'j': 'plain', 'n': NULL}",
    "[{'k': 'v'}]",
    "{'a b': [1, 2]}",
    "INTERVAL '-1 hour'",
    "INTERVAL '-1 day -1 hour'",
    "INTERVAL '-2 day'",
    "INTERVAL '-90 minutes -0.5 second'",
    "INTERVAL '1 day 1.5 second'",
    "INTERVAL '0 second'",
    "['a b']",
    "['']",
    "['null']",
    "[' x']",
    "['x ']",
    "['it''s']",
    "['a\\\\b']",
    "['f(x)']",
    "[TIME '12:30']",
    "[DATE '2020-01-01']",
    "[TIMESTAMP '2020-01-01 00:00:00']",
    "[1.5, NULL]",
    "[[1], [2, 3]]",
    "[{'k': [1]}]",
    "{'a, b': 1, 'c': 2}",
    "{'it''s': 'v'}",
    "MAP {'x': 'a=b'}",
    "MAP {TIME '12:30': 1}",
    "{'k': 'v', 'n': NULL}",
]


@pytest.mark.parametrize("expression", PRINTED)
def test_the_runner_prints_a_value_as_the_engine_does(expression: str) -> None:
    # Review round 5, findings 8 and 9: a LIST of MAP, a MAP inside a STRUCT,
    # and a negative interval all print as the engine prints them.
    con = duckdb.connect()
    plan = duckdb.sql(f"SELECT {expression} AS v, ({expression})::VARCHAR AS text")
    ((value, text),) = plan.rows(con)
    assert as_text(value, plan.types(con)[0]) == text


def test_placeholders_resolve_as_the_engines_runner_resolves_them(tmp_path: Path) -> None:
    text = "COPY t TO '{TEST_DIR}/a.csv'; FROM '__TEST_DIR__/b'; FROM '__WORKING_DIRECTORY__/data'; {BASE_TEST_NAME}"
    resolved = placeholders(text, tmp_path, Path("/checkout"), "join/x.test")
    assert resolved == (f"COPY t TO '{tmp_path}/a.csv'; FROM '{tmp_path}/b'; FROM '/checkout/data'; join_x.test")
    assert placeholders("{UUID}", tmp_path, Path("/c"), "t") != placeholders("{UUID}", tmp_path, Path("/c"), "t")


def test_a_file_using_the_current_placeholder_spelling_runs_to_its_queries(tmp_path: Path) -> None:
    # The engine's files write `{TEST_DIR}` now; a runner that only knew the
    # older spelling failed the setup and never reached the queries.
    file = tmp_path / "copy.test"
    file.write_text(
        "statement ok\nCOPY (SELECT 7 AS a) TO '{TEST_DIR}/one.csv' (HEADER)\n\n"
        "query I\nSELECT a FROM '{TEST_DIR}/one.csv'\n----\n7\n"
    )
    outcome = run_file(file, tmp_path)
    assert (outcome.carried, outcome.matched, outcome.not_carried, outcome.mismatched) == (2, 1, [], [])


def test_an_expected_error_matches_as_the_engines_runner_matches_it() -> None:
    message = 'Binder Error: Referenced column "x" not found in FROM clause!'
    assert error_matches(message, ['Referenced column "x"'])
    assert error_matches(message, ['<REGEX>:.*column "x".*'])
    assert error_matches(message, ["<!REGEX>:.*Parser Error.*"])
    assert error_matches(message, [])
    assert not error_matches(message, ["<REGEX>:Parser Error.*"])
    assert not error_matches(message, ["referenced column"])  # case matters, as it does in the engine's runner


@pytest.mark.parametrize(
    ("expected", "actual", "column_type", "verdict"),
    [
        ("0010", "10", "VARCHAR", False),
        ("0010", "10", "INTEGER", True),
        ("10", "10.0", "INTEGER", True),
        ("0.1", "0.1000001", "DOUBLE", True),
        ("0.1", "0.2", "DOUBLE", False),
        ("infinity", "inf", "VARCHAR", False),
        ("inf", "inf", "DOUBLE", True),
        ("NULL", "0", "INTEGER", False),
        ("1", "true", "BOOLEAN", True),
        ("0", "false", "BOOLEAN", True),
        ("<REGEX>:a.*", "abc", "VARCHAR", True),
        ("<REGEX>:b.*", "abc", "VARCHAR", False),
        ("1.00", "1", "DECIMAL(18,3)", True),
        ("[true]", "[false]", "BOOLEAN[]", False),
        ("[true]", "[true]", "BOOLEAN[]", True),
        ("x", "y", "BOOLEAN", False),
        ("NULL", "true", "BOOLEAN", False),
    ],
)
def test_values_compare_by_the_columns_type(expected: str, actual: str, column_type: str, verdict: bool) -> None:
    # The letter in a `query` line plays no part in the engine's runner; the
    # result's SQL type decides whether text compares as a number.
    assert close_enough(expected, actual, column_type) is verdict


def test_a_crash_in_the_client_during_a_statement_is_recorded_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file = tmp_path / "crash.test"
    file.write_text("statement ok\nSELECT 'boom'\n\nquery I\nSELECT 1\n----\n1\n")
    original = duckdb.Connection.run

    def run(self: duckdb.Connection, text: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> int:
        if "boom" in text:
            message = "a client-side failure"
            raise RuntimeError(message)
        return original(self, text, parameters)

    monkeypatch.setattr(duckdb.Connection, "run", run)
    outcome = run_file(file, tmp_path)
    assert outcome.not_carried == ["statement CRASHED: \"SELECT 'boom'\" -> RuntimeError: a client-side failure"]
