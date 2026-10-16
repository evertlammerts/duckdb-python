"""Shared helpers for filter pushdown tests.

Used by both ``test_filter_pushdown.py`` (pyarrow) and
``test_polars_filter_pushdown.py``. The leading underscore prevents pytest
from collecting this module as a test file.

The factories and parametrization data make the same set of comparison
correctness assertions reusable across every Arrow-shaped input the
replacement scan recognises (pyarrow Table, pyarrow Dataset, pyarrow-backed
pandas, polars LazyFrame, polars DataFrame).
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from conftest import PANDAS_GE_3

pa_ds = pytest.importorskip("pyarrow.dataset")


# ===========================================================================
# Conversion factories — pyarrow side
# ===========================================================================


def to_arrow_table(rel):
    return rel.to_arrow_table()


def to_arrow_via_pandas(rel):
    if PANDAS_GE_3:
        return rel.df()
    return rel.df().convert_dtypes(dtype_backend="pyarrow")


def to_arrow_dataset(rel):
    return pa_ds.dataset(rel.to_arrow_table())


def to_polars_lazyframe(rel):
    return rel.pl().lazy()


def to_polars_dataframe(rel):
    return rel.pl()


# Standard parametrization: every test that doesn't care about the conversion
# path runs against both the table and pandas factories.
ARROW_FACTORIES = [
    pytest.param(to_arrow_table, id="table"),
    pytest.param(to_arrow_via_pandas, id="pandas"),
]

ARROW_FACTORIES_WITH_DATASET = [
    *ARROW_FACTORIES,
    pytest.param(to_arrow_dataset, id="dataset"),
]


# ===========================================================================
# Typed data fixtures
#
# For every type we test the same fixed 4-row layout:
#     row 0: (low,  low,  low)
#     row 1: (mid,  mid,  mid)
#     row 2: (high, mid,  high)
#     row 3: (NULL, NULL, NULL)
#
# That layout makes the expected row counts for every comparison the same
# across all types, which is what lets us parametrize the comparison tests.
# ===========================================================================


class TypedCase(NamedTuple):
    id: str  # pytest id
    sql_type: str
    low: str  # SQL literal for the smallest value
    mid: str  # SQL literal for the middle value (duplicated in col b row 2)
    high: str  # SQL literal for the largest value


COMPARABLE_TYPES: list[TypedCase] = [
    # numeric
    TypedCase("tinyint", "TINYINT", "1", "10", "100"),
    TypedCase("smallint", "SMALLINT", "1", "10", "100"),
    TypedCase("integer", "INTEGER", "1", "10", "100"),
    TypedCase("bigint", "BIGINT", "1", "10", "100"),
    TypedCase("utinyint", "UTINYINT", "1", "10", "100"),
    TypedCase("usmallint", "USMALLINT", "1", "10", "100"),
    TypedCase("uinteger", "UINTEGER", "1", "10", "100"),
    TypedCase("ubigint", "UBIGINT", "1", "10", "100"),
    TypedCase("hugeint", "HUGEINT", "1", "10", "100"),
    TypedCase("float", "FLOAT", "1.0", "10.0", "100.0"),
    TypedCase("double", "DOUBLE", "1.0", "10.0", "100.0"),
    TypedCase("decimal_4_1", "DECIMAL(4,1)", "1.0", "10.0", "100.0"),
    TypedCase("decimal_9_1", "DECIMAL(9,1)", "1.0", "10.0", "100.0"),
    TypedCase("decimal_18_4", "DECIMAL(18,4)", "1.0", "10.0", "100.0"),
    TypedCase("decimal_30_12", "DECIMAL(30,12)", "1.0", "10.0", "100.0"),
    # string / blob
    TypedCase("varchar", "VARCHAR", "'1'", "'10'", "'100'"),
    TypedCase("blob", "BLOB", r"'\x01'", r"'\x02'", r"'\x03'"),
    # temporal
    TypedCase("date", "DATE", "'2000-01-01'", "'2000-10-01'", "'2010-01-01'"),
    TypedCase("time", "TIME", "'00:01:00'", "'00:10:00'", "'01:00:00'"),
    TypedCase("timestamp", "TIMESTAMP", "'2008-01-01 00:00:01'", "'2010-01-01 10:00:01'", "'2020-03-01 10:00:01'"),
    TypedCase("timestamptz", "TIMESTAMPTZ", "'2008-01-01 00:00:01'", "'2010-01-01 10:00:01'", "'2020-03-01 10:00:01'"),
]


def make_typed_table(con, factory, case: TypedCase) -> object:
    """Create the standard table for `case`, convert via `factory`, and register it as ``arrow_table``."""
    name = f"_t_{case.id}"
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(f"CREATE TABLE {name} (a {case.sql_type}, b {case.sql_type}, c {case.sql_type})")
    con.execute(
        f"""INSERT INTO {name} VALUES
            ({case.low},  {case.low},  {case.low}),
            ({case.mid},  {case.mid},  {case.mid}),
            ({case.high}, {case.mid},  {case.high}),
            (NULL, NULL, NULL)"""
    )
    arrow_table = factory(con.table(name))
    con.register("arrow_table", arrow_table)
    return arrow_table


def count(con, predicate: str) -> int:
    return con.execute(f"SELECT count(*) FROM arrow_table WHERE {predicate}").fetchone()[0]


# ===========================================================================
# Predicate templates parametrized in `test_comparisons`
# ===========================================================================


# (predicate template, expected row count). Templates reference {low}, {mid}, {high}.
COMPARISON_CASES = [
    pytest.param("a = {low}", 1, id="eq"),
    pytest.param("a != {low}", 2, id="ne"),
    pytest.param("a > {low}", 2, id="gt"),
    pytest.param("a >= {mid}", 2, id="ge"),
    pytest.param("a < {mid}", 1, id="lt"),
    pytest.param("a <= {mid}", 2, id="le"),
    pytest.param("a IS NULL", 1, id="is_null"),
    pytest.param("a IS NOT NULL", 3, id="is_not_null"),
    pytest.param("a = {mid} AND b = {low}", 0, id="and_empty"),
    pytest.param("a = {high} AND b = {mid} AND c = {high}", 1, id="and_match"),
    pytest.param("a = {high} OR b = {low}", 2, id="or"),
]


# ===========================================================================
# Plan-inspection helpers
# ===========================================================================


def arrow_scan_block(plan: str) -> str | None:
    """Return the ARROW_SCAN box (top border to bottom border) from an EXPLAIN plan.

    Works uniformly for pyarrow Tables, pyarrow Datasets, pl.LazyFrame, and
    pl.DataFrame inputs — they all bind through ``arrow_scan`` in DuckDB and
    render as ``ARROW_SCAN`` in the plan.

    Finds the box by its left and right corners nearest to the title or function line,
    then returns only that column range for each line. This excludes neighboring
    boxes in plans with side-by-side operators.
    """
    lines = plan.splitlines()
    scan_idx = None
    for i, line in enumerate(lines):
        if "Arrow Scan" in line or "ARROW_SCAN" in line:
            scan_idx = i
            break
    if scan_idx is None:
        return None

    line = lines[scan_idx]
    scan_pos = line.find("Arrow Scan") if "Arrow Scan" in line else line.find("ARROW_SCAN")

    left_col = None
    for i in range(scan_pos - 1, -1, -1):
        if i < len(line) and line[i] in ("┌", "╭"):
            left_col = i
            break
    if left_col is None:
        return None

    right_col = None
    for i in range(scan_pos + 1, len(line)):
        if line[i] in ("┐", "╮"):
            right_col = i
            break
    if right_col is None:
        return None

    bot = scan_idx
    while bot < len(lines) and left_col < len(lines[bot]) and lines[bot][left_col] not in ("└", "╰"):
        bot += 1

    result = [lines[i][left_col : right_col + 1] for i in range(scan_idx, bot + 1) if i < len(lines)]
    return "\n".join(result)


def _find_arrow_scan_nodes(nodes, parent_is_filter=False):
    """Recursively find all Arrow scan nodes in JSON plan tree.

    Returns list of (node, has_filter_ancestor) tuples.
    """
    result = []
    for node in nodes:
        node_name = node.get("name") or node.get("type")
        is_filter = node_name == "FILTER"

        extra_info = node.get("extra_info", {})
        function = extra_info.get("Function", "")
        if function == "ARROW_SCAN" or node_name == "ARROW_SCAN":
            result.append((node, parent_is_filter or is_filter))

        children = node.get("children", [])
        if children:
            result.extend(_find_arrow_scan_nodes(children, parent_is_filter or is_filter))

    return result


def was_pushed(con, query: str) -> bool:
    """True if EXPLAIN of `query` shows filters pushed to any Arrow scan.

    A filter is considered pushed if it appears in any Arrow scan's Filters field.
    Filters can be a string or a list; empty strings or empty lists mean no filter.
    """
    import json

    result = con.execute(f"EXPLAIN (FORMAT JSON) {query}").fetchone()
    plan = json.loads(result[1])

    scans = _find_arrow_scan_nodes(plan)
    for node, _ in scans:
        filters = node.get("extra_info", {}).get("Filters")
        if filters:
            if isinstance(filters, str):
                if filters.strip():
                    return True
            elif isinstance(filters, list) and len(filters) > 0:
                return True
    return False


def assert_fully_pushed(con, query: str, expected_count: int) -> None:
    """Assert that EXPLAIN ANALYZE of `query` shows its filter inside the only Arrow scan, with nothing above it.

    The scan must emit exactly `expected_count` rows, which shows the pushed filter ran inside the scan.
    """
    import json

    plan = json.loads(con.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}").fetchone()[1])
    scans = _find_arrow_scan_nodes(plan["operator"])
    assert len(scans) == 1
    scan, has_filter_above = scans[0]
    assert scan["extra_info"].get("Filters"), "filter was not pushed into the Arrow scan"
    assert not has_filter_above, "a FILTER operator stayed above the Arrow scan"
    assert scan["intermediate_rows"] == expected_count, "the Arrow scan emitted rows the pushed filter rejects"
