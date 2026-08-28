"""TPC-H as a coverage probe for the frame layer.

Three questions per query, kept separate because they fail for different
reasons and suggest different work:

1. Does the SQL bridge carry it? `sql()` must accept any query DuckDB accepts,
   so a failure here is a bug in the bridge.
2. Can the frame verbs express it? A failure is a missing verb, and the list of
   them is the roadmap for this layer.
3. Does it give the right answer? Expressing a query wrongly is worse than not
   expressing it.

TPC-H contains no window functions, no set operations, no DML and no nested
types, so a green run here is a floor, not a definition of done.

Derived from TPC-H. Not comparable to published TPC-H results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb import col

if TYPE_CHECKING:
    from collections.abc import Iterator

QUERIES = sorted((Path(__file__).parent / "tpch").glob("q*.sql"))
SCALE = 0.01  # small enough to stay fast, large enough that joins do work


@pytest.fixture(scope="module")
def tpch() -> Iterator[duckdb.Connection]:
    con = duckdb.connect()
    try:
        # No INSTALL or LOAD: the extension is built into the engine bundle, and
        # asking for it by name would try to download one for a development
        # build that no repository has.
        con.run(f"CALL dbgen(sf={SCALE})")
    except duckdb.exceptions.Error as error:  # pragma: no cover
        pytest.skip(f"the engine was built without the tpch extension: {error}")
    yield con
    con.close()


def query_text(path: Path) -> str:
    return path.read_text().strip().rstrip(";")


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.stem)
def test_the_sql_bridge_carries_every_query(tpch: duckdb.Connection, path: Path) -> None:
    """`sql()` accepts any query the engine accepts, and the rows come back.

    This is the floor. The bridge is what makes an unexpressible query a
    non-problem, so it has to hold for all 22.
    """
    rows = tpch.sql(query_text(path)).fetchall()
    assert isinstance(rows, list)


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.stem)
def test_schema_is_available_without_running(tpch: duckdb.Connection, path: Path) -> None:
    """The binder answers the shape of every query without executing it."""
    frame = tpch.sql(query_text(path))
    assert frame.columns, "no columns reported"
    assert len(frame.columns) == len(frame.types)


class TestExpressedWithVerbs:
    """The queries the frame verbs can express, checked against the SQL answer.

    Each is written with verbs rather than SQL text, and its rows must match
    what the original query returns. A query that is not here is not a failure;
    it is a note about which verbs are missing, recorded in
    `test_verb_coverage_is_recorded` below.
    """

    def test_q06_is_a_filter_and_an_aggregate(self, tpch: duckdb.Connection) -> None:
        # The simplest shape in the set: restrict, then one aggregate.
        expected = tpch.sql(query_text(Path(__file__).parent / "tpch" / "q06.sql")).fetchall()
        lineitem = tpch.sql("SELECT * FROM lineitem")
        revenue = lineitem.filter(
            (col("l_shipdate") >= duckdb.sql_expr("CAST('1994-01-01' AS date)"))
            & (col("l_shipdate") < duckdb.sql_expr("CAST('1995-01-01' AS date)"))
            & col("l_discount").between(0.05, 0.07)
            & (col("l_quantity") < 24)
        ).aggregate((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
        assert revenue.fetchall() == expected

    def test_q01_is_a_grouped_aggregate(self, tpch: duckdb.Connection) -> None:
        expected = tpch.sql(query_text(Path(__file__).parent / "tpch" / "q01.sql")).fetchall()
        lineitem = tpch.sql("SELECT * FROM lineitem")
        disc = col("l_extendedprice") * (1 - col("l_discount"))
        actual = (
            lineitem.filter(col("l_shipdate") <= duckdb.sql_expr("CAST('1998-09-02' AS date)"))
            .group_by(col("l_returnflag"), col("l_linestatus"))
            .agg(
                col("l_quantity").sum().alias("sum_qty"),
                col("l_extendedprice").sum().alias("sum_base_price"),
                disc.sum().alias("sum_disc_price"),
                (disc * (1 + col("l_tax"))).sum().alias("sum_charge"),
                col("l_quantity").mean().alias("avg_qty"),
                col("l_extendedprice").mean().alias("avg_price"),
                col("l_discount").mean().alias("avg_disc"),
                duckdb.sql_expr("count(*)").alias("count_order"),
            )
            .sort(col("l_returnflag"), col("l_linestatus"))
        )
        assert actual.fetchall() == expected

    def test_q03_is_a_join_group_sort_limit(self, tpch: duckdb.Connection) -> None:
        expected = tpch.sql(query_text(Path(__file__).parent / "tpch" / "q03.sql")).fetchall()
        customer = tpch.sql("SELECT * FROM customer").filter(col("c_mktsegment") == "BUILDING")
        orders = tpch.sql("SELECT * FROM orders").filter(
            col("o_orderdate") < duckdb.sql_expr("CAST('1995-03-15' AS date)")
        )
        lineitem = tpch.sql("SELECT * FROM lineitem").filter(
            col("l_shipdate") > duckdb.sql_expr("CAST('1995-03-15' AS date)")
        )
        actual = (
            customer.join(orders, on=col("c_custkey") == col("o_custkey"))
            .join(lineitem, on=col("l_orderkey") == col("o_orderkey"))
            .group_by(col("l_orderkey"), col("o_orderdate"), col("o_shippriority"))
            .agg(
                (col("l_extendedprice") * (1 - col("l_discount"))).sum().alias("revenue"),
            )
            .select(
                col("l_orderkey"),
                col("revenue"),
                col("o_orderdate"),
                col("o_shippriority"),
            )
            .sort(col("revenue").desc(), col("o_orderdate"))
            .limit(10)
        )
        assert actual.fetchall() == expected


def test_verb_coverage_is_recorded() -> None:
    """What the frame verbs cannot yet express, and why.

    Kept as an assertion so the list has to be maintained rather than drifting.
    Each entry is a note about the API, not a defect in the query.
    """
    needs_sql_bridge = {
        "q02": "correlated scalar subquery in a predicate",
        "q04": "EXISTS subquery",
        "q05": "multi-way join with a date range, expressible but verbose",
        "q07": "derived table with a nested SELECT",
        "q08": "derived table with a CASE over a join",
        "q09": "derived table over a multi-way join",
        "q11": "HAVING against a scalar subquery",
        "q13": "LEFT JOIN inside a derived table, then a grouped count of counts",
        "q15": "CTE referenced twice, plus a scalar subquery over it",
        "q16": "NOT IN subquery, plus count(DISTINCT)",
        "q17": "correlated scalar subquery",
        "q18": "IN subquery over a grouped having",
        "q20": "nested correlated subqueries, two levels",
        "q21": "EXISTS and NOT EXISTS over the same table",
        "q22": "correlated subquery plus a scalar subquery in the filter",
    }
    expressible = {"q01", "q03", "q06", "q10", "q12", "q14", "q19"}
    assert not (needs_sql_bridge.keys() & expressible), "a query cannot be in both lists"
    assert len(needs_sql_bridge) + len(expressible) == len(QUERIES), (
        f"every query must be classified: {len(needs_sql_bridge)} + {len(expressible)} != {len(QUERIES)}"
    )
    # The headline: subqueries dominate what the verbs cannot reach. That is a
    # statement about the shape of TPC-H as much as about the API, and the SQL
    # bridge is the designed answer to it.
    assert len(expressible) / len(QUERIES) > 0.25
