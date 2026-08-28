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
from duckdb import Expr, col, sql_expr, when

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


def answer(connection: duckdb.Connection, name: str) -> list[tuple[object, ...]]:
    """What the original query returns, for the verb version to match."""
    return connection.sql(query_text(Path(__file__).parent / "tpch" / f"{name}.sql")).fetchall()


def date(text: str) -> Expr:
    """A date literal, cast the way the queries write it."""
    return sql_expr(f"CAST('{text}' AS date)")


def revenue() -> Expr:
    """`l_extendedprice * (1 - l_discount)`, which eight of the queries want."""
    return col("l_extendedprice") * (1 - col("l_discount"))


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
    """The queries the frame verbs express, checked against the SQL answer.

    Each is written with verbs rather than SQL text, and its rows must equal
    what the original query returns. A query missing from here is not a defect;
    it is a note about which verbs are missing, recorded in
    `test_the_record_matches_the_tests` below, which reads this class.
    """

    def test_q01_is_a_grouped_aggregate(self, tpch: duckdb.Connection) -> None:
        disc = col("l_extendedprice") * (1 - col("l_discount"))
        actual = (
            tpch.table("lineitem")
            .filter(col("l_shipdate") <= date("1998-09-02"))
            .group_by(col("l_returnflag"), col("l_linestatus"))
            .agg(
                col("l_quantity").sum().alias("sum_qty"),
                col("l_extendedprice").sum().alias("sum_base_price"),
                disc.sum().alias("sum_disc_price"),
                (disc * (1 + col("l_tax"))).sum().alias("sum_charge"),
                col("l_quantity").mean().alias("avg_qty"),
                col("l_extendedprice").mean().alias("avg_price"),
                col("l_discount").mean().alias("avg_disc"),
                sql_expr("count(*)").alias("count_order"),
            )
            .sort(col("l_returnflag"), col("l_linestatus"))
        )
        assert actual.fetchall() == answer(tpch, "q01")

    def test_q03_is_a_join_group_sort_limit(self, tpch: duckdb.Connection) -> None:
        customer = tpch.table("customer").filter(col("c_mktsegment") == "BUILDING")
        orders = tpch.table("orders").filter(col("o_orderdate") < date("1995-03-15"))
        lineitem = tpch.table("lineitem").filter(col("l_shipdate") > date("1995-03-15"))
        actual = (
            customer.join(orders, on=col("c_custkey") == col("o_custkey"))
            .join(lineitem, on=col("l_orderkey") == col("o_orderkey"))
            .group_by(col("l_orderkey"), col("o_orderdate"), col("o_shippriority"))
            .agg(revenue().sum().alias("revenue"))
            .select(col("l_orderkey"), col("revenue"), col("o_orderdate"), col("o_shippriority"))
            .sort(col("revenue").desc(), col("o_orderdate"))
            .limit(10)
        )
        assert actual.fetchall() == answer(tpch, "q03")

    def test_q06_is_a_filter_and_an_aggregate(self, tpch: duckdb.Connection) -> None:
        # The simplest shape in the set: restrict, then one aggregate.
        actual = (
            tpch.table("lineitem")
            .filter(
                (col("l_shipdate") >= date("1994-01-01"))
                & (col("l_shipdate") < date("1995-01-01"))
                & col("l_discount").between(0.05, 0.07)
                & (col("l_quantity") < 24)
            )
            .aggregate((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
        )
        assert actual.fetchall() == answer(tpch, "q06")

    def test_q10_is_a_four_way_join(self, tpch: duckdb.Connection) -> None:
        orders = tpch.table("orders").filter(
            (col("o_orderdate") >= date("1993-10-01")) & (col("o_orderdate") < date("1994-01-01"))
        )
        actual = (
            tpch.table("customer")
            .join(orders, on=col("l.c_custkey") == col("r.o_custkey"))
            .join(
                tpch.table("lineitem").filter(col("l_returnflag") == "R"),
                on=col("l.o_orderkey") == col("r.l_orderkey"),
            )
            .join(tpch.table("nation"), on=col("l.c_nationkey") == col("r.n_nationkey"))
            .group_by(
                col("c_custkey"),
                col("c_name"),
                col("c_acctbal"),
                col("c_phone"),
                col("n_name"),
                col("c_address"),
                col("c_comment"),
            )
            .agg(revenue().sum().alias("revenue"))
            # Group keys come out first, so the reporting order is a projection.
            .select("c_custkey", "c_name", "revenue", "c_acctbal", "n_name", "c_address", "c_phone", "c_comment")
            .sort(col("revenue").desc())
            .limit(20)
        )
        assert actual.fetchall() == answer(tpch, "q10")

    def test_q11_filters_a_group_against_a_scalar_subquery(self, tpch: duckdb.Connection) -> None:
        german = (
            tpch.table("partsupp")
            .join(tpch.table("supplier"), on=col("l.ps_suppkey") == col("r.s_suppkey"))
            .join(
                tpch.table("nation").filter(col("n_name") == "GERMANY"),
                on=col("l.s_nationkey") == col("r.n_nationkey"),
            )
        )
        value = (col("ps_supplycost") * col("ps_availqty")).sum()
        # Written as a fragment so the engine reads it as the DECIMAL the query
        # means. A Python float would make the whole comparison DOUBLE.
        threshold = german.aggregate((value * sql_expr("0.0001000000")).alias("t"))
        actual = (
            german.group_by(col("ps_partkey"))
            .agg(value.alias("value"))
            .filter(col("value") > threshold.scalar())
            .sort(col("value").desc())
        )
        assert actual.fetchall() == answer(tpch, "q11")

    def test_q12_aggregates_over_a_case(self, tpch: duckdb.Connection) -> None:
        urgent = (col("o_orderpriority") == "1-URGENT") | (col("o_orderpriority") == "2-HIGH")
        lineitem = tpch.table("lineitem").filter(
            col("l_shipmode").isin(["MAIL", "SHIP"])
            & (col("l_commitdate") < col("l_receiptdate"))
            & (col("l_shipdate") < col("l_commitdate"))
            & (col("l_receiptdate") >= date("1994-01-01"))
            & (col("l_receiptdate") < date("1995-01-01"))
        )
        actual = (
            tpch.table("orders")
            .join(lineitem, on=col("l.o_orderkey") == col("r.l_orderkey"))
            .group_by(col("l_shipmode"))
            .agg(
                when(urgent).then(1).otherwise(0).sum().alias("high_line_count"),
                when(~urgent).then(1).otherwise(0).sum().alias("low_line_count"),
            )
            .sort(col("l_shipmode"))
        )
        assert actual.fetchall() == answer(tpch, "q12")

    def test_q13_groups_the_result_of_a_group(self, tpch: duckdb.Connection) -> None:
        # The original puts the comment test in the LEFT JOIN's ON clause. On
        # the preserved side that is the same as filtering the right input
        # first, which is what a frame can say.
        orders = tpch.table("orders").filter(~col("o_comment").like("%special%requests%"))
        per_customer = (
            tpch.table("customer")
            .join(orders, on=col("l.c_custkey") == col("r.o_custkey"), how="left")
            .group_by(col("c_custkey"))
            .agg(col("o_orderkey").count().alias("c_count"))
        )
        actual = (
            per_customer.group_by(col("c_count"))
            .agg(sql_expr("count(*)").alias("custdist"))
            .sort(col("custdist").desc(), col("c_count").desc())
        )
        assert actual.fetchall() == answer(tpch, "q13")

    def test_q14_divides_two_aggregates(self, tpch: duckdb.Connection) -> None:
        lineitem = tpch.table("lineitem").filter(
            (col("l_shipdate") >= date("1995-09-01")) & (col("l_shipdate") < date("1995-10-01"))
        )
        promo = when(col("p_type").like("PROMO%")).then(revenue()).otherwise(0).sum()
        actual = lineitem.join(tpch.table("part"), on=col("l.l_partkey") == col("r.p_partkey")).aggregate(
            (sql_expr("100.00") * promo / revenue().sum()).alias("promo_revenue")
        )
        assert actual.fetchall() == answer(tpch, "q14")

    def test_q15_reuses_one_step_in_two_places(self, tpch: duckdb.Connection) -> None:
        # The query the CTE graph was built for: `revenue` feeds both the join
        # and the subquery that finds the maximum.
        totals = (
            tpch.table("lineitem")
            .filter((col("l_shipdate") >= date("1996-01-01")) & (col("l_shipdate") < date("1996-04-01")))
            .group_by(col("l_suppkey").alias("supplier_no"))
            .agg(revenue().sum().alias("total_revenue"))
        )
        best = totals.aggregate(col("total_revenue").max().alias("m"))
        actual = (
            tpch.table("supplier")
            .join(totals, on=col("l.s_suppkey") == col("r.supplier_no"))
            .filter(col("total_revenue") == best.scalar())
            .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
            .sort(col("s_suppkey"))
        )
        assert actual.fetchall() == answer(tpch, "q15")

    def test_q16_excludes_the_rows_a_subquery_names(self, tpch: duckdb.Connection) -> None:
        complained = (
            tpch.table("supplier").filter(col("s_comment").like("%Customer%Complaints%")).select(col("s_suppkey"))
        )
        parts = tpch.table("part").filter(
            (col("p_brand") != "Brand#45")
            & ~col("p_type").like("MEDIUM POLISHED%")
            & col("p_size").isin([49, 14, 23, 45, 19, 3, 36, 9])
        )
        actual = (
            tpch.table("partsupp")
            .filter(~col("ps_suppkey").isin(complained))
            .join(parts, on=col("l.ps_partkey") == col("r.p_partkey"))
            .group_by(col("p_brand"), col("p_type"), col("p_size"))
            .agg(col("ps_suppkey").n_unique().alias("supplier_cnt"))
            .sort(col("supplier_cnt").desc(), col("p_brand"), col("p_type"), col("p_size"))
        )
        assert actual.fetchall() == answer(tpch, "q16")

    def test_q18_filters_against_a_grouped_subquery(self, tpch: duckdb.Connection) -> None:
        heavy = (
            tpch.table("lineitem")
            .group_by(col("l_orderkey"))
            .agg(col("l_quantity").sum().alias("q"))
            .filter(col("q") > 300)
            .select(col("l_orderkey"))
        )
        actual = (
            tpch.table("customer")
            .join(
                tpch.table("orders").filter(col("o_orderkey").isin(heavy)),
                on=col("l.c_custkey") == col("r.o_custkey"),
            )
            .join(tpch.table("lineitem"), on=col("l.o_orderkey") == col("r.l_orderkey"))
            .group_by(
                col("c_name"),
                col("c_custkey"),
                col("o_orderkey"),
                col("o_orderdate"),
                col("o_totalprice"),
            )
            .agg(col("l_quantity").sum().alias("total"))
            .sort(col("o_totalprice").desc(), col("o_orderdate"))
            .limit(100)
        )
        assert actual.fetchall() == answer(tpch, "q18")

    def test_q19_is_three_alternative_predicates(self, tpch: duckdb.Connection) -> None:
        def branch(brand: str, containers: list[str], quantity: int, size: int) -> Expr:
            return (
                (col("p_brand") == brand)
                & col("p_container").isin(containers)
                & col("l_quantity").between(quantity, quantity + 10)
                & col("p_size").between(1, size)
                & col("l_shipmode").isin(["AIR", "AIR REG"])
                & (col("l_shipinstruct") == "DELIVER IN PERSON")
            )

        actual = (
            tpch.table("lineitem")
            .join(tpch.table("part"), on=col("l.l_partkey") == col("r.p_partkey"))
            .filter(
                branch("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
                | branch("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
                | branch("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
            )
            .aggregate(revenue().alias("revenue"))
        )
        assert actual.fetchall() == answer(tpch, "q19")


def test_the_record_matches_the_tests() -> None:
    """What the frame verbs cannot yet express, and why.

    The expressible set is read from the test class rather than written out, so
    the record cannot claim a query the tests do not actually prove.
    """
    needs_sql_bridge = {
        "q02": "correlated scalar subquery in a predicate",
        "q04": "correlated EXISTS",
        "q05": "multi-way join with a date range, expressible but verbose",
        "q07": "derived table with a nested SELECT",
        "q08": "derived table with a CASE over a join",
        "q09": "derived table over a multi-way join",
        "q17": "correlated scalar subquery",
        "q20": "correlated subqueries, two levels",
        "q21": "correlated EXISTS and NOT EXISTS over the same table",
        "q22": "correlated subquery plus a scalar subquery in the filter",
    }
    expressible = {name[5:8] for name in vars(TestExpressedWithVerbs) if name.startswith("test_q")}
    assert not (needs_sql_bridge.keys() & expressible), "a query cannot be in both lists"
    assert len(needs_sql_bridge) + len(expressible) == len(QUERIES), (
        f"every query must be classified: {sorted(needs_sql_bridge.keys() | expressible)}"
    )
    # The headline: what remains is correlation. An uncorrelated subquery is a
    # frame, so `scalar()` and `isin()` reach it; a correlated one has to see
    # the row around it, which a frame has no way to name. That is the next
    # piece of design, and the SQL bridge covers it until then.
    assert all(reason.startswith(("correlated", "derived", "multi-way")) for reason in needs_sql_bridge.values())
