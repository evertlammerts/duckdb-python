"""TPC-H as a coverage probe: every query as SQL, and as a chain of verbs, against the same answers.

Derived from TPC-H. Not comparable to published TPC-H results.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb import Expr, col, fn, sql_expr, when

if TYPE_CHECKING:
    from collections.abc import Iterator

QUERIES = sorted((Path(__file__).parent / "tpch").glob("q*.sql"))
SCALE = 0.01  # small enough to stay fast, large enough that joins do work


@pytest.fixture(scope="module")
def tpch() -> Iterator[duckdb.Connection]:
    con = duckdb.connect()
    try:
        # No INSTALL or LOAD: the tpch extension is built in, and asking by name would try to download one.
        con.run(f"CALL dbgen(sf={SCALE})")
    except duckdb.exceptions.Error as error:  # pragma: no cover
        pytest.skip(f"the engine was built without the tpch extension: {error}")
    yield con
    con.close()


def query_text(path: Path) -> str:
    return path.read_text().strip().rstrip(";")


def answer(connection: duckdb.Connection, name: str) -> list[tuple[object, ...]]:
    """What the original query returns, for the verb version to match."""
    return duckdb.sql(query_text(Path(__file__).parent / "tpch" / f"{name}.sql")).rows(connection)


def date(text: str) -> Expr:
    """A date literal, cast the way the queries write it."""
    return sql_expr(f"CAST('{text}' AS date)")


def revenue() -> Expr:
    """`l_extendedprice * (1 - l_discount)`, which eight of the queries want."""
    return col("l_extendedprice") * (1 - col("l_discount"))


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.stem)
def test_the_sql_bridge_carries_every_query(tpch: duckdb.Connection, path: Path) -> None:
    """`sql()` accepts any query DuckDB accepts, which is what makes an unexpressible query harmless."""
    rows = duckdb.sql(query_text(path)).rows(tpch)
    assert isinstance(rows, list)


@pytest.mark.parametrize("path", QUERIES, ids=lambda p: p.stem)
def test_schema_is_available_without_running(tpch: duckdb.Connection, path: Path) -> None:
    """Every query's column names and types are available without running it."""
    frame = duckdb.sql(query_text(path))
    assert frame.columns(tpch), "no columns reported"
    assert len(frame.columns(tpch)) == len(frame.types(tpch))


class TestExpressedDirectly:
    """Queries whose verb version follows the structure of the SQL."""

    def test_q01(self, tpch: duckdb.Connection) -> None:
        disc = revenue()
        actual = (
            duckdb.table("lineitem")
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
        assert actual.rows(tpch) == answer(tpch, "q01")

    def test_q03(self, tpch: duckdb.Connection) -> None:
        customer = duckdb.table("customer").filter(col("c_mktsegment") == "BUILDING")
        orders = duckdb.table("orders").filter(col("o_orderdate") < date("1995-03-15"))
        lineitem = duckdb.table("lineitem").filter(col("l_shipdate") > date("1995-03-15"))
        actual = (
            customer.join(orders, on=col("c_custkey") == col("o_custkey"))
            .join(lineitem, on=col("l_orderkey") == col("o_orderkey"))
            .group_by(col("l_orderkey"), col("o_orderdate"), col("o_shippriority"))
            .agg(revenue().sum().alias("revenue"))
            .select(col("l_orderkey"), col("revenue"), col("o_orderdate"), col("o_shippriority"))
            .sort(col("revenue").desc(), col("o_orderdate"))
            .limit(10)
        )
        assert actual.rows(tpch) == answer(tpch, "q03")

    def test_q05(self, tpch: duckdb.Connection) -> None:
        orders = duckdb.table("orders").filter(
            (col("o_orderdate") >= date("1994-01-01")) & (col("o_orderdate") < date("1995-01-01"))
        )
        actual = (
            duckdb.table("customer")
            .join(orders, on=lambda left, right: left["c_custkey"] == right["o_custkey"])
            .join(duckdb.table("lineitem"), on=lambda left, right: left["o_orderkey"] == right["l_orderkey"])
            .join(
                duckdb.table("supplier"),
                on=lambda left, right: (
                    (left["l_suppkey"] == right["s_suppkey"]) & (left["c_nationkey"] == right["s_nationkey"])
                ),
            )
            .join(duckdb.table("nation"), on=lambda left, right: left["s_nationkey"] == right["n_nationkey"])
            .join(
                duckdb.table("region").filter(col("r_name") == "ASIA"),
                on=lambda left, right: left["n_regionkey"] == right["r_regionkey"],
            )
            .group_by(col("n_name"))
            .agg(revenue().sum().alias("revenue"))
            .sort(col("revenue").desc())
        )
        assert actual.rows(tpch) == answer(tpch, "q05")

    def test_q06(self, tpch: duckdb.Connection) -> None:
        # The simplest query in the set: restrict, then one aggregate.
        actual = (
            duckdb.table("lineitem")
            .filter(
                (col("l_shipdate") >= date("1994-01-01"))
                & (col("l_shipdate") < date("1995-01-01"))
                & col("l_discount").between(0.05, 0.07)
                & (col("l_quantity") < 24)
            )
            .aggregate((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
        )
        assert actual.rows(tpch) == answer(tpch, "q06")

    def test_q07(self, tpch: duckdb.Connection) -> None:
        # `nation` is joined twice, and the second join suffixes its side, so it arrives as n_name_cust.
        actual = (
            duckdb.table("supplier")
            .join(
                duckdb.table("lineitem").filter(col("l_shipdate").between(date("1995-01-01"), date("1996-12-31"))),
                on=lambda left, right: left["s_suppkey"] == right["l_suppkey"],
            )
            .join(duckdb.table("orders"), on=lambda left, right: left["l_orderkey"] == right["o_orderkey"])
            .join(duckdb.table("customer"), on=lambda left, right: left["o_custkey"] == right["c_custkey"])
            .join(duckdb.table("nation"), on=lambda left, right: left["s_nationkey"] == right["n_nationkey"])
            .join(
                duckdb.table("nation"),
                on=lambda left, right: left["c_nationkey"] == right["n_nationkey"],
                suffix="_cust",
            )
            .filter(
                ((col("n_name") == "FRANCE") & (col("n_name_cust") == "GERMANY"))
                | ((col("n_name") == "GERMANY") & (col("n_name_cust") == "FRANCE"))
            )
            .with_columns(l_year=fn("year", col("l_shipdate")), volume=revenue())
            .group_by(col("n_name"), col("n_name_cust"), col("l_year"))
            .agg(col("volume").sum().alias("revenue"))
            .sort(col("n_name"), col("n_name_cust"), col("l_year"))
        )
        assert actual.rows(tpch) == answer(tpch, "q07")

    def test_q08(self, tpch: duckdb.Connection) -> None:
        # The customer's nation is joined first, keeps the plain names, and feeds the region join.
        actual = (
            duckdb.table("part")
            .filter(col("p_type") == "ECONOMY ANODIZED STEEL")
            .join(duckdb.table("lineitem"), on=lambda left, right: left["p_partkey"] == right["l_partkey"])
            .join(duckdb.table("supplier"), on=lambda left, right: left["l_suppkey"] == right["s_suppkey"])
            .join(
                duckdb.table("orders").filter(col("o_orderdate").between(date("1995-01-01"), date("1996-12-31"))),
                on=lambda left, right: left["l_orderkey"] == right["o_orderkey"],
            )
            .join(duckdb.table("customer"), on=lambda left, right: left["o_custkey"] == right["c_custkey"])
            .join(duckdb.table("nation"), on=lambda left, right: left["c_nationkey"] == right["n_nationkey"])
            .join(
                duckdb.table("region").filter(col("r_name") == "AMERICA"),
                on=lambda left, right: left["n_regionkey"] == right["r_regionkey"],
            )
            .join(
                duckdb.table("nation"),
                on=lambda left, right: left["s_nationkey"] == right["n_nationkey"],
                suffix="_supp",
            )
            .with_columns(o_year=fn("year", col("o_orderdate")), volume=revenue())
            .group_by(col("o_year"))
            .agg(
                (
                    when(col("n_name_supp") == "BRAZIL").then(col("volume")).otherwise(0).sum() / col("volume").sum()
                ).alias("mkt_share")
            )
            .sort(col("o_year"))
        )
        assert actual.rows(tpch) == answer(tpch, "q08")

    def test_q09(self, tpch: duckdb.Connection) -> None:
        actual = (
            duckdb.table("part")
            .filter(col("p_name").like("%green%"))
            .join(duckdb.table("lineitem"), on=lambda left, right: left["p_partkey"] == right["l_partkey"])
            .join(duckdb.table("supplier"), on=lambda left, right: left["l_suppkey"] == right["s_suppkey"])
            .join(
                duckdb.table("partsupp"),
                on=lambda left, right: (
                    (left["l_suppkey"] == right["ps_suppkey"]) & (left["l_partkey"] == right["ps_partkey"])
                ),
            )
            .join(duckdb.table("orders"), on=lambda left, right: left["l_orderkey"] == right["o_orderkey"])
            .join(duckdb.table("nation"), on=lambda left, right: left["s_nationkey"] == right["n_nationkey"])
            .with_columns(
                nation=col("n_name"),
                o_year=fn("year", col("o_orderdate")),
                amount=revenue() - col("ps_supplycost") * col("l_quantity"),
            )
            .group_by(col("nation"), col("o_year"))
            .agg(col("amount").sum().alias("sum_profit"))
            .sort(col("nation"), col("o_year").desc())
        )
        assert actual.rows(tpch) == answer(tpch, "q09")

    def test_q10(self, tpch: duckdb.Connection) -> None:
        orders = duckdb.table("orders").filter(
            (col("o_orderdate") >= date("1993-10-01")) & (col("o_orderdate") < date("1994-01-01"))
        )
        actual = (
            duckdb.table("customer")
            .join(orders, on=lambda left, right: left["c_custkey"] == right["o_custkey"])
            .join(
                duckdb.table("lineitem").filter(col("l_returnflag") == "R"),
                on=lambda left, right: left["o_orderkey"] == right["l_orderkey"],
            )
            .join(duckdb.table("nation"), on=lambda left, right: left["c_nationkey"] == right["n_nationkey"])
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
        assert actual.rows(tpch) == answer(tpch, "q10")

    def test_q11(self, tpch: duckdb.Connection) -> None:
        german = (
            duckdb.table("partsupp")
            .join(duckdb.table("supplier"), on=lambda left, right: left["ps_suppkey"] == right["s_suppkey"])
            .join(
                duckdb.table("nation").filter(col("n_name") == "GERMANY"),
                on=lambda left, right: left["s_nationkey"] == right["n_nationkey"],
            )
        )
        value = (col("ps_supplycost") * col("ps_availqty")).sum()
        # Raw SQL keeps the DECIMAL the query means; a Python float would make the comparison DOUBLE.
        threshold = german.aggregate((value * sql_expr("0.0001000000")).alias("t"))
        actual = (
            german.group_by(col("ps_partkey"))
            .agg(value.alias("value"))
            .filter(col("value") > threshold.scalar())
            .sort(col("value").desc())
        )
        assert actual.rows(tpch) == answer(tpch, "q11")

    def test_q12(self, tpch: duckdb.Connection) -> None:
        urgent = (col("o_orderpriority") == "1-URGENT") | (col("o_orderpriority") == "2-HIGH")
        lineitem = duckdb.table("lineitem").filter(
            col("l_shipmode").isin(["MAIL", "SHIP"])
            & (col("l_commitdate") < col("l_receiptdate"))
            & (col("l_shipdate") < col("l_commitdate"))
            & (col("l_receiptdate") >= date("1994-01-01"))
            & (col("l_receiptdate") < date("1995-01-01"))
        )
        actual = (
            duckdb.table("orders")
            .join(lineitem, on=lambda left, right: left["o_orderkey"] == right["l_orderkey"])
            .group_by(col("l_shipmode"))
            .agg(
                when(urgent).then(1).otherwise(0).sum().alias("high_line_count"),
                when(~urgent).then(1).otherwise(0).sum().alias("low_line_count"),
            )
            .sort(col("l_shipmode"))
        )
        assert actual.rows(tpch) == answer(tpch, "q12")

    def test_q14(self, tpch: duckdb.Connection) -> None:
        lineitem = duckdb.table("lineitem").filter(
            (col("l_shipdate") >= date("1995-09-01")) & (col("l_shipdate") < date("1995-10-01"))
        )
        promo = when(col("p_type").like("PROMO%")).then(revenue()).otherwise(0).sum()
        actual = lineitem.join(
            duckdb.table("part"), on=lambda left, right: left["l_partkey"] == right["p_partkey"]
        ).aggregate((sql_expr("100.00") * promo / revenue().sum()).alias("promo_revenue"))
        assert actual.rows(tpch) == answer(tpch, "q14")

    def test_q15(self, tpch: duckdb.Connection) -> None:
        # `totals` feeds both the join and the subquery for the maximum, and is computed once.
        totals = (
            duckdb.table("lineitem")
            .filter((col("l_shipdate") >= date("1996-01-01")) & (col("l_shipdate") < date("1996-04-01")))
            .group_by(col("l_suppkey").alias("supplier_no"))
            .agg(revenue().sum().alias("total_revenue"))
        )
        best = totals.aggregate(col("total_revenue").max().alias("m"))
        actual = (
            duckdb.table("supplier")
            .join(totals, on=lambda left, right: left["s_suppkey"] == right["supplier_no"])
            .filter(col("total_revenue") == best.scalar())
            .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
            .sort(col("s_suppkey"))
        )
        assert actual.rows(tpch) == answer(tpch, "q15")

    def test_q16(self, tpch: duckdb.Connection) -> None:
        complained = (
            duckdb.table("supplier").filter(col("s_comment").like("%Customer%Complaints%")).select(col("s_suppkey"))
        )
        parts = duckdb.table("part").filter(
            (col("p_brand") != "Brand#45")
            & ~col("p_type").like("MEDIUM POLISHED%")
            & col("p_size").isin([49, 14, 23, 45, 19, 3, 36, 9])
        )
        actual = (
            duckdb.table("partsupp")
            .filter(~col("ps_suppkey").isin(complained))
            .join(parts, on=lambda left, right: left["ps_partkey"] == right["p_partkey"])
            .group_by(col("p_brand"), col("p_type"), col("p_size"))
            .agg(col("ps_suppkey").n_unique().alias("supplier_cnt"))
            .sort(col("supplier_cnt").desc(), col("p_brand"), col("p_type"), col("p_size"))
        )
        assert actual.rows(tpch) == answer(tpch, "q16")

    def test_q18(self, tpch: duckdb.Connection) -> None:
        heavy = (
            duckdb.table("lineitem")
            .group_by(col("l_orderkey"))
            .agg(col("l_quantity").sum().alias("q"))
            .filter(col("q") > 300)
            .select(col("l_orderkey"))
        )
        actual = (
            duckdb.table("customer")
            .join(
                duckdb.table("orders").filter(col("o_orderkey").isin(heavy)),
                on=lambda left, right: left["c_custkey"] == right["o_custkey"],
            )
            .join(duckdb.table("lineitem"), on=lambda left, right: left["o_orderkey"] == right["l_orderkey"])
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
        assert actual.rows(tpch) == answer(tpch, "q18")

    def test_q19(self, tpch: duckdb.Connection) -> None:
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
            duckdb.table("lineitem")
            .join(duckdb.table("part"), on=lambda left, right: left["l_partkey"] == right["p_partkey"])
            .filter(
                branch("Brand#12", ["SM CASE", "SM BOX", "SM PACK", "SM PKG"], 1, 5)
                | branch("Brand#23", ["MED BAG", "MED BOX", "MED PKG", "MED PACK"], 10, 10)
                | branch("Brand#34", ["LG CASE", "LG BOX", "LG PACK", "LG PKG"], 20, 15)
            )
            .aggregate(revenue().sum().alias("revenue"))
        )
        assert actual.rows(tpch) == answer(tpch, "q19")


class TestExpressedAfterRewriting:
    """Queries the verbs reach only if the reader restructures them first, and each test names its rewrite."""

    def test_q02(self, tpch: duckdb.Connection) -> None:
        # The correlated min() becomes a group-by joined back, and `europe` is used twice but computed once.
        europe = (
            duckdb.table("partsupp")
            .join(duckdb.table("supplier"), on=lambda left, right: left["ps_suppkey"] == right["s_suppkey"])
            .join(duckdb.table("nation"), on=lambda left, right: left["s_nationkey"] == right["n_nationkey"])
            .join(
                duckdb.table("region").filter(col("r_name") == "EUROPE"),
                on=lambda left, right: left["n_regionkey"] == right["r_regionkey"],
            )
        )
        cheapest = europe.group_by(col("ps_partkey").alias("cheap_partkey")).agg(
            col("ps_supplycost").min().alias("min_cost")
        )
        actual = (
            duckdb.table("part")
            .filter((col("p_size") == 15) & col("p_type").like("%BRASS"))
            .join(europe, on=lambda left, right: left["p_partkey"] == right["ps_partkey"])
            .join(
                cheapest,
                on=lambda left, right: (
                    (left["p_partkey"] == right["cheap_partkey"]) & (left["ps_supplycost"] == right["min_cost"])
                ),
            )
            .select("s_acctbal", "s_name", "n_name", "p_partkey", "p_mfgr", "s_address", "s_phone", "s_comment")
            .sort(col("s_acctbal").desc(), col("n_name"), col("s_name"), col("p_partkey"))
            .limit(100)
        )
        assert actual.rows(tpch) == answer(tpch, "q02")

    def test_q04(self, tpch: duckdb.Connection) -> None:
        # EXISTS becomes a semi join: the left rows that have a match, without the right side's columns.
        actual = (
            duckdb.table("orders")
            .filter((col("o_orderdate") >= date("1993-07-01")) & (col("o_orderdate") < date("1993-10-01")))
            .join(
                duckdb.table("lineitem").filter(col("l_commitdate") < col("l_receiptdate")),
                on=lambda left, right: left["o_orderkey"] == right["l_orderkey"],
                how="semi",
            )
            .group_by(col("o_orderpriority"))
            .agg(sql_expr("count(*)").alias("order_count"))
            .sort(col("o_orderpriority"))
        )
        assert actual.rows(tpch) == answer(tpch, "q04")

    def test_q13(self, tpch: duckdb.Connection) -> None:
        # `on` is a join condition, and for a LEFT JOIN filtering the right input first says the same thing.
        orders = duckdb.table("orders").filter(~col("o_comment").like("%special%requests%"))
        per_customer = (
            duckdb.table("customer")
            .join(orders, on=lambda left, right: left["c_custkey"] == right["o_custkey"], how="left")
            .group_by(col("c_custkey"))
            .agg(col("o_orderkey").count().alias("c_count"))
        )
        actual = (
            per_customer.group_by(col("c_count"))
            .agg(sql_expr("count(*)").alias("custdist"))
            .sort(col("custdist").desc(), col("c_count").desc())
        )
        assert actual.rows(tpch) == answer(tpch, "q13")

    def test_q17(self, tpch: duckdb.Connection) -> None:
        # The correlated avg() becomes a group-by on l_partkey, joined back.
        thresholds = (
            duckdb.table("lineitem")
            .group_by(col("l_partkey").alias("t_partkey"))
            .agg((sql_expr("0.2") * col("l_quantity").mean()).alias("threshold"))
        )
        actual = (
            duckdb.table("lineitem")
            .join(
                duckdb.table("part").filter((col("p_brand") == "Brand#23") & (col("p_container") == "MED BOX")),
                on=lambda left, right: left["l_partkey"] == right["p_partkey"],
            )
            .join(thresholds, on=lambda left, right: left["l_partkey"] == right["t_partkey"])
            .filter(col("l_quantity") < col("threshold"))
            .aggregate((col("l_extendedprice").sum() / sql_expr("7.0")).alias("avg_yearly"))
        )
        assert actual.rows(tpch) == answer(tpch, "q17")

    def test_q20(self, tpch: duckdb.Connection) -> None:
        # The correlated sum is rewritten as a join, which drops the rows the original's NULL comparison did.
        forest = duckdb.table("part").filter(col("p_name").like("forest%")).select(col("p_partkey"))
        shipped = (
            duckdb.table("lineitem")
            .filter((col("l_shipdate") >= date("1994-01-01")) & (col("l_shipdate") < date("1995-01-01")))
            .group_by(col("l_partkey").alias("q_partkey"), col("l_suppkey").alias("q_suppkey"))
            .agg((sql_expr("0.5") * col("l_quantity").sum()).alias("half"))
        )
        excess = (
            duckdb.table("partsupp")
            .filter(col("ps_partkey").isin(forest))
            .join(
                shipped,
                on=lambda left, right: (
                    (left["ps_partkey"] == right["q_partkey"]) & (left["ps_suppkey"] == right["q_suppkey"])
                ),
            )
            .filter(col("ps_availqty") > col("half"))
            .select(col("ps_suppkey"))
        )
        actual = (
            duckdb.table("supplier")
            .filter(col("s_suppkey").isin(excess))
            .join(
                duckdb.table("nation").filter(col("n_name") == "CANADA"),
                on=lambda left, right: left["s_nationkey"] == right["n_nationkey"],
            )
            .select("s_name", "s_address")
            .sort(col("s_name"))
        )
        assert actual.rows(tpch) == answer(tpch, "q20")

    def test_q21(self, tpch: duckdb.Connection) -> None:
        # EXISTS and NOT EXISTS become a semi join and an anti join, both correlating on more than equality.
        late = duckdb.table("lineitem").filter(col("l_receiptdate") > col("l_commitdate"))

        def another_supplier(left: duckdb.Side, right: duckdb.Side) -> duckdb.Expr:
            return (left["l_orderkey"] == right["l_orderkey"]) & (left["l_suppkey"] != right["l_suppkey"])

        actual = (
            duckdb.table("supplier")
            .join(late, on=lambda left, right: left["s_suppkey"] == right["l_suppkey"])
            .join(
                duckdb.table("orders").filter(col("o_orderstatus") == "F"),
                on=lambda left, right: left["l_orderkey"] == right["o_orderkey"],
            )
            .join(
                duckdb.table("nation").filter(col("n_name") == "SAUDI ARABIA"),
                on=lambda left, right: left["s_nationkey"] == right["n_nationkey"],
            )
            .join(duckdb.table("lineitem"), on=another_supplier, how="semi")
            .join(late, on=another_supplier, how="anti")
            .group_by(col("s_name"))
            .agg(sql_expr("count(*)").alias("numwait"))
            .sort(col("numwait").desc(), col("s_name"))
            .limit(100)
        )
        assert actual.rows(tpch) == answer(tpch, "q21")

    def test_q22(self, tpch: duckdb.Connection) -> None:
        # The scalar subquery is uncorrelated, so only the NOT EXISTS needs rewriting into an anti join.
        code = fn("substring", col("c_phone"), 1, 2)
        codes = ["13", "31", "23", "29", "30", "18", "17"]
        average = (
            duckdb.table("customer")
            .filter((col("c_acctbal") > 0.00) & code.isin(codes))
            .aggregate(col("c_acctbal").mean().alias("m"))
        )
        actual = (
            duckdb.table("customer")
            .filter(code.isin(codes))
            .filter(col("c_acctbal") > average.scalar())
            .join(duckdb.table("orders"), on=lambda left, right: left["c_custkey"] == right["o_custkey"], how="anti")
            .with_columns(cntrycode=code)
            .group_by(col("cntrycode"))
            .agg(sql_expr("count(*)").alias("numcust"), col("c_acctbal").sum().alias("totacctbal"))
            .sort(col("cntrycode"))
        )
        assert actual.rows(tpch) == answer(tpch, "q22")


def test_every_query_is_classified_and_proven() -> None:
    """The two classes account for all 22 queries, read from the test methods rather than written out."""

    def covered(cls: type) -> set[str]:
        return {name[len("test_") :] for name in vars(cls) if name.startswith("test_q")}

    direct, rewritten = covered(TestExpressedDirectly), covered(TestExpressedAfterRewriting)
    assert not direct & rewritten, f"a query cannot be in both classes: {sorted(direct & rewritten)}"
    files = {path.stem for path in QUERIES}
    assert direct | rewritten == files, f"unclassified: {sorted(files ^ (direct | rewritten))}"
    # Two thirds transcribe directly and the rest need a known rewrite; none needs raw SQL any more.
    assert len(rewritten) == 7
