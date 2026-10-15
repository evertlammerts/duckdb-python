"""A relation built by rel.query(name, sql) reads the relation it came from, whatever is registered under name later."""

import gc
import io
import re

import pytest

import duckdb

LATE_BINDING = "limitation: a user CTE named like the relation keeps late binding"


def reuse_name(con, name):
    """Point the temporary view called name at a different relation, the way a second query() call does."""
    con.sql("select 999 as a, 'x' as s").query(name, "select 42")


class TestQueryBindsItsNameOnce:
    def test_second_query_with_the_same_name_keeps_the_first(self):
        con = duckdb.connect()
        a = con.sql("select 1 as a")
        b = con.sql("select 999 as a")
        first = a.query("v", "select a from v")
        assert first.fetchall() == [(1,)]
        second = b.query("v", "select a from v")
        assert second.fetchall() == [(999,)]
        assert first.fetchall() == [(1,)]

    def test_reuse_before_first_execution(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")
        reuse_name(con, "v")
        assert first.fetchall() == [(1,)]

    def test_reuse_with_a_different_schema(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")
        con.sql("select 'x' as s").query("v", "select s from v")
        assert first.columns == ["a"]
        assert first.types == [duckdb.sqltypes.INTEGER]
        assert first.fetchall() == [(1,)]

    def test_view_dropped_after_query(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")
        con.execute("drop view v")
        assert first.fetchall() == [(1,)]

    @pytest.mark.parametrize(
        ("name", "sql", "expected"),
        [
            ("v", "select a from v union all select a + 1 from v order by a", [(1,), (2,)]),
            ("v", "select (select max(a) from v) as m, exists (from v) as e", [(1, True)]),
            ("v", "from v", [(1,)]),
            ("v", "with w as (select a + 1 as a from v) select a from w", [(2,)]),
            ("V", "select a from v", [(1,)]),
            ("my view", 'select a from "my view"', [(1,)]),
            ("v", "describe v", [("a", "INTEGER", "YES", None, None, None)]),
            ("v", "describe select * from v", [("a", "INTEGER", "YES", None, None, None)]),
            ("v", "select column_name, column_type from (summarize v)", [("a", "INTEGER")]),
        ],
    )
    def test_statement_shapes(self, name, sql, expected):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query(name, sql)
        reuse_name(con, name)
        assert rel.fetchall() == expected

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=LATE_BINDING)
    def test_user_cte_with_the_same_name_reads_the_source(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", "with v as (select a + 1 as a from v) select a from v")
        reuse_name(con, "v")
        assert rel.fetchall() == [(2,)]

    def test_user_cte_with_the_same_name_shadows_the_source(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", "with v as (select 5 as a) select a from v")
        assert rel.fetchall() == [(5,)]

    def test_pragma_show(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", "pragma \"show\"('v')")
        reuse_name(con, "v")
        assert rel.fetchall() == [("a", "INTEGER", "YES", None, None, None)]

    def test_pragma_show_describes_its_argument(self):
        con = duckdb.connect()
        con.execute("create table other(o varchar)")
        rel = con.sql("select 1 as a").query("v", "pragma \"show\"('other')")
        assert rel.fetchall() == [("o", "VARCHAR", "YES", None, None, None)]

    @pytest.mark.parametrize("sql", ['pragma "show"', "pragma \"show\"('other', 1)"])
    def test_pragma_show_without_one_table_name_raises_like_sql(self, sql):
        con = duckdb.connect()
        con.execute("create table other(o varchar)")
        with pytest.raises(duckdb.BinderException) as from_sql:
            con.sql(sql)
        with pytest.raises(duckdb.BinderException, match=re.escape(str(from_sql.value))):
            con.sql("select 1 as a").query("v", sql)

    def test_pragma_show_of_an_empty_name_raises_like_sql(self):
        con = duckdb.connect()
        with pytest.raises(duckdb.ParserException) as from_sql:
            con.sql("pragma \"show\"('')")
        with pytest.raises(duckdb.ParserException, match=re.escape(str(from_sql.value))):
            con.sql("select 1 as a").query("v", "pragma \"show\"('')")

    @pytest.mark.parametrize(
        "sql", ["select column_name from pragma_show('v')", "select name from pragma_table_info('v')"]
    )
    def test_catalog_table_functions_read_the_view(self, sql):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", sql)
        reuse_name(con, "v")
        assert rel.fetchall() == [("a",), ("s",)]


class TestUserCteWithTheSourceName:
    def test_duplicate_column_names(self):
        pytest.importorskip("pyarrow")
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", "with v as (select a + 1 as a from v) select a, a from v")
        assert rel.to_arrow_table().column_names == ["a", "a"]
        assert rel.fetchall() == [(2, 2)]

    @pytest.mark.parametrize(
        "sql",
        [
            "with v as (insert into t select a from v returning a) select a from v",
            "with v as (select a + 1 as a from v), x as (insert into t select a - 1 from v returning a) from x",
        ],
    )
    def test_data_modifying_cte(self, sql):
        con = duckdb.connect()
        con.execute("create table t(a int)")
        rel = con.sql("select 1 as a").query("v", sql)
        assert rel.fetchall() == [(1,)]
        assert con.sql("from t").fetchall() == [(1,)]

    def test_order_by_and_limit(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query(
            "v", "with v as (select a + g as a from v, range(5) r(g)) select a from v order by a desc limit 3"
        )
        assert rel.fetchall() == [(5,), (4,), (3,)]

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=LATE_BINDING)
    def test_sql_query_round_trip(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", "with v as (select a + 1 as a from v) select a from v")
        reuse_name(con, "v")
        assert con.sql(rel.sql_query()).fetchall() == [(2,)]

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=LATE_BINDING)
    def test_earlier_sibling_reads_the_source(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query(
            "v", "with w as (select a + 10 as a from v), v as (select a from w) select a from v"
        )
        reuse_name(con, "v")
        assert rel.fetchall() == [(11,)]

    def test_recursive_cte_anchor_reads_the_source(self):
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query(
            "v",
            "with recursive r(n) as (select a from v union all select n + 1 from r where n < 3) "
            "select n from r order by n",
        )
        reuse_name(con, "v")
        assert rel.fetchall() == [(1,), (2,), (3,)]

    def test_user_ctes_do_not_capture_the_source_tables(self):
        con = duckdb.connect()
        con.execute("create table t as select 1 as a")
        rel = con.sql("select * from t").query("v", "with t as (select 2 as a), v as (select * from v) select * from v")
        assert rel.fetchall() == [(1,)]

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=LATE_BINDING)
    def test_user_ctes_do_not_capture_the_source_tables_after_reuse(self):
        con = duckdb.connect()
        con.execute("create table t as select 1 as a")
        rel = con.sql("select * from t").query("v", "with t as (select 2 as a), v as (select * from v) select * from v")
        reuse_name(con, "v")
        assert rel.fetchall() == [(1,)]


class TestRelationsDerivedFromQuery:
    def test_chain_with_the_same_name(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")
        second = first.query("v", "select a + 1 as a from v")
        assert second.fetchall() == [(2,)]
        assert first.fetchall() == [(1,)]

    def test_chain_with_another_name(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")
        second = first.query("w", "select a + 1 as a from w")
        reuse_name(con, "v")
        assert second.fetchall() == [(2,)]

    def test_join_of_two_queries_with_the_same_name(self):
        con = duckdb.connect()
        left = con.sql("select 1 as a").query("v", "select a from v")
        right = con.sql("select 999 as a").query("v", "select a as b from v")
        assert left.cross(right).fetchall() == [(1, 999)]

    def test_join_of_two_queries_with_different_names(self):
        con = duckdb.connect()
        left = con.sql("select 1 as a").query("x", "select a from x")
        right = con.sql("select 999 as a").query("y", "select a as b from y")
        assert left.cross(right).fetchall() == [(1, 999)]

    def test_union_of_two_queries_with_the_same_name(self):
        con = duckdb.connect()
        left = con.sql("select 1 as a").query("v", "select a from v")
        right = con.sql("select 999 as a").query("v", "select a from v")
        assert sorted(left.union(right).fetchall()) == [(1,), (999,)]

    def test_filter_on_a_query(self):
        con = duckdb.connect()
        rel = con.sql("select * from range(3) t(a)").query("v", "select a from v").filter("a > 0")
        reuse_name(con, "v")
        assert rel.order("a").fetchall() == [(1,), (2,)]

    def test_query_relation_read_by_python_variable_name(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")  # noqa: F841
        reuse_name(con, "v")
        assert con.sql("select a from first").fetchall() == [(1,)]

    def test_view_created_from_a_query(self):
        con = duckdb.connect()
        con.sql("select 1 as a").query("v", "select a from v").create_view("w")
        reuse_name(con, "v")
        assert con.sql("select a from w").fetchall() == [(1,)]

    def test_sql_query_round_trip(self):
        con = duckdb.connect()
        first = con.sql("select 1 as a").query("v", "select a from v")
        reuse_name(con, "v")
        assert con.sql(first.sql_query()).fetchall() == [(1,)]


class TestQuerySources:
    def test_dataframe_source(self):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        rel = con.from_df(pd.DataFrame({"a": [1]})).query("v", "select a from v")
        con.from_df(pd.DataFrame({"a": [999]})).query("v", "select a from v")
        assert rel.fetchall() == [(1,)]

    def test_dataframe_source_read_twice(self):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        rel = con.from_df(pd.DataFrame({"a": [1, 2]})).query(
            "v", "select v.a, w.a as b from v, v as w union all select a, a from v order by all"
        )
        expected = [(1, 1), (1, 1), (1, 2), (2, 1), (2, 2), (2, 2)]
        assert rel.fetchall() == expected
        reuse_name(con, "v")
        assert rel.fetchall() == expected

    def test_dataframe_source_outlives_its_relation(self):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        rel = con.from_df(pd.DataFrame({"a": [1, 2]})).query("v", "select sum(a) from v")
        reuse_name(con, "v")
        gc.collect()
        assert rel.fetchall() == [(3,)]

    def test_python_object_named_in_user_sql(self):
        pd = pytest.importorskip("pandas")
        con = duckdb.connect()
        other = pd.DataFrame({"b": [7]})  # noqa: F841
        rel = con.sql("select 1 as a").query("v", "select a, b from v, other")
        reuse_name(con, "v")
        assert rel.fetchall() == [(1, 7)]

    def test_file_like_source_outlives_its_relation(self):
        pytest.importorskip("fsspec")
        con = duckdb.connect()
        rel = con.read_csv(io.BytesIO(b"a\n1\n2\n")).query("v", "select sum(a) from v")
        gc.collect()
        assert rel.fetchall() == [(3,)]

    def test_table_source_is_read_at_execution(self):
        con = duckdb.connect()
        con.execute("create table t as select 1 as a")
        rel = con.table("t").query("v", "select count(*) from v")
        con.execute("insert into t values (2)")
        assert rel.fetchall() == [(2,)]


class TestQueryKeepsItsShape:
    def test_duplicate_column_names(self):
        pytest.importorskip("pyarrow")
        con = duckdb.connect()
        rel = con.sql("select 1 as a").query("v", "select a, a from v")
        assert rel.columns == ["a", "a"]
        assert rel.to_arrow_table().column_names == ["a", "a"]

    def test_data_modifying_cte(self):
        con = duckdb.connect()
        con.execute("create table t(a int)")
        rel = con.sql("select 1 as a").query(
            "v", "with x as (insert into t select a from v returning a) select * from x"
        )
        assert rel.fetchall() == [(1,)]
        assert con.sql("from t").fetchall() == [(1,)]

    def test_source_is_not_materialized(self):
        con = duckdb.connect()
        rel = con.sql("select random() as r").query("v", "select v.r = w.r from v, v as w")
        assert "CTE" not in rel.explain()
        assert rel.fetchall() == [(False,)]


class TestQueryStillCreatesTheView:
    def test_view_is_temporary_and_readable(self):
        con = duckdb.connect()
        con.sql("select 1 as a").query("v", "select a from v")
        assert con.sql("select temporary from duckdb_views() where view_name = 'v'").fetchall() == [(True,)]
        assert con.sql("select a from v").fetchall() == [(1,)]

    def test_view_follows_the_latest_query(self):
        con = duckdb.connect()
        con.sql("select 1 as a").query("v", "select a from v")
        con.sql("select 999 as a").query("v", "select a from v")
        assert con.sql("select a from v").fetchall() == [(999,)]
