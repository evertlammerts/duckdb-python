"""The frame layer: building a query a step at a time.

Grouped by what can break rather than by method, because the interesting
failures are shared: a literal reaching the SQL text, a step that cannot follow
a WITH, a name that turns into syntax.
"""

from __future__ import annotations

import datetime
import gc
import pathlib
from typing import TYPE_CHECKING, cast

import pytest

import duckdb
from duckdb import _duckdb, col, declare, exceptions, lit, param, sql_expr, star
from duckdb.expr import ParamSink, Star, render_literal, suspended_sinks

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def con() -> duckdb.Connection:
    connection = duckdb.connect()
    connection.run(
        "CREATE TABLE orders AS SELECT * FROM (VALUES "
        "(1, 'nl', 120), (2, 'be', 80), (3, 'nl', 300), (4, 'de', 50), (5, 'nl', NULL)"
        ") v(id, country, amount)"
    )
    connection.run(
        "CREATE TABLE countries AS SELECT * FROM (VALUES ('nl', 'Netherlands'), ('be', 'Belgium')) v(code, label)"
    )
    return connection


@pytest.fixture
def orders(con: duckdb.Connection) -> duckdb.Frame:
    return duckdb.table("orders")


class TestGraph:
    """Steps become CTEs, and a step used twice is still computed once."""

    def test_a_single_step_needs_no_cte(self, orders: duckdb.Frame) -> None:
        assert orders.render() == 'SELECT * FROM "orders"'

    def test_each_step_becomes_a_cte(self, orders: duckdb.Frame) -> None:
        sql = orders.filter(col("amount") > 100).select(col("id")).render()
        assert sql.count(" AS (") == 2
        assert sql.startswith("WITH ")

    def test_a_reused_step_is_rendered_once(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # The whole point of walking by identity: the engine sees one scan of
        # the shared step, not two copies of its subtree.
        shared = orders.filter(col("country") == "nl")
        joined = shared.join(shared, on=col("l.id") == col("r.id"), suffix="_r")
        # A suffixed join needs to know which names clash, so it is rendered
        # with the resolution execution would use.
        sql = joined.render(con)
        assert sql.count("WHERE") == 1
        assert len(joined.fetchall(con)) == 3

    def test_a_self_join_is_unambiguous(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # Both sides name the same CTE, so without the l/r aliases the FROM
        # clause would not parse.
        pairs = orders.join(orders, on=col("l.id") == (col("r.id") - 1), suffix="_r").fetchall(con)
        assert len(pairs) == 4

    def test_a_frame_can_be_extended_twice_independently(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        base = orders.filter(col("amount").is_not_null())
        assert len(base.filter(col("country") == "nl").fetchall(con)) == 2
        assert len(base.filter(col("country") == "be").fetchall(con)) == 1


class TestLiteralsAreBound:
    """No value a caller supplied is ever written into the SQL text."""

    def test_a_string_filter_binds_rather_than_inlines(self, orders: duckdb.Frame) -> None:
        sql, values = orders.filter(col("country") == "nl")._sql_and_values()
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_a_quote_in_a_value_cannot_escape(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        hostile = "nl'; DROP TABLE orders; --"
        assert orders.filter(col("country") == hostile).fetchall(con) == []
        assert duckdb.table("orders").count(con) == 5

    def test_literals_inside_a_subquery_are_bound_too(self, orders: duckdb.Frame) -> None:
        inner = orders.filter(col("country") == "nl").select(col("id"))
        sql, values = orders.filter(col("id").isin(inner))._sql_and_values()
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_numbers_stay_in_the_text(self, orders: duckdb.Frame) -> None:
        # Nothing to escape, and inlining lets DuckDB type the literal itself.
        sql, values = orders.filter(col("amount") > 100)._sql_and_values()
        assert "100" in sql
        assert values is None


class TestSchema:
    """The binder answers the shape, without running anything."""

    def test_columns_and_types_come_from_the_binder(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.columns(con) == ["id", "country", "amount"]
        assert orders.types(con) == ["INTEGER", "VARCHAR", "INTEGER"]

    def test_a_derived_column_is_typed_by_the_binder(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # INTEGER times a decimal literal is DECIMAL, not DOUBLE. The client
        # does not work that out; it asks, which is the whole reason `.schema(con)`
        # goes to the binder.
        widened = orders.with_columns(doubled=col("amount") * 2.5)
        assert widened.types(con)[-1] == "DECIMAL(12,1)"

    def test_the_schema_is_asked_afresh(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # Deliberately not cached. What a source holds belongs to a catalog at
        # a moment, so it is asked again rather than remembered.
        assert orders.schema(con) == orders.schema(con)

    def test_a_bad_query_reports_the_engine_error(self, con: duckdb.Connection) -> None:
        with pytest.raises(exceptions.CatalogError):
            _ = duckdb.sql("SELECT * FROM missing").columns(con)


class TestProjection:
    def test_select_keeps_only_what_is_named(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.select(col("id"), col("country")).columns(con) == ["id", "country"]

    def test_a_bare_string_selects_a_column(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # The opposite of the rule inside an expression, where a string is a
        # value. Position decides, and nothing else a string could mean here.
        assert orders.select("id").columns(con) == ["id"]

    def test_select_refuses_anything_else(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="column name or expression"):
            orders.select(3.5)

    def test_star_can_exclude(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.select(star(exclude=["amount"])).columns(con) == ["id", "country"]

    def test_with_columns_appends_a_new_name(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        added = orders.with_columns(big=col("amount") > 100)
        assert added.columns(con) == ["id", "country", "amount", "big"]

    def test_with_columns_replaces_in_place(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # Replacing must keep the column where it was, so downstream positional
        # code does not silently shift.
        replaced = orders.with_columns(amount=col("amount") * 2)
        assert replaced.columns(con) == ["id", "country", "amount"]
        assert replaced.filter(col("id") == 1).fetchall(con) == [(1, "nl", 240)]

    def test_with_columns_can_add_and_replace_together(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        both = orders.with_columns(amount=col("amount") + 1, note=sql_expr("'x'"))
        assert both.columns(con) == ["id", "country", "amount", "note"]

    def test_drop_and_rename(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.drop("amount").columns(con) == ["id", "country"]
        assert orders.rename(country="iso").columns(con) == ["id", "iso", "amount"]

    def test_getitem_gives_one_column(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders["country"].columns(con) == ["country"]


class TestRows:
    def test_filter_accepts_an_expression_or_sql(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert len(orders.filter(col("country") == "nl").fetchall(con)) == 3
        assert len(orders.filter("country = 'nl'").fetchall(con)) == 3

    def test_sort_takes_direction_and_nulls(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        ordered = orders.sort(col("amount").desc().nulls_last()).fetchall(con)
        assert [row[0] for row in ordered] == [3, 1, 2, 4, 5]

    def test_limit_head_and_offset(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        ordered = orders.sort(col("id"))
        assert [r[0] for r in ordered.limit(2).fetchall(con)] == [1, 2]
        assert [r[0] for r in ordered.head(3).fetchall(con)] == [1, 2, 3]
        assert [r[0] for r in ordered.offset(3).fetchall(con)] == [4, 5]
        assert [r[0] for r in ordered.limit(2, offset=1).fetchall(con)] == [2, 3]

    def test_distinct_over_all_columns_and_over_keys(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert len(orders.select(col("country")).distinct().fetchall(con)) == 3
        assert len(orders.distinct(on="country").fetchall(con)) == 3

    def test_len_counts_without_fetching(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.count(con) == 5

    def test_iteration_crosses_batch_boundaries(self, con: duckdb.Connection) -> None:
        # __iter__ pulls 1024 rows at a time, so a shorter run would never test
        # the loop that refills.
        many = duckdb.sql("SELECT * FROM range(3000)")
        assert sum(1 for _ in many.rows(con)) == 3000

    def test_iterating_a_plan_without_a_connection_is_refused(self, con: duckdb.Connection) -> None:
        # A plan is not a sequence. Without __iter__ saying so, Python's old
        # protocol would call __getitem__ with 0, 1, 2 ... and never stop.
        with pytest.raises(TypeError, match="needs a connection"):
            list(duckdb.table("orders"))
        with pytest.raises(TypeError, match="not by int"):
            _ = duckdb.table("orders")[0]

    def test_fetchone_on_an_empty_result(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.filter(col("id") < 0).fetchone(con) is None


class TestAggregation:
    def test_aggregate_without_keys(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.aggregate(col("amount").sum().alias("total")).fetchall(con) == [(550,)]

    def test_group_keys_come_first(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        grouped = orders.group_by(col("country")).agg(col("amount").sum().alias("total"))
        assert grouped.columns(con) == ["country", "total"]
        assert sorted(grouped.fetchall(con)) == [("be", 80), ("de", 50), ("nl", 420)]

    def test_aggregate_takes_keys_directly(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        by_keyword = orders.aggregate(col("amount").sum().alias("total"), group_by="country")
        assert sorted(by_keyword.fetchall(con)) == [("be", 80), ("de", 50), ("nl", 420)]

    def test_filtering_after_agg_is_having(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # There is no `having` verb because there is no need for one: every
        # step is its own CTE, so a filter after an aggregate already runs
        # against the grouped rows.
        grouped = orders.group_by(col("country")).agg(col("amount").sum().alias("total"))
        assert grouped.filter(col("total") > 100).fetchall(con) == [("nl", 420)]

    def test_a_window_does_not_group(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        ranked = orders.with_columns(rank=duckdb.row_number().over(order_by=col("id")))
        assert len(ranked.fetchall(con)) == 5


class TestJoins:
    def test_join_on_a_shared_name(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        renamed = duckdb.table("countries").rename(code="country")
        assert len(orders.join(renamed, on="country").fetchall(con)) == 4

    def test_join_on_an_expression_names_the_sides(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        joined = orders.join(duckdb.table("countries"), on=col("l.country") == col("r.code"))
        assert len(joined.fetchall(con)) == 4

    def test_join_on_several_names(self, con: duckdb.Connection) -> None:
        left = duckdb.sql("SELECT 1 AS a, 2 AS b, 'x' AS v")
        right = duckdb.sql("SELECT 1 AS a, 2 AS b, 'y' AS w")
        assert left.join(right, on=["a", "b"]).fetchall(con) == [(1, 2, "x", "y")]

    @pytest.mark.parametrize(
        ("how", "expected"),
        [("inner", 4), ("left", 5), ("semi", 4), ("anti", 1), ("outer", 5)],
    )
    def test_join_kinds(self, orders: duckdb.Frame, con: duckdb.Connection, how: str, expected: int) -> None:
        joined = orders.join(duckdb.table("countries"), on=col("l.country") == col("r.code"), how=how)
        assert len(joined.fetchall(con)) == expected

    def test_cross_join_needs_no_keys(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        assert len(orders.cross(duckdb.table("countries")).fetchall(con)) == 10

    def test_a_keyed_join_without_keys_is_refused(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        with pytest.raises(TypeError, match="needs `on`"):
            orders.join(duckdb.table("countries"))


class TestSetOperations:
    def test_union_keeps_duplicates_unless_asked(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        nl = orders.filter(col("country") == "nl")
        assert len(nl.union(nl).fetchall(con)) == 6
        assert len(nl.union(nl, all=False).fetchall(con)) == 3

    def test_union_by_name_ignores_position(self, con: duckdb.Connection) -> None:
        left = duckdb.sql("SELECT 1 AS a, 2 AS b")
        right = duckdb.sql("SELECT 3 AS b, 4 AS a")
        assert sorted(left.union_by_name(right).fetchall(con)) == [(1, 2), (4, 3)]

    def test_intersect_and_except(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        nl = orders.filter(col("country") == "nl")
        low = orders.filter(col("amount") < 200)
        assert [r[0] for r in nl.intersect(low).fetchall(con)] == [1]
        assert sorted(r[0] for r in nl.except_(low).fetchall(con)) == [3, 5]


class TestSample:
    def test_a_row_count_is_exact(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert len(orders.sample(2, seed=7).fetchall(con)) == 2

    def test_a_seed_repeats(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.sample(3, seed=11).fetchall(con) == orders.sample(3, seed=11).fetchall(con)

    def test_a_percentage_stays_within_the_table(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert len(orders.sample(percent=50, seed=3).fetchall(con)) <= 5

    def test_a_method_can_be_chosen(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert len(orders.sample(percent=100, method="bernoulli", seed=1).fetchall(con)) == 5

    def test_a_size_is_required(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="either n or percent"):
            orders.sample()

    def test_two_sizes_are_refused(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="either n or percent"):
            orders.sample(2, percent=50)

    def test_a_method_name_cannot_be_syntax(self, orders: duckdb.Frame) -> None:
        # The method is a bare word in the SQL, so it cannot be quoted; it is
        # checked instead.
        with pytest.raises(ValueError, match="copy option name"):
            orders.sample(2, method="reservoir); DROP TABLE orders; --")


class TestReshaping:
    def test_unnest_expands_a_list(self, con: duckdb.Connection) -> None:
        rows = duckdb.sql("SELECT 1 AS k, [10, 20] AS xs").unnest("xs")
        assert rows.fetchall(con) == [(1, 10), (2 - 1, 20)]

    def test_unnest_keeps_the_column_where_it_was(self, con: duckdb.Connection) -> None:
        rows = duckdb.sql("SELECT [1, 2] AS xs, 'tail' AS t").unnest("xs")
        assert rows.columns(con) == ["xs", "t"]

    def test_unnesting_two_columns_walks_them_in_step(self, con: duckdb.Connection) -> None:
        rows = duckdb.sql("SELECT [1, 2] AS xs, ['a', 'b'] AS ys").unnest("xs", "ys")
        assert rows.fetchall(con) == [(1, "a"), (2, "b")]

    def test_unnest_needs_a_column(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="at least one column"):
            orders.unnest()

    def test_unpivot_folds_columns_into_rows(self, con: duckdb.Connection) -> None:
        wide = duckdb.sql("SELECT 'nl' AS country, 1 AS q1, 2 AS q2")
        long = wide.unpivot("q1", "q2", name="quarter", value="sales")
        assert long.fetchall(con) == [("nl", "q1", 1), ("nl", "q2", 2)]

    def test_unpivot_names_are_quoted(self, con: duckdb.Connection) -> None:
        wide = duckdb.sql("SELECT 1 AS q1")
        long = wide.unpivot("q1", name="the name", value="the value")
        assert long.columns(con) == ["the name", "the value"]

    def test_unpivot_needs_a_column(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="at least one column"):
            orders.unpivot()

    def test_reshaping_works_mid_chain(self, con: duckdb.Connection) -> None:
        # SUMMARIZE and UNPIVOT cannot follow a WITH, so both are wrapped in a
        # SELECT. Every step but the first is reached through a WITH, which is
        # what this checks.
        wide = duckdb.sql("SELECT 'nl' AS country, 1 AS q1, 2 AS q2").filter(col("country") == "nl")
        assert len(wide.unpivot("q1", "q2").filter(col("value") > 1).fetchall(con)) == 1


class TestInspection:
    def test_describe_reports_statistics(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        stats = orders.describe()
        assert stats.columns(con)[:2] == ["column_name", "column_type"]
        assert len(stats.fetchall(con)) == 3

    def test_describe_works_mid_chain(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert len(orders.filter(col("country") == "nl").describe().fetchall(con)) == 3

    def test_explain_returns_a_plan(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert "Seq Scan" in orders.explain(con) or "SEQ_SCAN" in orders.explain(con)

    def test_explain_analyze_runs_the_query(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert "Total Time" in orders.explain(con, analyze=True)

    def test_preview_draws_a_table(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        drawn = orders.sort(col("id")).preview(con)
        assert "id" in drawn
        assert "INTEGER" in drawn
        assert drawn.startswith("┌")
        assert "NULL" in drawn  # the missing amount, not an empty cell

    def test_preview_says_when_there_are_more_rows(self, con: duckdb.Connection) -> None:
        drawn = duckdb.sql("SELECT * FROM range(50)").preview(con, 3)
        assert "there are more" in drawn
        assert drawn.count("\n│") == 5  # heading, types, and three rows

    def test_preview_of_an_empty_frame_still_draws(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        drawn = orders.filter(col("id") < 0).preview(con)
        assert "id" in drawn
        assert "there are more" not in drawn

    def test_long_values_are_shortened(self, con: duckdb.Connection) -> None:
        drawn = duckdb.sql("SELECT repeat('x', 200) AS wide").preview(con)
        assert "…" in drawn
        assert max(len(line) for line in drawn.splitlines()) < 60

    def test_show_prints_the_preview(
        self, con: duckdb.Connection, orders: duckdb.Frame, capsys: pytest.CaptureFixture[str]
    ) -> None:
        orders.show(con, 2)
        assert capsys.readouterr().out.strip() == orders.preview(con, 2)

    def test_repr_shows_the_sql(self, con: duckdb.Connection) -> None:
        # A plan holds no connection, so there are no rows to show and nothing
        # to fail. Its repr is what it is: the query.
        assert repr(duckdb.sql("SELECT * FROM missing")) == "<Frame SELECT * FROM missing>"
        assert "lines" in repr(duckdb.table("orders").filter(col("id") > 1))


class TestSubqueries:
    def test_scalar_supplies_a_single_value(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        average = orders.aggregate(col("amount").mean().alias("m"))
        above = orders.filter(col("amount") > average.scalar())
        assert [row[0] for row in above.fetchall(con)] == [3]

    def test_isin_accepts_a_query(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        nl_ids = orders.filter(col("country") == "nl").select(col("id"))
        assert [row[0] for row in orders.filter(col("id").isin(nl_ids)).fetchall(con)] == [1, 3, 5]

    def test_isin_still_accepts_values(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert [row[0] for row in orders.filter(col("id").isin([1, 2])).fetchall(con)] == [1, 2]

    def test_a_subquery_is_a_step_of_the_plan(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # The plan a subquery refers to joins the graph: one WITH, and the
        # subquery renders as a reference to its step.
        inner = orders.filter(col("country") == "nl").sort(col("amount")).limit(1).select(col("id"))
        sql = orders.filter(col("id").isin(inner)).render()
        assert sql.count("WITH") == 1
        assert "IN (SELECT * FROM" in sql
        assert [row[0] for row in orders.filter(col("id").isin(inner)).fetchall(con)] == [1]


class TestSinks:
    def test_create_stores_the_rows(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.filter(col("country") == "nl").create(con, "nl") == 3
        assert duckdb.table("nl").count(con) == 3

    def test_create_refuses_to_clobber_unless_asked(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        orders.create(con, "copy")
        with pytest.raises(exceptions.CatalogError):
            orders.create(con, "copy")
        assert orders.filter(col("id") == 1).create(con, "copy", replace=True) == 1
        assert duckdb.table("copy").count(con) == 1

    def test_a_temporary_table_is_not_in_the_catalog(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        orders.create(con, "scratch", temporary=True)
        assert duckdb.table("scratch").count(con) == 5
        listed = duckdb.sql("SELECT temporary FROM duckdb_tables() WHERE table_name = 'scratch'").fetchall(con)
        assert listed == [(True,)]

    def test_insert_into_appends(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        orders.filter(col("id") == 1).create(con, "some")
        assert orders.filter(col("id") == 2).insert_into(con, "some") == 1
        assert duckdb.table("some").count(con) == 2

    def test_a_sink_binds_its_literals(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # The COPY wrapper must not break the sink: the filter's value still
        # has to be bound rather than written into the statement.
        sql, values = orders.filter(col("country") == "nl")._sql_and_values(lambda q: f"CREATE TABLE t AS {q}")
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_to_parquet_round_trips(self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path) -> None:
        path = tmp_path / "orders.parquet"
        assert orders.to_parquet(con, str(path)) == 5
        assert path.exists()
        assert duckdb.sql(f"SELECT * FROM read_parquet('{path}')").count(con) == 5

    def test_to_csv_round_trips(self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path) -> None:
        path = tmp_path / "orders.csv"
        assert orders.to_csv(con, str(path), header=True) == 5
        assert path.read_text().startswith("id,country,amount")

    def test_copy_options_reach_the_writer(self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path) -> None:
        path = tmp_path / "orders.csv"
        orders.to_csv(con, str(path), header=False, delimiter="|")
        assert "|" in path.read_text()
        assert not path.read_text().startswith("id")

    def test_an_option_name_that_is_not_a_name_is_refused(
        self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path
    ) -> None:
        # Option names are bare words in the SQL and cannot be quoted, so the
        # only defence is to check them.
        with pytest.raises(ValueError, match="copy option name"):
            orders.to_csv(con, str(tmp_path / "x.csv"), **{"header, ROW_GROUP_SIZE": 1})  # type: ignore[arg-type]

    def test_a_quote_in_a_path_cannot_escape(
        self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path
    ) -> None:
        path = tmp_path / "od'; DROP TABLE orders; --.csv"
        orders.to_csv(con, str(path))
        assert path.exists()
        assert duckdb.table("orders").count(con) == 5


class TestTableSource:
    def test_a_table_is_read_by_name(self, con: duckdb.Connection) -> None:
        assert duckdb.table("orders").count(con) == 5

    def test_a_name_is_quoted_not_spliced(self, con: duckdb.Connection) -> None:
        # The reason this verb exists: an f-string into sql() would run the
        # second statement.
        with pytest.raises(exceptions.CatalogError):
            duckdb.table("orders; DROP TABLE orders").fetchall(con)
        assert duckdb.table("orders").count(con) == 5

    def test_a_qualified_name_stays_two_identifiers(self, con: duckdb.Connection) -> None:
        assert duckdb.table("main.orders").count(con) == 5

    def test_an_awkward_name_needs_no_escaping_by_the_caller(self, con: duckdb.Connection) -> None:
        con.run('CREATE TABLE "select ""x""" AS SELECT 1 AS v')
        assert duckdb.table('select "x"').fetchall(con) == [(1,)]


class TestDerivedNamesMatchTheEngine:
    """Every derived shape must equal what the binder would have said.

    Deriving names is the whole point of the shape layer, and drifting from the
    engine is the one way it can go wrong. Each case here derives a shape and
    then asks the engine the same question, so a rule that is subtly wrong
    fails rather than silently disagreeing.
    """

    def frames(self, con: duckdb.Connection) -> dict[str, duckdb.Frame]:
        orders, countries = duckdb.table("orders"), duckdb.table("countries")
        lists = duckdb.sql("SELECT 1 AS k, [10, 20] AS xs")
        wide = duckdb.sql("SELECT 'nl' AS c, 1 AS q1, 2 AS q2")
        return {
            "table": orders,
            "sql": duckdb.sql("SELECT 1 AS a, 'b' AS b"),
            "filter": orders.filter(col("amount") > 100),
            "select names": orders.select("id", "country"),
            "select exprs": orders.select(col("id"), col("amount").alias("total")),
            "select star": orders.select(star()),
            "star exclude": orders.select(star(exclude=["amount"])),
            "star rename": orders.select(star(rename={"amount": "value"})),
            "with_columns add": orders.with_columns(big=col("amount") > 100),
            "with_columns replace": orders.with_columns(amount=col("amount") * 2),
            "with_columns both": orders.with_columns(amount=col("amount") + 1, extra=col("id")),
            "drop": orders.drop("amount"),
            "rename": orders.rename(country="iso"),
            "sort": orders.sort(col("id")),
            "limit": orders.limit(2),
            "offset": orders.offset(1),
            "distinct": orders.distinct(),
            "distinct on": orders.distinct(on="country"),
            "sample": orders.sample(1, seed=1),
            "aggregate": orders.aggregate(col("amount").sum().alias("total")),
            "grouped": orders.group_by(col("country")).agg(col("amount").sum().alias("total")),
            "grouped alias": orders.group_by(col("country").alias("iso")).agg(col("id").count().alias("n")),
            "join on expr": orders.join(countries, on=col("l.country") == col("r.code")),
            "join using": orders.rename(country="code").join(countries, on="code"),
            "join suffix": orders.join(orders, on=col("l.id") == col("r.id"), suffix="_r"),
            "join semi": orders.join(countries, on=col("l.country") == col("r.code"), how="semi"),
            "cross": orders.cross(countries),
            "union": orders.union(orders),
            "intersect": orders.intersect(orders),
            "except": orders.except_(orders),
            "unnest": lists.unnest("xs"),
            "unpivot": wide.unpivot("q1", "q2", name="quarter", value="sales"),
            "describe": orders.describe(),
            "deep chain": orders.filter(col("id") > 0).select("id", "amount").sort(col("id")).limit(3),
            "join then verbs": orders.join(countries, on=col("l.country") == col("r.code"))
            .drop("code")
            .rename(label="name"),
        }

    def test_every_verb_derives_what_the_engine_reports(self, con: duckdb.Connection) -> None:
        wrong = {}
        for label, frame in self.frames(con).items():
            derived = frame.columns(con)
            truth = [name for name, _ in whole_bind(frame, con)]
            if derived != truth:
                wrong[label] = (derived, truth)
        assert not wrong, f"derived shape disagrees with the binder: {wrong}"

    def test_known_types_agree_with_the_engine(self, con: duckdb.Connection) -> None:
        wrong = {}
        for label, frame in self.frames(con).items():
            truth = dict(whole_bind(frame, con))
            for column in frame.resolve(con):
                if column.type is not None and truth.get(column.name) != column.type:
                    wrong[f"{label}.{column.name}"] = (column.type, truth.get(column.name))
        assert not wrong, f"carried type disagrees with the binder: {wrong}"


def _argument_for(type_text: str) -> object:
    """A stand-in value of a SQL type, for calling a generated method."""
    upper = type_text.upper()
    if upper.endswith("[]"):
        return lit([1])
    if upper in {"VARCHAR", "ANY"}:
        return "a"
    if "INT" in upper:
        return 1
    if upper in {"DOUBLE", "FLOAT", "DECIMAL"}:
        return 1.0
    if upper == "BOOLEAN":
        return True
    if upper == "INTERVAL":
        return datetime.timedelta(days=1)
    if upper == "DATE":
        return datetime.date(2026, 1, 1)
    if upper.startswith("TIMESTAMP"):
        return datetime.datetime(2026, 1, 1, 12, 0)
    if upper.startswith("TIME"):
        return datetime.time(12, 0)
    if upper == "BLOB":
        return b"a"
    return sql_expr(f"NULL::{type_text}")


def whole_bind(frame: duckdb.Frame, con: duckdb.Connection) -> list[tuple[str, str]]:
    """The engine's own answer for the whole query, to check derivations against."""
    with suspended_sinks():
        output, _ = con._engine().bind(frame.render(con))
    return list(output)


def record_binds(con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every query put to the binder from here on."""
    seen: list[str] = []
    engine = type(con._engine())
    original = engine.bind

    def recording(self: _duckdb.Connection, sql: str) -> object:
        seen.append(sql)
        return original(self, sql)

    monkeypatch.setattr(engine, "bind", recording)
    return seen


class TestTheEngineIsAskedSparingly:
    """Names are derived here, so the binder is asked once per source."""

    def test_building_a_frame_asks_nothing(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = record_binds(con, monkeypatch)
        duckdb.table("orders").filter(col("amount") > 100).select("id").sort(col("id"))
        assert calls == [], "building must not reach the engine"

    def test_columns_costs_one_bind_per_source(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = record_binds(con, monkeypatch)
        joined = (
            duckdb.table("orders")
            .filter(col("amount") > 100)
            .join(duckdb.table("countries"), on=col("l.country") == col("r.code"))
            .select("id", "label")
        )
        assert joined.columns(con) == ["id", "label"]
        assert len(calls) == 2, f"one per source, got {len(calls)}: {calls}"

    def test_types_come_along_with_the_source(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = duckdb.table("orders").filter(col("amount") > 100).select("id")
        calls = record_binds(con, monkeypatch)
        # A column carried through untouched keeps the type the source's own
        # reflection gave it, so types cost no more than names do.
        assert frame.types(con) == ["INTEGER"]
        assert calls == ['SELECT * FROM "orders"']

    def test_a_declared_source_is_asked_nothing(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        # A schema the caller stated is not a memory of an engine, so it needs
        # no connection and no reflection.
        stated = declare("orders", [("id", "INTEGER"), ("amount", "INTEGER")])
        calls = record_binds(con, monkeypatch)
        assert stated.filter(col("amount") > 0).types(con) == ["INTEGER", "INTEGER"]
        assert calls == []

    def test_an_engine_named_column_is_bound_on_a_stub(
        self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        duckdb.table("orders").columns(con)  # pay for the source first
        calls = record_binds(con, monkeypatch)
        # Nothing here knows what DuckDB will call this column, so it asks. The
        # question it asks is this step over an empty input, not the whole query.
        frame = duckdb.table("orders").select(sql_expr("amount * 2"))
        assert frame.columns(con) == ["(amount * 2)"]
        assert len(calls) == 2, calls  # the fresh source, then the stub
        assert "WHERE FALSE" in calls[-1]
        assert "NULL::INTEGER" in calls[-1]

    def test_a_stub_does_not_grow_with_the_chain(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        deep = duckdb.table("orders")
        for _ in range(8):
            deep = deep.filter(col("amount") > 0)
        deep.columns(con)
        calls = record_binds(con, monkeypatch)
        assert deep.select(sql_expr("amount * 2")).columns(con) == ["(amount * 2)"]
        # One question for the source, then the stub. The point of the stub is
        # that it is one step's worth of SQL however deep the chain.
        stub = calls[-1]
        assert stub.count("SELECT") == 2, stub
        assert "WHERE FALSE" in stub

    def test_a_stub_answer_is_remembered_by_its_text(
        self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A stub names no table: it carries its whole input inline, so its
        # answer is a function of the text and worth keeping. A source is the
        # opposite, and is asked again every time.
        plan = duckdb.table("orders").select(sql_expr("amount * 2"))
        assert plan.columns(con) == ["(amount * 2)"]
        calls = record_binds(con, monkeypatch)
        assert plan.columns(con) == ["(amount * 2)"]
        assert calls == ['SELECT * FROM "orders"'], "the source is re-read, the stub is not"


class TestJoinRefusesToDuplicateAName:
    """A join carries both sides through, so a shared name would appear twice.

    DuckDB does not catch that once the join is behind a step: it binds the
    later reference to whichever came first and answers. That silent wrong
    answer is what this refuses.
    """

    def test_a_shared_name_is_refused(self, con: duckdb.Connection) -> None:
        # A plan is built without asking anything, so the clash is reported
        # when the columns are worked out, which is also before anything runs.
        joined = duckdb.table("orders").join(duckdb.table("orders"), on=col("l.id") == col("r.id"))
        with pytest.raises(ValueError, match="both sides of this join have"):
            joined.columns(con)

    def test_the_message_names_the_columns(self, con: duckdb.Connection) -> None:
        joined = duckdb.table("orders").join(duckdb.table("orders"), on=col("l.id") == col("r.id"))
        with pytest.raises(ValueError, match=r"'id'.*'country'.*'amount'"):
            joined.columns(con)

    def test_a_suffix_renames_the_right_side(self, con: duckdb.Connection) -> None:
        joined = duckdb.table("orders").join(duckdb.table("orders"), on=col("l.id") == col("r.id"), suffix="_r")
        assert joined.columns(con) == ["id", "country", "amount", "id_r", "country_r", "amount_r"]
        assert joined.columns(con) == [name for name, _ in whole_bind(joined, con)]

    def test_a_suffixed_column_is_reachable(self, con: duckdb.Connection) -> None:
        joined = duckdb.table("orders").join(
            duckdb.table("orders"), on=col("l.id") == (col("r.id") - 1), suffix="_next"
        )
        pairs = joined.select("id", "id_next").sort(col("id")).fetchall(con)
        assert pairs == [(1, 2), (2, 3), (3, 4), (4, 5)]

    def test_a_using_key_is_not_a_clash(self, con: duckdb.Connection) -> None:
        # USING folds its key into one column, so it cannot appear twice.
        left = duckdb.table("orders").select("id", "country")
        right = duckdb.table("countries").rename(code="country")
        assert left.join(right, on="country").columns(con) == ["id", "country", "label"]

    def test_a_semi_join_keeps_only_the_left(self, con: duckdb.Connection) -> None:
        # Nothing from the right survives, so nothing can collide.
        joined = duckdb.table("orders").join(duckdb.table("orders"), on=col("l.id") == col("r.id"), how="semi")
        assert joined.columns(con) == ["id", "country", "amount"]

    def test_disjoint_sides_need_no_suffix(self, con: duckdb.Connection) -> None:
        joined = duckdb.table("orders").join(duckdb.table("countries"), on=col("l.country") == col("r.code"))
        assert joined.columns(con) == ["id", "country", "amount", "code", "label"]
        assert "RENAME" not in joined.render()


class TestReviewRoundTwo:
    """Regressions for the fifteen findings of the second review.

    Each of these produced a wrong answer or a silent surprise. The tests
    assert rows and shapes rather than SQL text wherever the bug was that the
    answer came out wrong.
    """

    def test_aggregate_binds_its_literals(self, orders: duckdb.Frame) -> None:
        # Finding 1. Every other verb defers rendering into the parameter sink;
        # aggregate rendered at call time and wrote literals into the text.
        sql, values = orders.aggregate((col("country") == "nl").sum().alias("n"))._sql_and_values()
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_grouped_agg_binds_its_literals(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        grouped = orders.group_by(col("country")).agg((col("amount") > 100).sum().alias("big"))
        assert "'nl'" not in grouped.render()
        assert sorted(grouped.fetchall(con)) == [("be", 0), ("de", 0), ("nl", 2)]

    def test_a_blob_literal_survives_both_paths(self, con: duckdb.Connection) -> None:
        # Findings 1 and 11 together. The blob rendered as one escape over every
        # hex digit, so the schema-oracle path disagreed with the bound path.
        con.run("CREATE TABLE blobs AS SELECT unhex('AABBCC') AS v")
        blobs, needle = duckdb.table("blobs"), bytes([0xAA, 0xBB, 0xCC])
        assert blobs.filter(col("v") == needle).fetchall(con) == [(needle,)]
        assert blobs.aggregate((col("v") == needle).sum().alias("n")).fetchall(con) == [(1,)]

    def test_a_blob_literal_is_escaped_per_byte(self) -> None:
        assert render_literal(bytes([0xAA, 0xBB, 0xCC])) == r"'\xaa\xbb\xcc'::BLOB"

    def test_one_plan_runs_on_any_connection(self, con: duckdb.Connection) -> None:
        # Finding 2 dissolved too. A plan named no connection, so combining two
        # of them cannot mix databases; the same plan just runs on each.
        plan = duckdb.table("orders").filter(col("amount") > 100).select("id")
        assert plan.fetchall(con) == [(1,), (3,)]
        elsewhere = duckdb.connect()
        elsewhere.run("CREATE TABLE orders AS SELECT 99 AS id, 'zz' AS country, 900 AS amount")
        assert plan.fetchall(elsewhere) == [(99,)]

    def test_user_sql_may_contain_braces(self, con: duckdb.Connection) -> None:
        # Finding 4. Bodies were format templates, so DuckDB's own struct
        # syntax was read as a format field.
        assert duckdb.sql("SELECT {'a': 1, 'b': 2} AS s").fetchall(con) == [({"a": 1, "b": 2},)]

    def test_a_table_name_may_contain_braces(self, con: duckdb.Connection) -> None:
        con.run('CREATE TABLE "weird{0}name" AS SELECT 1 AS v')
        assert duckdb.table("weird{0}name").fetchall(con) == [(1,)]

    def test_a_column_name_may_contain_braces(self, con: duckdb.Connection) -> None:
        con.run('CREATE TABLE braces AS SELECT 1 AS "a{0}b", 2 AS keep')
        assert duckdb.table("braces").drop("a{0}b").columns(con) == ["keep"]

    def test_closing_releases_the_database(self, con: duckdb.Connection, tmp_path: Path) -> None:
        # Finding 5, which the decoupling dissolved rather than fixed: a plan
        # holds nothing, so there is nothing left to keep the file open.
        path = str(tmp_path / "held.db")
        con = duckdb.connect(path)
        con.run("CREATE TABLE t AS SELECT 1 AS v")
        plan = duckdb.table("t")
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            plan.fetchall(con)
        del con
        gc.collect()
        # The plan is still perfectly good; it just needs a connection.
        reopened = duckdb.connect(path)
        assert plan.fetchall(reopened) == [(1,)]

    def test_with_columns_refuses_a_list(self, orders: duckdb.Frame) -> None:
        # Finding 6. A list was read as positional arguments and only its first
        # element survived, so `tags` became a copy of the column named 'x'.
        with pytest.raises(TypeError, match="wrap a value in lit"):
            orders.with_columns(tags=["id", "country"])

    def test_with_columns_takes_a_list_through_lit(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        rows = orders.with_columns(tags=lit(["a", "b"])).select("tags").fetchone(con)
        assert rows == (["a", "b"],)

    def test_a_suffix_that_would_still_collide_is_refused(self, con: duckdb.Connection) -> None:
        # Finding 7. Only the raw right-hand names were checked, never the
        # renamed ones, which is the collision the check exists to prevent.
        con.run("CREATE TABLE lhs AS SELECT 1 AS id, 10 AS amount, 99 AS amount_r")
        con.run("CREATE TABLE rhs AS SELECT 1 AS id, 20 AS amount")
        with pytest.raises(ValueError, match="'amount_r' more than once"):
            duckdb.table("lhs").join(duckdb.table("rhs"), on="id", suffix="_r").columns(con)
        assert duckdb.table("lhs").join(duckdb.table("rhs"), on="id", suffix="_b").columns(con) == [
            "id",
            "amount",
            "amount_r",
            "amount_b",
        ]

    def test_isin_refuses_a_bare_string(self) -> None:
        # Finding 8. list("US") is ['U', 'S'], so the membership test silently
        # became one over single characters.
        with pytest.raises(TypeError, match="for one value use =="):
            col("code").isin("US")

    def test_isin_still_takes_a_list(self, con: duckdb.Connection) -> None:
        assert duckdb.table("orders").filter(col("country").isin(["nl", "de"])).columns(con) == [
            "id",
            "country",
            "amount",
        ]

    def test_an_unsupplied_parameter_is_refused(self, orders: duckdb.Frame) -> None:
        # Finding 9. Every reference became None, so a query using param()
        # matched nothing and said nothing.
        with pytest.raises(ValueError, match="no value for parameter 'needle'"):
            orders.filter(col("country") == param("needle"))._sql_and_values()

    @pytest.mark.parametrize(
        ("verb", "call"),
        [
            ("rename", lambda f: f.rename(id="x", country="x")),
            ("select", lambda f: f.select(col("id"), col("id"))),
            ("select", lambda f: f.select(col("id").alias("k"), col("country").alias("k"))),
            ("aggregate", lambda f: f.group_by(col("country")).agg(col("id").count().alias("country"))),
        ],
    )
    def test_a_verb_may_not_produce_one_name_twice(
        self, con: duckdb.Connection, orders: duckdb.Frame, verb: str, call: Callable[[duckdb.Frame], duckdb.Frame]
    ) -> None:
        # Finding 10. The engine will not catch this once the step is behind a
        # WITH: it binds a later reference to whichever came first.
        with pytest.raises(ValueError, match=f"{verb} would produce"):
            call(orders).columns(con)

    def test_an_expression_has_no_truth_value(self) -> None:
        # Finding 12. `==` builds a node, so an expression is always truthy and
        # `in`, `if` and `and` all quietly did the wrong thing.
        with pytest.raises(TypeError, match="no truth value"):
            bool(col("a") == 1)
        with pytest.raises(TypeError, match="no truth value"):
            _ = col("a") in [col("b")]

    def test_presentation_does_not_alias_the_original(self) -> None:
        # Finding 13. _with shared list and dict fields with every clone.
        first = cast("Star", star(exclude=["a"]))
        second = cast("Star", first.alias("renamed"))
        second.exclude.append("b")
        assert first.exclude == ["a"]

    def test_run_closes_its_result_when_the_statement_fails(self, con: duckdb.Connection) -> None:
        # Finding 14. Without try/finally the failed result stayed open, and
        # one live result per connection blocks whatever comes next.
        con.run("CREATE TABLE unique_v (v INTEGER PRIMARY KEY)")
        con.run("INSERT INTO unique_v VALUES (1)")
        with pytest.raises(exceptions.Error):
            con.run("INSERT INTO unique_v VALUES (1)")
        assert con.run("INSERT INTO unique_v VALUES (2)") == 1
        assert duckdb.table("unique_v").count(con) == 2

    def test_a_very_long_chain_does_not_overflow_the_stack(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # Finding 15. _order() recursed, so a pipeline built in a loop failed
        # at a few hundred steps. Every render and schema lookup goes through it.
        deep = orders
        for _ in range(3000):
            deep = deep.filter(col("amount") > 0)
        assert deep.columns(con) == ["id", "country", "amount"]
        assert deep.render().count("WITH") == 1

    def test_step_order_is_unchanged_by_the_iterative_walk(self, con: duckdb.Connection) -> None:
        # The rewrite had to keep the visit order, because step numbering and
        # therefore parameter order follow it.
        left = duckdb.table("orders").filter(col("country") == "nl")
        right = duckdb.table("countries").filter(col("code") == "be")
        _sql, values = left.join(right, on=col("l.country") == col("r.code"))._sql_and_values()
        assert values == ["nl", "be"], "inputs must still be visited left to right"


class TestAPlanHoldsNothing:
    """A plan is a plan. It names no connection and no database.

    This is what makes the same object a macro body, a query on one
    connection, and a query on another. It is also why `close()` closes and
    why two plans can never disagree about which database they mean: there is
    nothing on a plan to disagree with.
    """

    def test_a_plan_holds_no_connection(self) -> None:
        plan = duckdb.table("orders").filter(col("amount") > 100).select("id")
        held = {name for name in vars(plan) if not name.startswith("__")}
        assert held == {"_body", "_inputs", "_uses", "_shape_rule", "_declared", "_aliases"}

    def test_a_plan_renders_with_no_engine(self) -> None:
        plan = duckdb.table("orders").filter(col("amount") > 100).select("id")
        rendered = plan.render()
        assert 'FROM "orders"' in rendered
        assert "SELECT" in rendered

    def test_literals_still_bind_with_no_engine(self) -> None:
        sql, values = duckdb.table("orders").filter(col("country") == "nl")._sql_and_values()
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_one_plan_two_databases(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").filter(col("amount") > 100).select("id").sort(col("id"))
        assert plan.fetchall(con) == [(1,), (3,)]
        elsewhere = duckdb.connect()
        elsewhere.run("CREATE TABLE orders AS SELECT 42 AS id, 'zz' AS country, 900 AS amount")
        assert plan.fetchall(elsewhere) == [(42,)]

    def test_a_declared_source_resolves_with_no_engine(self) -> None:
        # The memo's `template(sql, schema)` idea, as an ordinary option: say
        # what a source holds and the whole plan works out its columns alone.
        orders = declare("orders", [("id", "INTEGER"), ("amount", "INTEGER")])
        plan = orders.filter(col("amount") > 0).with_columns(doubled=col("amount")).select("id", "doubled")
        assert plan.columns() == ["id", "doubled"]

    def test_a_plan_is_a_macro_body(self, con: duckdb.Connection) -> None:
        # The reason binding had to become optional. A macro body references
        # parameters that do not exist until it is called, so nothing can bind
        # it, and nothing needs to.
        body = duckdb.sql('SELECT unnest("i") AS uid').filter(col("uid") > 2)
        con.run("CREATE MACRO big_ids(i) AS TABLE " + body.render())
        assert duckdb.sql("SELECT * FROM big_ids([1, 3, 9])").fetchall(con) == [(3,), (9,)]

    def test_the_same_macro_body_on_another_connection(self, con: duckdb.Connection) -> None:
        body = duckdb.sql('SELECT unnest("i") AS uid').filter(col("uid") > 2)
        elsewhere = duckdb.connect()
        for connection in (con, elsewhere):
            connection.run("CREATE MACRO big_ids(i) AS TABLE " + body.render())
        assert duckdb.sql("SELECT * FROM big_ids([5])").fetchall(elsewhere) == [(5,)]

    def test_asking_for_columns_without_a_connection_says_so(self) -> None:
        plan = duckdb.table("orders").filter(col("amount") > 0)
        with pytest.raises(ValueError, match="needs a connection"):
            plan.columns()

    def test_running_needs_a_connection(self) -> None:
        with pytest.raises(TypeError):
            duckdb.table("orders").fetchall()  # type: ignore[call-arg]


class TestNothingAnEngineSaidIsKept:
    """A plan never remembers what a connection told it.

    A schema is one catalog's answer at one moment. A plan that kept it would
    render by whatever it was told first: right on the connection that
    answered, and quietly wrong on the next one and after any DDL. Every case
    here failed before, and all three failed silently.
    """

    def two_databases(self) -> tuple[duckdb.Connection, duckdb.Connection]:
        """The same table name, different columns. One schema twice cannot fail."""
        first, second = duckdb.connect(), duckdb.connect()
        first.run("CREATE TABLE orders AS SELECT 1 AS id, 10 AS amount")
        second.run("CREATE TABLE orders AS SELECT 1 AS id, 10 AS amount, 99 AS x")
        return first, second

    def test_one_plan_two_schemas_replaces_on_one_appends_on_the_other(self) -> None:
        first, second = self.two_databases()
        plan = duckdb.table("orders").with_columns(x=col("amount") * 2)
        # `x` is absent in the first, so it is appended; present in the second,
        # so it is replaced in place. Both are three columns.
        assert plan.columns(first) == ["id", "amount", "x"]
        assert plan.fetchall(first) == [(1, 10, 20)]
        assert plan.columns(second) == ["id", "amount", "x"]
        assert plan.fetchall(second) == [(1, 10, 20)]

    def test_the_order_of_the_two_does_not_matter(self) -> None:
        first, second = self.two_databases()
        plan = duckdb.table("orders").with_columns(x=col("amount") * 2)
        assert plan.fetchall(second) == [(1, 10, 20)]
        assert plan.fetchall(first) == [(1, 10, 20)]

    def test_a_plan_read_on_one_connection_is_right_on_another(self) -> None:
        # The memo's case: reflect through one connection, run on a second
        # whose schema differs, and be right on the second.
        first, second = self.two_databases()
        plan = duckdb.table("orders").with_columns(x=col("amount") * 2)
        assert plan.schema(first) == [("id", "INTEGER"), ("amount", "INTEGER"), ("x", "INTEGER")]
        assert plan.fetchall(second) == [(1, 10, 20)], "the first connection must not decide this"

    def test_the_join_guard_runs_on_every_connection(self) -> None:
        # The guard exists to refuse two columns of one name. Resolved once and
        # remembered, it simply never ran on the second connection.
        clear, clashing = duckdb.connect(), duckdb.connect()
        for connection, right in ((clear, "SELECT 1 AS id, 20 AS y"), (clashing, "SELECT 1 AS id, 20 AS x")):
            connection.run("CREATE TABLE l AS SELECT 1 AS id, 10 AS x")
            connection.run(f"CREATE TABLE r AS {right}")
        joined = duckdb.table("l").join(duckdb.table("r"), on="id")
        assert joined.columns(clear) == ["id", "x", "y"]
        with pytest.raises(ValueError, match="both sides of this join have 'x'"):
            joined.columns(clashing)
        with pytest.raises(ValueError, match="both sides of this join have 'x'"):
            joined.fetchall(clashing)

    def test_ddl_on_the_same_connection_is_seen(self, con: duckdb.Connection) -> None:
        con.run("CREATE TABLE t AS SELECT 1 AS a")
        plan = duckdb.table("t").with_columns(b=col("a") + 1)
        assert plan.fetchall(con) == [(1, 2)]
        con.run("ALTER TABLE t ADD COLUMN b INTEGER DEFAULT 7")
        # `b` now exists, so the same plan replaces where it used to append.
        assert plan.columns(con) == ["a", "b"]
        assert plan.fetchall(con) == [(1, 2)]

    def test_a_dropped_column_is_seen_too(self, con: duckdb.Connection) -> None:
        con.run("CREATE TABLE t AS SELECT 1 AS a, 5 AS b")
        plan = duckdb.table("t").with_columns(b=col("a") + 1)
        assert plan.fetchall(con) == [(1, 2)]
        con.run("ALTER TABLE t DROP COLUMN b")
        assert plan.fetchall(con) == [(1, 2)]

    def test_a_stated_schema_is_not_a_memory(self, con: duckdb.Connection) -> None:
        # Declaring is the caller speaking, not the engine being remembered, so
        # it holds wherever the plan goes and is the caller's to keep true.
        stated = declare("orders", [("id", "INTEGER"), ("amount", "INTEGER")])
        plan = stated.with_columns(doubled=col("amount") * 2)
        assert plan.columns() == ["id", "amount", "doubled"]
        first, second = self.two_databases()
        assert plan.columns(first) == ["id", "amount", "doubled"]
        assert plan.columns(second) == ["id", "amount", "doubled"]


class TestRenderIsTotal:
    """O1. Rendering never needs a connection, except for one join form."""

    def test_with_columns_renders_blind(self) -> None:
        plan = duckdb.table("orders").with_columns(doubled=col("amount") * 2)
        sql = plan.render()
        assert "COLUMNS(lambda c: c NOT IN ('doubled'))" in sql
        assert repr(plan).startswith("<Frame WITH")

    def test_the_blind_form_gives_the_same_rows(self, con: duckdb.Connection) -> None:
        # Blind, a set column that already existed moves to the end; that is
        # the price of not knowing. The values are the same either way.
        plan = duckdb.table("orders").with_columns(country=col("country").str.upper(), big=col("amount") > 100)
        resolved = plan.fetchall(con)
        blind = duckdb.sql(plan.render()).fetchall(con)
        assert plan.columns(con) == ["id", "country", "amount", "big"]
        assert duckdb.sql(plan.render()).columns(con) == ["id", "amount", "country", "big"]
        assert sorted(sorted(map(str, row)) for row in blind) == sorted(sorted(map(str, row)) for row in resolved)

    def test_a_computed_column_can_be_a_macro_body(self, con: duckdb.Connection) -> None:
        body = duckdb.sql('SELECT unnest("i") AS n').with_columns(double=col("n") * 2)
        con.run("CREATE MACRO doubled(i) AS TABLE " + body.render())
        assert duckdb.sql("SELECT * FROM doubled([1, 2])").fetchall(con) == [(1, 2), (2, 4)]

    def test_a_suffixed_join_is_the_one_step_that_needs_a_connection(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").join(duckdb.table("orders"), on="id", suffix="_r")
        with pytest.raises(ValueError, match="needs a connection to render"):
            plan.render()
        assert repr(plan).startswith("<Frame, renders with a connection:")
        assert "RENAME" in plan.render(con)

    def test_an_unsuffixed_join_renders_blind(self) -> None:
        assert "JOIN" in duckdb.table("a").join(duckdb.table("b"), on="id").render()


class TestJoinKindIsAClosedSet:
    """O2. `how` can never carry text into the statement."""

    def test_a_typo_is_refused_at_the_call(self) -> None:
        with pytest.raises(ValueError, match="unknown join kind 'innner'"):
            duckdb.table("a").join(duckdb.table("b"), on="id", how="innner")

    def test_text_cannot_ride_in(self) -> None:
        with pytest.raises(ValueError, match="unknown join kind"):
            duckdb.table("a").join(duckdb.table("b"), on="id", how="inner JOIN evil ON true --")

    def test_case_does_not_matter(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").join(duckdb.table("countries"), on=col("l.country") == col("r.code"), how="LEFT")
        assert plan.count(con) == 5


class TestOnlyAPlanIsASubquery:
    """O3. A `render` method is not enough to be spliced into the SQL."""

    def test_a_frame_is_a_plan(self) -> None:
        from duckdb.expr import PlanBase

        assert isinstance(duckdb.table("t"), PlanBase)

    def test_something_else_with_a_render_method_is_not(self) -> None:
        class Template:
            def render(self) -> str:
                return "DROP TABLE orders"

        with pytest.raises(TypeError):
            col("id").isin(Template())  # type: ignore[arg-type]


class TestTerminalsTakeARelationalConnection:
    """O4. A DB-API connection looks the part and is refused."""

    def test_a_dbapi_connection_is_refused(self) -> None:
        raw = duckdb.dbapi.connect()
        with pytest.raises(TypeError, match=r"duckdb\.Connection, not Connection"):
            duckdb.sql("SELECT 1").fetchall(raw)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=r"duckdb\.Connection"):
            duckdb.sql("SELECT 1").create(raw, "t")  # type: ignore[arg-type]


class TestASubqueryIsComputedOnce:
    """O5. A plan used through a subquery is a step of the plan it lands in."""

    def test_a_scalar_subquery_is_rendered_once(self, con: duckdb.Connection) -> None:
        con.run("CREATE SEQUENCE seq")
        con.run("CREATE TABLE t AS SELECT * FROM (VALUES (1), (2), (3)) v(id)")
        tagged = duckdb.sql("SELECT nextval('seq') AS tag")
        combined = duckdb.table("t").filter(col("id") > tagged.scalar()).cross(tagged)
        sql = combined.render()
        assert sql.count("nextval") == 1
        # One value consumed, so the filter and the column agree.
        rows = combined.fetchall(con)
        assert all(id_ > tag for id_, tag in rows)
        assert {tag for _, tag in rows} == {1}

    def test_a_plan_used_twice_through_isin_is_one_step(self, con: duckdb.Connection) -> None:
        wanted = duckdb.table("orders").filter(col("country") == "nl").select("id")
        plan = duckdb.table("orders").filter(col("id").isin(wanted) | ~col("id").isin(wanted))
        # The step renders once, so its literal appears once in the text and
        # binds once, however many times the expression refers to it.
        assert plan.render().count("'nl'") == 1
        assert plan._sql_and_values()[1] == ["nl"]
        assert plan.count(con) == 5

    def test_a_subquery_keeps_the_stub_free_of_the_catalog(
        self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wanted = duckdb.table("orders").filter(col("country") == "nl").select("id")
        plan = duckdb.table("orders").with_columns(flag=col("id").isin(wanted))
        calls = record_binds(con, monkeypatch)
        assert plan.types(con)[-1] == "BOOLEAN"
        stub = next(call for call in calls if "WHERE FALSE" in call)
        assert '"_use0"' in stub
        assert '"orders"' not in stub


class TestStubsAreOneStepDeep:
    """O6. A step whose input has unknown types completes that input, not the chain."""

    def test_no_bind_ever_covers_more_than_one_step(
        self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = duckdb.table("orders")
        for i in range(5):
            plan = plan.with_columns(**{f"c{i}": col("amount") + i})
        plan = plan.select(sql_expr("c4 * 2"))
        calls = record_binds(con, monkeypatch)
        assert plan.columns(con) == ["(c4 * 2)"]
        # The source, then one stub per step that needed completing.
        assert calls[0] == 'SELECT * FROM "orders"'
        for call in calls[1:]:
            assert call.count("SELECT") == 2, f"more than one step in: {call[:80]}"
        assert len(calls) <= 7


class TestCloseReachesLiveResults:
    """O7. Closing closes what is still being read."""

    def test_a_paused_iterator_stops_at_close(self, tmp_path: Path) -> None:
        path = str(tmp_path / "live.db")
        con = duckdb.connect(path)
        con.run("CREATE TABLE t AS SELECT * FROM range(5000)")
        rows = duckdb.table("t").rows(con)
        assert next(rows) == (0,)
        con.close()

        def keep_reading() -> None:
            # The first batch is already in Python; the next fetch hits the
            # closed result.
            for _ in range(2000):
                next(rows)

        with pytest.raises(exceptions.Error):
            keep_reading()
        # And the file is free while the iterator object is still alive.
        del con
        gc.collect()
        assert duckdb.sql("SELECT count(*) FROM t").fetchall(duckdb.connect(path)) == [(5000,)]

    def test_a_consumed_result_is_not_held(self, con: duckdb.Connection) -> None:
        duckdb.table("orders").fetchall(con)
        gc.collect()
        assert len(con._live) == 0


class TestParametersAreSupplied:
    """O8. `param(name)` takes its value from `parameters=`."""

    def test_a_value_is_supplied_by_name(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").filter(col("country") == param("where")).select("id")
        assert plan.fetchall(con, parameters={"where": "nl"}) == [(1,), (3,), (5,)]
        assert plan.fetchall(con, parameters={"where": "be"}) == [(2,)]

    def test_names_and_literals_share_one_numbering(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").filter((col("country") == "nl") & (col("amount") > param("floor")))
        sql, values = plan._sql_and_values(parameters={"floor": 200})
        assert "$1" in sql
        assert "$2" in sql
        assert values == ["nl", 200]
        assert plan.select("id").fetchall(con, parameters={"floor": 200}) == [(3,)]

    def test_a_missing_value_is_refused(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").filter(col("country") == param("where"))
        with pytest.raises(ValueError, match="no value for parameter 'where'"):
            plan.fetchall(con)

    def test_an_unused_value_is_refused(self, con: duckdb.Connection) -> None:
        plan = duckdb.table("orders").filter(col("country") == param("where"))
        with pytest.raises(ValueError, match="'wher' are not used"):
            plan.fetchall(con, parameters={"where": "nl", "wher": "nl"})

    def test_every_terminal_takes_parameters(self, con: duckdb.Connection, tmp_path: Path) -> None:
        plan = duckdb.table("orders").filter(col("country") == param("where"))
        values = {"where": "nl"}
        assert plan.count(con, parameters=values) == 3
        assert plan.fetchone(con, parameters=values) is not None
        assert len(list(plan.rows(con, parameters=values))) == 3
        assert plan.create(con, "nl", parameters=values) == 3
        assert plan.insert_into(con, "nl", parameters=values) == 3
        assert plan.to_csv(con, str(tmp_path / "nl.csv"), parameters=values) == 3
        assert plan.explain(con, parameters=values)
        assert "nl" in plan.preview(con, parameters=values)


class TestDeclaredRelations:
    """O9. A hole with a heading, closed by `bind`."""

    def heading(self) -> list[tuple[str, str]]:
        return [("id", "INTEGER"), ("country", "VARCHAR"), ("amount", "INTEGER")]

    def test_columns_come_from_the_heading(self) -> None:
        plan = declare("src", self.heading()).filter(col("amount") > 0).select("id", "country")
        assert plan.columns() == ["id", "country"]
        assert plan.types() == ["INTEGER", "VARCHAR"]

    def test_the_hole_renders_as_its_name(self) -> None:
        assert 'FROM "src"' in declare("src", self.heading()).filter(col("amount") > 0).render()

    def test_bind_substitutes_a_plan(self, con: duckdb.Connection) -> None:
        plan = declare("src", self.heading()).filter(col("amount") > 100).select("id")
        assert plan.bind(src=duckdb.table("orders")).fetchall(con) == [(1,), (3,)]
        assert plan.bind(src=duckdb.table("orders").filter(col("country") == "nl")).fetchall(con) == [(1,), (3,)]

    def test_an_open_hole_is_supplied_by_the_catalog(self, con: duckdb.Connection) -> None:
        plan = declare("orders", self.heading()).filter(col("amount") > 100).select("id")
        assert plan.fetchall(con) == [(1,), (3,)]

    def test_bind_checks_names(self) -> None:
        plan = declare("src", self.heading()).select("id")
        with pytest.raises(ValueError, match="declared with columns"):
            plan.bind(src=declare("other", [("a", "INTEGER"), ("b", "INTEGER")]))

    def test_bind_cannot_check_what_only_the_engine_knows(self, con: duckdb.Connection) -> None:
        # A raw SQL source has no derivable columns, so the check waits for the
        # engine, which refuses the mismatch when the plan runs.
        plan = declare("src", self.heading()).select("id")
        bound = plan.bind(src=duckdb.sql("SELECT 1 AS a"))
        with pytest.raises(exceptions.Error):
            bound.fetchall(con)

    def test_bind_checks_types_where_known(self) -> None:
        plan = declare("src", [("id", "INTEGER")]).select("id")
        with pytest.raises(ValueError, match="declares id as INTEGER"):
            plan.bind(src=declare("other", [("id", "VARCHAR")]))

    def test_bind_refuses_an_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="no declared relation named 'nope'"):
            declare("src", self.heading()).bind(nope=duckdb.table("orders"))

    def test_a_partial_bind_leaves_a_hole(self, con: duckdb.Connection) -> None:
        left, right = declare("l", [("id", "INTEGER")]), declare("countries", [("code", "VARCHAR")])
        plan = left.join(right, on=col("l.id") == col("r.code"), how="cross")
        half = plan.bind(l=duckdb.table("orders").select("id"))
        assert 'FROM "countries"' in half.render()
        assert half.count(con) == 10

    def test_bind_reaches_a_subquery_over_the_relation(self, con: duckdb.Connection) -> None:
        # The subquery holds the plan it saw before the bind; the rebuilt step
        # answers for it, so the bound plan is still one step, computed once.
        src = declare("src", self.heading())
        plan = src.filter(col("amount") > src.aggregate(col("amount").mean().alias("m")).scalar()).select("id")
        bound = plan.bind(src=duckdb.table("orders"))
        assert bound.render().count('"orders"') == 1
        assert bound.fetchall(con) == [(3,)]

    def test_a_heading_may_not_repeat_a_name(self) -> None:
        with pytest.raises(ValueError, match="declare would produce 'id' more than once"):
            declare("src", [("id", "INTEGER"), ("id", "INTEGER")])

    def test_table_no_longer_asserts_a_schema(self) -> None:
        with pytest.raises(TypeError):
            duckdb.table("orders", schema=[("id", "INTEGER")])  # type: ignore[call-arg]


class TestAggregatesAreRealMethods:
    """O10. Written out, so they type, complete and can be checked."""

    def test_methods_are_functions_with_docstrings(self) -> None:
        assert callable(duckdb.Expr.sum)
        assert "sum" in (duckdb.Expr.sum.__doc__ or "")
        assert "quantile_cont" in (duckdb.Expr.quantile.__doc__ or "")

    def test_an_unknown_name_is_an_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            col("v").no_such_aggregate()  # type: ignore[attr-defined]

    def test_count_all_counts_rows(self, con: duckdb.Connection) -> None:
        from duckdb import count_all

        plan = duckdb.table("orders").aggregate(count_all().alias("rows"), col("amount").count().alias("values"))
        assert plan.fetchall(con) == [(5, 4)]

    def test_the_generated_module_is_current(self) -> None:
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
        from gen_aggregates import render

        committed = (pathlib.Path(__file__).parent.parent / "src" / "duckdb" / "_aggregates.py").read_text()
        assert committed == render(), "run scripts/gen_aggregates.py"


class TestWindowFrames:
    """O11. Rows and range bounds, and IGNORE NULLS."""

    def test_a_rows_frame(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT unnest([1, 2, 3, 4]) AS v").select(
            col("v").sum().over(order_by=col("v"), rows=(-1, 0)).alias("running")
        )
        assert [r[0] for r in plan.fetchall(con)] == [1, 3, 5, 7]

    def test_a_range_frame(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT unnest([1, 2, 3]) AS v").select(
            col("v").sum().over(order_by=col("v"), range=(None, 0)).alias("cumulative")
        )
        assert [r[0] for r in plan.fetchall(con)] == [1, 3, 6]

    def test_both_bounds_are_refused(self) -> None:
        with pytest.raises(TypeError, match="not both"):
            col("v").sum().over(rows=(-1, 0), range=(-1, 0))

    def test_ignore_nulls_goes_inside_the_call(self, con: duckdb.Connection) -> None:
        from duckdb import last_value

        expression = last_value(col("v")).ignore_nulls().over(order_by=col("i"))
        assert 'last_value("v" IGNORE NULLS) OVER' in expression.fragment()
        plan = duckdb.sql("SELECT unnest([1, 2, 3]) AS i, unnest([1, NULL, 3]) AS v").select(expression.alias("lv"))
        assert [r[0] for r in plan.fetchall(con)] == [1, 1, 3]

    def test_ignore_nulls_needs_a_function_call(self) -> None:
        with pytest.raises(TypeError, match="function call"):
            col("v").ignore_nulls()


class TestExpressionNamespaces:
    """O12. `.str`, `.dt` and `.list`, generated from the engine's catalog."""

    def test_string_methods(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT 'Hello World' AS s").select(
            col("s").str.upper().alias("u"), col("s").str.contains("World").alias("c"), col("s").str.length().alias("n")
        )
        assert plan.fetchall(con) == [("HELLO WORLD", True, 11)]

    def test_date_methods(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT DATE '2026-03-15' AS d").select(
            col("d").dt.year().alias("y"), col("d").dt.trunc("month").alias("m"), col("d").dt.dayname().alias("n")
        )
        assert plan.fetchall(con) == [(2026, datetime.datetime(2026, 3, 1), "Sunday")]

    def test_list_methods(self, con: duckdb.Connection) -> None:
        plan = duckdb.sql("SELECT [3, 1, 2] AS l").select(
            col("l").list.sort().alias("s"), col("l").list.contains(2).alias("c"), col("l").list.unique().alias("u")
        )
        assert plan.fetchall(con) == [([1, 2, 3], True, 3)]

    def test_arguments_are_bound(self) -> None:
        with ParamSink() as sink:
            sql = col("s").str.contains("x'; DROP TABLE t; --").fragment()
        assert "DROP" not in sql
        assert sink.entries[0][1] == "x'; DROP TABLE t; --"

    def test_every_generated_function_exists_in_this_engine(self, con: duckdb.Connection) -> None:
        from duckdb import _namespaces

        known = {row[0] for row in duckdb.sql("SELECT DISTINCT function_name FROM duckdb_functions()").fetchall(con)}
        missing = [
            f"{cls.__name__}.{method} -> {function}"
            for cls in (_namespaces.StringMethods, _namespaces.DateMethods, _namespaces.ListMethods)
            for method, (function, _, _) in cls.SPEC.items()
            if function not in known
        ]
        assert not missing, missing

    @pytest.mark.parametrize("namespace", ["str", "dt", "list"])
    def test_every_generated_method_calls_its_function_the_right_way_round(
        self, con: duckdb.Connection, namespace: str
    ) -> None:
        """Call every method with arguments made from its parameter types.

        A wrong subject position or arity is a binder error saying no function
        matches. Any other complaint is the engine judging the values, which is
        not what this checks, so only that error class fails the test.
        """
        from duckdb import _namespaces

        cls = {"str": _namespaces.StringMethods, "dt": _namespaces.DateMethods, "list": _namespaces.ListMethods}[
            namespace
        ]
        subject = {"str": "'abc'", "dt": "TIMESTAMP '2026-03-15 10:00:00'", "list": "[1, 2, 3]"}[namespace]
        source = duckdb.sql(f"SELECT {subject} AS x")
        wrong = []
        for method, (_function, position, types) in cls.SPEC.items():
            others = [t for i, t in enumerate(types) if i != position]
            # A macro carries no types; give it one value per leading slot.
            arguments = [_argument_for(t) for t in others] if types else [1] * position
            call = getattr(col("x").__getattribute__(namespace), method)
            try:
                source.select(call(*arguments).alias("v")).columns(con)
            except exceptions.Error as error:
                if "No function matches" in str(error) or "does not exist" in str(error):
                    wrong.append(f"{namespace}.{method}: {str(error).splitlines()[0][:90]}")
        assert not wrong, wrong

    def test_the_generated_module_is_current(self, con: duckdb.Connection) -> None:
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
        from gen_namespaces import catalog, classify, render

        committed = (pathlib.Path(__file__).parent.parent / "src" / "duckdb" / "_namespaces.py").read_text()
        assert committed == render(classify(catalog(con))), "run scripts/gen_namespaces.py"


class TestReviewRoundThree:
    """Regressions for the third review, of the decoupling and the generated surface."""

    def test_a_using_join_with_a_suffix_folds_the_key(self, con: duckdb.Connection) -> None:
        # Finding 1. `keys` was only filled for a list of names, so a single
        # name left the right side's copy of the key in the rows: three names
        # over four values, and preview() failed zipping them.
        con.run("CREATE TABLE a AS SELECT 1 AS id, 10 AS amount")
        con.run("CREATE TABLE b AS SELECT 1 AS id, 20 AS amount")
        joined = duckdb.table("a").join(duckdb.table("b"), on="id", suffix="_r")
        assert joined.columns(con) == ["id", "amount", "amount_r"]
        assert joined.fetchall(con) == [(1, 10, 20)]
        assert "amount_r" in joined.preview(con)

    def test_bind_reaches_a_hole_used_directly_as_a_subquery(self, con: duckdb.Connection) -> None:
        # Finding 2. The replacement entry skipped the alias bookkeeping, so a
        # subquery holding the hole itself kept rendering the hole's name and
        # ran against whatever table happened to have it.
        con.run("CREATE TABLE a AS SELECT 1 AS id, 10 AS amount")
        con.run("CREATE TABLE src AS SELECT 999 AS amount")
        hole = declare("src", [("id", "INTEGER"), ("amount", "INTEGER")])
        plan = hole.filter(col("amount") < hole.aggregate(col("amount").max().alias("m")).scalar() + 1)
        bound = plan.bind(src=duckdb.table("a"))
        assert '"src"' not in bound.render()
        assert bound.fetchall(con) == [(1, 10)]

    def test_a_setting_change_forgets_stub_answers(self, con: duckdb.Connection) -> None:
        # Finding 3. A stub answer depends on settings, and settings change
        # through run(); the answer was kept across it.
        con.run("CREATE TABLE t3 AS SELECT 7 AS x, 2 AS y")
        plan = duckdb.table("t3").with_columns(ratio=col("x") / col("y"))
        assert plan.types(con)[-1] == "DOUBLE"
        con.run("SET integer_division = true")
        try:
            assert plan.types(con)[-1] == "INTEGER"
        finally:
            con.run("SET integer_division = false")

    def test_bind_checks_a_plan_that_cannot_resolve_for_its_own_reasons(self) -> None:
        # Finding 5. Every ValueError from resolve() was taken to mean "needs a
        # connection"; a bound plan's own defect was swallowed with it.
        bad = declare("t", [("a", "INTEGER"), ("b", "INTEGER")]).rename(a="x", b="x")
        outer = declare("src", [("x", "INTEGER")]).select("x")
        with pytest.raises(ValueError, match="rename would produce 'x' more than once"):
            outer.bind(src=bad)

    def test_two_holes_of_one_name_must_agree(self, con: duckdb.Connection) -> None:
        # Finding 7. Holes were keyed by name, so two different headings under
        # one name were both replaced and checked against only the last.
        left = declare("dup", [("x", "INTEGER")]).select("x")
        right = declare("dup", [("y", "INTEGER")]).select("y")
        with pytest.raises(ValueError, match="declared twice with different headings"):
            left.union(right).bind(dup=duckdb.table("orders"))

    def test_two_holes_of_one_name_and_heading_are_one_relation(self, con: duckdb.Connection) -> None:
        heading = [("id", "INTEGER"), ("country", "VARCHAR"), ("amount", "INTEGER")]
        left, right = declare("o", heading).select("id"), declare("o", heading).select("id")
        assert left.union(right).bind(o=duckdb.table("orders")).count(con) == 10

    def test_close_closes_every_live_result_even_if_one_refuses(self, con: duckdb.Connection) -> None:
        # Finding 9. One result whose close raised left the rest open and the
        # connection looking open.
        from duckdb.connection import LiveResult

        class Stubborn(LiveResult):
            def close(self) -> None:
                message = "will not close"
                raise RuntimeError(message)

        # One live result per engine connection, so the results come from
        # siblings; what is under test is this connection's bookkeeping.
        first = con._track(con.duplicate()._engine().execute("SELECT 1"))
        stubborn = Stubborn(con.duplicate()._engine().execute("SELECT 2"))
        con._live.add(stubborn)
        second = con._track(con.duplicate()._engine().execute("SELECT 3"))
        with pytest.raises(RuntimeError, match="will not close"):
            con.close()
        assert con._raw is None
        assert con._database is None
        for live in (first, second):
            with pytest.raises(exceptions.Error):
                live.fetch_all()

    def test_tracking_and_closing_from_two_threads(self) -> None:
        # Finding 10. `_live` was a bare WeakSet touched from any thread.
        import threading

        con = duckdb.connect()
        con.run("CREATE TABLE t AS SELECT * FROM range(100000)")
        stop = threading.Event()
        errors: list[BaseException] = []

        def churn() -> None:
            try:
                while not stop.is_set():
                    con._track(con._engine().execute("SELECT 1"))
            except exceptions.Error:
                pass  # the connection closed under us, which is the point
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=churn)
        worker.start()
        con.close()
        stop.set()
        worker.join()
        assert not errors

    def test_dbapi_survives_a_statement_that_fails_while_running(self) -> None:
        # Finding 11. The cursor's execute and the connection's transaction
        # statements drained without try/finally. A constraint violation fails
        # at execute rather than drain, so this checks the cursor carries on
        # after a failure; the try/finally itself is confirmed by reading.
        raw = duckdb.dbapi.connect()
        cursor = raw.cursor()
        cursor.execute("CREATE TABLE u (v INTEGER PRIMARY KEY)")
        cursor.execute("INSERT INTO u VALUES (1)")
        raw.commit()
        with pytest.raises(exceptions.Error):
            cursor.execute("INSERT INTO u VALUES (1)")
        raw.rollback()
        cursor.execute("INSERT INTO u VALUES (2)")
        assert cursor.execute("SELECT count(*) FROM u").fetchone() == (2,)

    @pytest.mark.parametrize("make", [lambda names: (n for n in names), set, frozenset, lambda names: map(str, names)])
    def test_any_iterable_of_names_is_a_column_list(
        self, con: duckdb.Connection, make: Callable[[list[str]], object]
    ) -> None:
        # Finding 12. Only list and tuple were sequences; a generator or a set
        # was wrapped whole and refused.
        keys = make(["country"])
        grouped = duckdb.table("orders").aggregate(col("amount").sum().alias("total"), group_by=keys)
        assert grouped.columns(con) == ["country", "total"]
        assert duckdb.table("orders").distinct(on=make(["country"])).count(con) == 3


class TestReviewRoundFour:
    """Regressions for the fourth review: bind aliasing, the caches, the generator."""

    def test_bound_and_unbound_descendants_of_one_hole_coexist(self, con: duckdb.Connection) -> None:
        # Finding 1. The alias backfill was global, so the bound copy's name
        # overwrote the hole's own, and the unbound branch named a CTE twice.
        con.run("CREATE TABLE catalog_x AS SELECT 1 AS v")
        con.run("CREATE TABLE x AS SELECT 2 AS v")
        hole = declare("x", [("v", "INTEGER")])
        step = hole.select(col("v"))
        both = step.bind(x=duckdb.table("catalog_x")).union(step)
        assert sorted(both.fetchall(con)) == [(1,), (2,)]

    def test_a_setting_changed_through_a_plan_forgets_stub_answers(self, con: duckdb.Connection) -> None:
        # Finding 2. Only run() forgot; a SET executed as a plan did not.
        con.run("CREATE TABLE t2 AS SELECT 7 AS x, 2 AS y")
        plan = duckdb.table("t2").with_columns(ratio=col("x") / col("y"))
        assert plan.types(con)[-1] == "DOUBLE"
        duckdb.sql("SET integer_division = true").fetchall(con)
        try:
            assert plan.types(con)[-1] == "INTEGER"
        finally:
            con.run("SET integer_division = false")

    def test_a_query_does_not_forget_stub_answers(
        self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other half of finding 2: a plain SELECT must leave them alone, or
        # the cache would empty on every fetch.
        plan = duckdb.table("orders").select(sql_expr("amount * 2"))
        assert plan.columns(con) == ["(amount * 2)"]
        duckdb.table("orders").fetchall(con)
        calls = record_binds(con, monkeypatch)
        assert plan.columns(con) == ["(amount * 2)"]
        assert all("WHERE FALSE" not in call for call in calls), "the stub was asked again"

    def test_list_prepend_puts_the_list_second(self, con: duckdb.Connection) -> None:
        # Finding 3. Every list function had its subject hardwired first;
        # `list_prepend(element, list)` does not.
        assert duckdb.sql("SELECT [1, 2] AS l").select(col("l").list.prepend(0).alias("p")).fetchall(con) == [
            ([0, 1, 2],)
        ]

    def test_timezone_binds_the_family_its_docstring_describes(self, con: duckdb.Connection) -> None:
        # Finding 4. The two-argument conversion family had won, under the
        # one-argument extraction's description.
        from duckdb._namespaces import DateMethods

        assert DateMethods.SPEC["timezone"][1] == 0
        assert "offset" in (DateMethods.timezone.__doc__ or "")
        plan = duckdb.sql("SELECT TIMESTAMPTZ '2026-03-15 10:00:00+00' AS ts").select(
            col("ts").dt.timezone().alias("z")
        )
        assert plan.fetchall(con)[0][0] is not None

    def test_nothing_numeric_is_filed_under_dt(self) -> None:
        # Finding 6. Any overload with a temporal first parameter admitted a
        # function; `isfinite(DATE)` brought the whole of isfinite along.
        from duckdb._namespaces import DateMethods

        assert not {"isfinite", "isinf", "generate_series", "range"} & set(DateMethods.SPEC)

    def test_a_result_tracked_after_close_is_refused_and_closed(self) -> None:
        # Finding 7. A result that raced close() was tracked into an empty set
        # and held the engine past a close that had returned.
        con = duckdb.connect()
        raw = con._engine().execute("SELECT 1")
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            con._track(raw)
        assert len(con._live) == 0
        with pytest.raises(exceptions.Error):
            raw.fetch_all()

    def test_closing_under_load_leaves_nothing_live(self, tmp_path: Path) -> None:
        # The round-3 thread test only checked for exceptions. This checks the
        # outcome: nothing live, and the file released.
        import threading

        path = str(tmp_path / "load.db")
        con = duckdb.connect(path)
        con.run("CREATE TABLE t AS SELECT 1 AS v")
        stop = threading.Event()

        def churn() -> None:
            while not stop.is_set():
                try:
                    con._execute("SELECT 1")
                except exceptions.Error:
                    return

        workers = [threading.Thread(target=churn) for _ in range(4)]
        for worker in workers:
            worker.start()
        con.close()
        stop.set()
        for worker in workers:
            worker.join()
        assert len(con._live) == 0
        # close() dropped the handle and the database itself, so the file is
        # free without waiting for this object to be collected.
        assert duckdb.sql("SELECT count(*) FROM t").fetchall(duckdb.connect(path)) == [(1,)]

    def test_the_stub_cache_evicts_one_at_a_time(self, con: duckdb.Connection) -> None:
        # Finding 13. At the limit the whole cache was cleared.
        from duckdb import frame

        con._stub_answers.clear()
        for i in range(frame._STUB_LIMIT + 3):
            con._stub_answers[f"q{i}"] = ()
            while len(con._stub_answers) > frame._STUB_LIMIT:
                con._stub_answers.pop(next(iter(con._stub_answers)))
        assert len(con._stub_answers) == frame._STUB_LIMIT
        assert "q0" not in con._stub_answers
        assert f"q{frame._STUB_LIMIT + 2}" in con._stub_answers

    def test_no_generated_docstring_has_a_double_period(self) -> None:
        # Finding 10.
        import inspect

        from duckdb import _namespaces

        assert ".." not in inspect.getsource(_namespaces)

    @pytest.mark.parametrize("how", sorted(["semi", "anti"]))
    def test_a_join_kind_row_decides_what_it_keeps(self, con: duckdb.Connection, how: str) -> None:
        # Finding 14. One descriptor per kind; the keeps-right rule read from it.
        joined = duckdb.table("orders").join(duckdb.table("orders"), on="id", how=how)
        assert joined.columns(con) == ["id", "country", "amount"]

    def test_a_join_condition_with_a_subquery_registers_the_plan(self, con: duckdb.Connection) -> None:
        # Finding 15. join() derived its uses by hand; it now goes through the
        # one helper every verb uses, so the subquery's plan is one step.
        nl = duckdb.table("orders").filter(col("country") == "nl").select("id")
        joined = duckdb.table("orders").join(
            duckdb.table("countries"), on=(col("l.country") == col("r.code")) & col("l.id").isin(nl)
        )
        assert len(joined._uses) == 1
        assert joined.render().count("'nl'") == 1
        assert joined.count(con) == 3

    def test_render_with_a_connection_is_the_executed_text(self, con: duckdb.Connection) -> None:
        # Finding 5. The blind form and the executed form can order a
        # replaced column differently; render(con) gives the executed one.
        plan = duckdb.table("orders").with_columns(country=col("country").str.upper())
        assert "COLUMNS(lambda" in plan.render()
        assert "REPLACE" in plan.render(con)
        assert duckdb.sql(plan.render(con)).columns(con) == plan.columns(con)
