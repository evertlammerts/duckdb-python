"""The frame layer: building a query a step at a time.

Grouped by what can break rather than by method, because the interesting
failures are shared: a literal reaching the SQL text, a step that cannot follow
a WITH, a name that turns into syntax.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb import _duckdb, col, exceptions, sql_expr, star

if TYPE_CHECKING:
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
    return con.table("orders")


class TestGraph:
    """Steps become CTEs, and a step used twice is still computed once."""

    def test_a_single_step_needs_no_cte(self, orders: duckdb.Frame) -> None:
        assert orders.render() == 'SELECT * FROM "orders"'

    def test_each_step_becomes_a_cte(self, orders: duckdb.Frame) -> None:
        sql = orders.filter(col("amount") > 100).select(col("id")).render()
        assert sql.count(" AS (") == 2
        assert sql.startswith("WITH ")

    def test_a_reused_step_is_rendered_once(self, orders: duckdb.Frame) -> None:
        # The whole point of walking by identity: the engine sees one scan of
        # the shared step, not two copies of its subtree.
        shared = orders.filter(col("country") == "nl")
        joined = shared.join(shared, on=col("l.id") == col("r.id"), suffix="_r")
        sql = joined.render()
        assert sql.count("WHERE") == 1
        assert len(joined.fetchall()) == 3

    def test_a_self_join_is_unambiguous(self, orders: duckdb.Frame) -> None:
        # Both sides name the same CTE, so without the l/r aliases the FROM
        # clause would not parse.
        pairs = orders.join(orders, on=col("l.id") == (col("r.id") - 1), suffix="_r").fetchall()
        assert len(pairs) == 4

    def test_a_frame_can_be_extended_twice_independently(self, orders: duckdb.Frame) -> None:
        base = orders.filter(col("amount").is_not_null())
        assert len(base.filter(col("country") == "nl").fetchall()) == 2
        assert len(base.filter(col("country") == "be").fetchall()) == 1


class TestLiteralsAreBound:
    """No value a caller supplied is ever written into the SQL text."""

    def test_a_string_filter_binds_rather_than_inlines(self, orders: duckdb.Frame) -> None:
        sql, values = orders.filter(col("country") == "nl")._bind()
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_a_quote_in_a_value_cannot_escape(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        hostile = "nl'; DROP TABLE orders; --"
        assert orders.filter(col("country") == hostile).fetchall() == []
        assert len(con.table("orders")) == 5

    def test_literals_inside_a_subquery_are_bound_too(self, orders: duckdb.Frame) -> None:
        inner = orders.filter(col("country") == "nl").select(col("id"))
        sql, values = orders.filter(col("id").isin(inner))._bind()
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_numbers_stay_in_the_text(self, orders: duckdb.Frame) -> None:
        # Nothing to escape, and inlining lets DuckDB type the literal itself.
        sql, values = orders.filter(col("amount") > 100)._bind()
        assert "100" in sql
        assert values is None


class TestSchema:
    """The binder answers the shape, without running anything."""

    def test_columns_and_types_come_from_the_binder(self, orders: duckdb.Frame) -> None:
        assert orders.columns == ["id", "country", "amount"]
        assert orders.types == ["INTEGER", "VARCHAR", "INTEGER"]

    def test_a_derived_column_is_typed_by_the_binder(self, orders: duckdb.Frame) -> None:
        # INTEGER times a decimal literal is DECIMAL, not DOUBLE. The client
        # does not work that out; it asks, which is the whole reason `.schema`
        # goes to the binder.
        widened = orders.with_columns(doubled=col("amount") * 2.5)
        assert widened.types[-1] == "DECIMAL(12,1)"

    def test_the_schema_is_asked_for_once(self, orders: duckdb.Frame) -> None:
        # Identity would not show this any more: the schema is built from a
        # cached shape, so what matters is that the engine is asked once.
        assert orders.schema == orders.schema
        assert orders._cached_shape is not None

    def test_a_bad_query_reports_the_engine_error(self, con: duckdb.Connection) -> None:
        with pytest.raises(exceptions.CatalogError):
            _ = con.sql("SELECT * FROM missing").columns


class TestProjection:
    def test_select_keeps_only_what_is_named(self, orders: duckdb.Frame) -> None:
        assert orders.select(col("id"), col("country")).columns == ["id", "country"]

    def test_a_bare_string_selects_a_column(self, orders: duckdb.Frame) -> None:
        # The opposite of the rule inside an expression, where a string is a
        # value. Position decides, and nothing else a string could mean here.
        assert orders.select("id").columns == ["id"]

    def test_select_refuses_anything_else(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="column name or expression"):
            orders.select(3.5)

    def test_star_can_exclude(self, orders: duckdb.Frame) -> None:
        assert orders.select(star(exclude=["amount"])).columns == ["id", "country"]

    def test_with_columns_appends_a_new_name(self, orders: duckdb.Frame) -> None:
        added = orders.with_columns(big=col("amount") > 100)
        assert added.columns == ["id", "country", "amount", "big"]

    def test_with_columns_replaces_in_place(self, orders: duckdb.Frame) -> None:
        # Replacing must keep the column where it was, so downstream positional
        # code does not silently shift.
        replaced = orders.with_columns(amount=col("amount") * 2)
        assert replaced.columns == ["id", "country", "amount"]
        assert replaced.filter(col("id") == 1).fetchall() == [(1, "nl", 240)]

    def test_with_columns_can_add_and_replace_together(self, orders: duckdb.Frame) -> None:
        both = orders.with_columns(amount=col("amount") + 1, note=sql_expr("'x'"))
        assert both.columns == ["id", "country", "amount", "note"]

    def test_drop_and_rename(self, orders: duckdb.Frame) -> None:
        assert orders.drop("amount").columns == ["id", "country"]
        assert orders.rename(country="iso").columns == ["id", "iso", "amount"]

    def test_getitem_gives_one_column(self, orders: duckdb.Frame) -> None:
        assert orders["country"].columns == ["country"]


class TestRows:
    def test_filter_accepts_an_expression_or_sql(self, orders: duckdb.Frame) -> None:
        assert len(orders.filter(col("country") == "nl").fetchall()) == 3
        assert len(orders.filter("country = 'nl'").fetchall()) == 3

    def test_sort_takes_direction_and_nulls(self, orders: duckdb.Frame) -> None:
        ordered = orders.sort(col("amount").desc().nulls_last()).fetchall()
        assert [row[0] for row in ordered] == [3, 1, 2, 4, 5]

    def test_limit_head_and_offset(self, orders: duckdb.Frame) -> None:
        ordered = orders.sort(col("id"))
        assert [r[0] for r in ordered.limit(2).fetchall()] == [1, 2]
        assert [r[0] for r in ordered.head(3).fetchall()] == [1, 2, 3]
        assert [r[0] for r in ordered.offset(3).fetchall()] == [4, 5]
        assert [r[0] for r in ordered.limit(2, offset=1).fetchall()] == [2, 3]

    def test_distinct_over_all_columns_and_over_keys(self, orders: duckdb.Frame) -> None:
        assert len(orders.select(col("country")).distinct().fetchall()) == 3
        assert len(orders.distinct(on="country").fetchall()) == 3

    def test_len_counts_without_fetching(self, orders: duckdb.Frame) -> None:
        assert len(orders) == 5

    def test_iteration_crosses_batch_boundaries(self, con: duckdb.Connection) -> None:
        # __iter__ pulls 1024 rows at a time, so a shorter run would never test
        # the loop that refills.
        many = con.sql("SELECT * FROM range(3000)")
        assert sum(1 for _ in many) == 3000

    def test_fetchone_on_an_empty_result(self, orders: duckdb.Frame) -> None:
        assert orders.filter(col("id") < 0).fetchone() is None


class TestAggregation:
    def test_aggregate_without_keys(self, orders: duckdb.Frame) -> None:
        assert orders.aggregate(col("amount").sum().alias("total")).fetchall() == [(550,)]

    def test_group_keys_come_first(self, orders: duckdb.Frame) -> None:
        grouped = orders.group_by(col("country")).agg(col("amount").sum().alias("total"))
        assert grouped.columns == ["country", "total"]
        assert sorted(grouped.fetchall()) == [("be", 80), ("de", 50), ("nl", 420)]

    def test_aggregate_takes_keys_directly(self, orders: duckdb.Frame) -> None:
        by_keyword = orders.aggregate(col("amount").sum().alias("total"), group_by="country")
        assert sorted(by_keyword.fetchall()) == [("be", 80), ("de", 50), ("nl", 420)]

    def test_filtering_after_agg_is_having(self, orders: duckdb.Frame) -> None:
        # There is no `having` verb because there is no need for one: every
        # step is its own CTE, so a filter after an aggregate already runs
        # against the grouped rows.
        grouped = orders.group_by(col("country")).agg(col("amount").sum().alias("total"))
        assert grouped.filter(col("total") > 100).fetchall() == [("nl", 420)]

    def test_a_window_does_not_group(self, orders: duckdb.Frame) -> None:
        ranked = orders.with_columns(rank=duckdb.row_number().over(order_by=col("id")))
        assert len(ranked.fetchall()) == 5


class TestJoins:
    def test_join_on_a_shared_name(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        renamed = con.table("countries").rename(code="country")
        assert len(orders.join(renamed, on="country").fetchall()) == 4

    def test_join_on_an_expression_names_the_sides(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        joined = orders.join(con.table("countries"), on=col("l.country") == col("r.code"))
        assert len(joined.fetchall()) == 4

    def test_join_on_several_names(self, con: duckdb.Connection) -> None:
        left = con.sql("SELECT 1 AS a, 2 AS b, 'x' AS v")
        right = con.sql("SELECT 1 AS a, 2 AS b, 'y' AS w")
        assert left.join(right, on=["a", "b"]).fetchall() == [(1, 2, "x", "y")]

    @pytest.mark.parametrize(
        ("how", "expected"),
        [("inner", 4), ("left", 5), ("semi", 4), ("anti", 1), ("outer", 5)],
    )
    def test_join_kinds(self, orders: duckdb.Frame, con: duckdb.Connection, how: str, expected: int) -> None:
        joined = orders.join(con.table("countries"), on=col("l.country") == col("r.code"), how=how)
        assert len(joined.fetchall()) == expected

    def test_cross_join_needs_no_keys(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        assert len(orders.cross(con.table("countries")).fetchall()) == 10

    def test_a_keyed_join_without_keys_is_refused(self, orders: duckdb.Frame, con: duckdb.Connection) -> None:
        with pytest.raises(TypeError, match="needs `on`"):
            orders.join(con.table("countries"))


class TestSetOperations:
    def test_union_keeps_duplicates_unless_asked(self, orders: duckdb.Frame) -> None:
        nl = orders.filter(col("country") == "nl")
        assert len(nl.union(nl).fetchall()) == 6
        assert len(nl.union(nl, all=False).fetchall()) == 3

    def test_union_by_name_ignores_position(self, con: duckdb.Connection) -> None:
        left = con.sql("SELECT 1 AS a, 2 AS b")
        right = con.sql("SELECT 3 AS b, 4 AS a")
        assert sorted(left.union_by_name(right).fetchall()) == [(1, 2), (4, 3)]

    def test_intersect_and_except(self, orders: duckdb.Frame) -> None:
        nl = orders.filter(col("country") == "nl")
        low = orders.filter(col("amount") < 200)
        assert [r[0] for r in nl.intersect(low).fetchall()] == [1]
        assert sorted(r[0] for r in nl.except_(low).fetchall()) == [3, 5]


class TestSample:
    def test_a_row_count_is_exact(self, orders: duckdb.Frame) -> None:
        assert len(orders.sample(2, seed=7).fetchall()) == 2

    def test_a_seed_repeats(self, orders: duckdb.Frame) -> None:
        assert orders.sample(3, seed=11).fetchall() == orders.sample(3, seed=11).fetchall()

    def test_a_percentage_stays_within_the_table(self, orders: duckdb.Frame) -> None:
        assert len(orders.sample(percent=50, seed=3).fetchall()) <= 5

    def test_a_method_can_be_chosen(self, orders: duckdb.Frame) -> None:
        assert len(orders.sample(percent=100, method="bernoulli", seed=1).fetchall()) == 5

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
        rows = con.sql("SELECT 1 AS k, [10, 20] AS xs").unnest("xs")
        assert rows.fetchall() == [(1, 10), (2 - 1, 20)]

    def test_unnest_keeps_the_column_where_it_was(self, con: duckdb.Connection) -> None:
        rows = con.sql("SELECT [1, 2] AS xs, 'tail' AS t").unnest("xs")
        assert rows.columns == ["xs", "t"]

    def test_unnesting_two_columns_walks_them_in_step(self, con: duckdb.Connection) -> None:
        rows = con.sql("SELECT [1, 2] AS xs, ['a', 'b'] AS ys").unnest("xs", "ys")
        assert rows.fetchall() == [(1, "a"), (2, "b")]

    def test_unnest_needs_a_column(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="at least one column"):
            orders.unnest()

    def test_unpivot_folds_columns_into_rows(self, con: duckdb.Connection) -> None:
        wide = con.sql("SELECT 'nl' AS country, 1 AS q1, 2 AS q2")
        long = wide.unpivot("q1", "q2", name="quarter", value="sales")
        assert long.fetchall() == [("nl", "q1", 1), ("nl", "q2", 2)]

    def test_unpivot_names_are_quoted(self, con: duckdb.Connection) -> None:
        wide = con.sql("SELECT 1 AS q1")
        long = wide.unpivot("q1", name="the name", value="the value")
        assert long.columns == ["the name", "the value"]

    def test_unpivot_needs_a_column(self, orders: duckdb.Frame) -> None:
        with pytest.raises(TypeError, match="at least one column"):
            orders.unpivot()

    def test_reshaping_works_mid_chain(self, con: duckdb.Connection) -> None:
        # SUMMARIZE and UNPIVOT cannot follow a WITH, so both are wrapped in a
        # SELECT. Every step but the first is reached through a WITH, which is
        # what this checks.
        wide = con.sql("SELECT 'nl' AS country, 1 AS q1, 2 AS q2").filter(col("country") == "nl")
        assert len(wide.unpivot("q1", "q2").filter(col("value") > 1).fetchall()) == 1


class TestInspection:
    def test_describe_reports_statistics(self, orders: duckdb.Frame) -> None:
        stats = orders.describe()
        assert stats.columns[:2] == ["column_name", "column_type"]
        assert len(stats.fetchall()) == 3

    def test_describe_works_mid_chain(self, orders: duckdb.Frame) -> None:
        assert len(orders.filter(col("country") == "nl").describe().fetchall()) == 3

    def test_explain_returns_a_plan(self, orders: duckdb.Frame) -> None:
        assert "Seq Scan" in orders.explain() or "SEQ_SCAN" in orders.explain()

    def test_explain_analyze_runs_the_query(self, orders: duckdb.Frame) -> None:
        assert "Total Time" in orders.explain(analyze=True)

    def test_preview_draws_a_table(self, orders: duckdb.Frame) -> None:
        drawn = orders.sort(col("id")).preview()
        assert "id" in drawn
        assert "INTEGER" in drawn
        assert drawn.startswith("┌")
        assert "NULL" in drawn  # the missing amount, not an empty cell

    def test_preview_says_when_there_are_more_rows(self, con: duckdb.Connection) -> None:
        drawn = con.sql("SELECT * FROM range(50)").preview(3)
        assert "there are more" in drawn
        assert drawn.count("\n│") == 5  # heading, types, and three rows

    def test_preview_of_an_empty_frame_still_draws(self, orders: duckdb.Frame) -> None:
        drawn = orders.filter(col("id") < 0).preview()
        assert "id" in drawn
        assert "there are more" not in drawn

    def test_long_values_are_shortened(self, con: duckdb.Connection) -> None:
        drawn = con.sql("SELECT repeat('x', 200) AS wide").preview()
        assert "…" in drawn
        assert max(len(line) for line in drawn.splitlines()) < 60

    def test_show_prints_the_preview(self, orders: duckdb.Frame, capsys: pytest.CaptureFixture[str]) -> None:
        orders.show(2)
        assert capsys.readouterr().out.strip() == orders.preview(2)

    def test_repr_names_the_error_rather_than_raising(self, con: duckdb.Connection) -> None:
        # A repr that raises makes a debugger unusable; one that hides the
        # reason is worse.
        text = repr(con.sql("SELECT * FROM missing"))
        assert "CatalogError" in text
        assert "missing" in text


class TestSubqueries:
    def test_scalar_supplies_a_single_value(self, orders: duckdb.Frame) -> None:
        average = orders.aggregate(col("amount").mean().alias("m"))
        above = orders.filter(col("amount") > average.scalar())
        assert [row[0] for row in above.fetchall()] == [3]

    def test_isin_accepts_a_query(self, orders: duckdb.Frame) -> None:
        nl_ids = orders.filter(col("country") == "nl").select(col("id"))
        assert [row[0] for row in orders.filter(col("id").isin(nl_ids)).fetchall()] == [1, 3, 5]

    def test_isin_still_accepts_values(self, orders: duckdb.Frame) -> None:
        assert [row[0] for row in orders.filter(col("id").isin([1, 2])).fetchall()] == [1, 2]

    def test_a_subquery_carries_its_own_steps(self, orders: duckdb.Frame) -> None:
        # The subquery renders its own WITH inside the parentheses. Its step
        # names shadow the outer ones rather than colliding with them.
        inner = orders.filter(col("country") == "nl").sort(col("amount")).limit(1).select(col("id"))
        sql = orders.filter(col("id").isin(inner)).render()
        assert sql.count("WITH") == 2
        assert [row[0] for row in orders.filter(col("id").isin(inner)).fetchall()] == [1]


class TestSinks:
    def test_create_stores_the_rows(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        assert orders.filter(col("country") == "nl").create("nl") == 3
        assert len(con.table("nl")) == 3

    def test_create_refuses_to_clobber_unless_asked(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        orders.create("copy")
        with pytest.raises(exceptions.CatalogError):
            orders.create("copy")
        assert orders.filter(col("id") == 1).create("copy", replace=True) == 1
        assert len(con.table("copy")) == 1

    def test_a_temporary_table_is_not_in_the_catalog(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        orders.create("scratch", temporary=True)
        assert len(con.table("scratch")) == 5
        listed = con.sql("SELECT temporary FROM duckdb_tables() WHERE table_name = 'scratch'").fetchall()
        assert listed == [(True,)]

    def test_insert_into_appends(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        orders.filter(col("id") == 1).create("some")
        assert orders.filter(col("id") == 2).insert_into("some") == 1
        assert len(con.table("some")) == 2

    def test_a_sink_binds_its_literals(self, con: duckdb.Connection, orders: duckdb.Frame) -> None:
        # The COPY wrapper must not break the sink: the filter's value still
        # has to be bound rather than written into the statement.
        sql, values = orders.filter(col("country") == "nl")._bind(lambda q: f"CREATE TABLE t AS {q}")
        assert "'nl'" not in sql
        assert values == ["nl"]

    def test_to_parquet_round_trips(self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path) -> None:
        path = tmp_path / "orders.parquet"
        assert orders.to_parquet(str(path)) == 5
        assert path.exists()
        assert len(con.sql(f"SELECT * FROM read_parquet('{path}')")) == 5

    def test_to_csv_round_trips(self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path) -> None:
        path = tmp_path / "orders.csv"
        assert orders.to_csv(str(path), header=True) == 5
        assert path.read_text().startswith("id,country,amount")

    def test_copy_options_reach_the_writer(self, orders: duckdb.Frame, tmp_path: Path) -> None:
        path = tmp_path / "orders.csv"
        orders.to_csv(str(path), header=False, delimiter="|")
        assert "|" in path.read_text()
        assert not path.read_text().startswith("id")

    def test_an_option_name_that_is_not_a_name_is_refused(self, orders: duckdb.Frame, tmp_path: Path) -> None:
        # Option names are bare words in the SQL and cannot be quoted, so the
        # only defence is to check them.
        with pytest.raises(ValueError, match="copy option name"):
            orders.to_csv(str(tmp_path / "x.csv"), **{"header, ROW_GROUP_SIZE": 1})

    def test_a_quote_in_a_path_cannot_escape(
        self, con: duckdb.Connection, orders: duckdb.Frame, tmp_path: Path
    ) -> None:
        path = tmp_path / "od'; DROP TABLE orders; --.csv"
        orders.to_csv(str(path))
        assert path.exists()
        assert len(con.table("orders")) == 5


class TestTableSource:
    def test_a_table_is_read_by_name(self, con: duckdb.Connection) -> None:
        assert len(con.table("orders")) == 5

    def test_a_name_is_quoted_not_spliced(self, con: duckdb.Connection) -> None:
        # The reason this verb exists: an f-string into sql() would run the
        # second statement.
        with pytest.raises(exceptions.CatalogError):
            con.table("orders; DROP TABLE orders").fetchall()
        assert len(con.table("orders")) == 5

    def test_a_qualified_name_stays_two_identifiers(self, con: duckdb.Connection) -> None:
        assert len(con.table("main.orders")) == 5

    def test_an_awkward_name_needs_no_escaping_by_the_caller(self, con: duckdb.Connection) -> None:
        con.run('CREATE TABLE "select ""x""" AS SELECT 1 AS v')
        assert con.table('select "x"').fetchall() == [(1,)]


class TestDerivedNamesMatchTheEngine:
    """Every derived shape must equal what the binder would have said.

    Deriving names is the whole point of the shape layer, and drifting from the
    engine is the one way it can go wrong. Each case here derives a shape and
    then asks the engine the same question, so a rule that is subtly wrong
    fails rather than silently disagreeing.
    """

    def frames(self, con: duckdb.Connection) -> dict[str, duckdb.Frame]:
        orders, countries = con.table("orders"), con.table("countries")
        lists = con.sql("SELECT 1 AS k, [10, 20] AS xs")
        wide = con.sql("SELECT 'nl' AS c, 1 AS q1, 2 AS q2")
        return {
            "table": orders,
            "sql": con.sql("SELECT 1 AS a, 'b' AS b"),
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
            derived = frame.columns
            truth = [column.name for column in frame._bind_whole()]
            if derived != truth:
                wrong[label] = (derived, truth)
        assert not wrong, f"derived shape disagrees with the binder: {wrong}"

    def test_known_types_agree_with_the_engine(self, con: duckdb.Connection) -> None:
        wrong = {}
        for label, frame in self.frames(con).items():
            truth = dict(frame._bind_whole())
            for column in frame.shape:
                if column.type is not None and truth.get(column.name) != column.type:
                    wrong[f"{label}.{column.name}"] = (column.type, truth.get(column.name))
        assert not wrong, f"carried type disagrees with the binder: {wrong}"


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

    def test_building_a_frame_asks_nothing(self, con: duckdb.Connection) -> None:
        frame = con.table("orders").filter(col("amount") > 100).select("id").sort(col("id"))
        assert frame._cached_shape is None, "building must not reach the engine"

    def test_columns_costs_one_bind_per_source(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = record_binds(con, monkeypatch)
        joined = (
            con.table("orders")
            .filter(col("amount") > 100)
            .join(con.table("countries"), on=col("l.country") == col("r.code"))
            .select("id", "label")
        )
        assert joined.columns == ["id", "label"]
        assert len(calls) == 2, f"one per source, got {len(calls)}: {calls}"

    def test_types_come_along_with_the_source(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        frame = con.table("orders").filter(col("amount") > 100).select("id")
        assert frame.columns == ["id"]  # pays for the source
        calls = record_binds(con, monkeypatch)
        # A column carried through untouched keeps the type the engine gave it,
        # so asking for types again costs nothing.
        assert frame.types == ["INTEGER"]
        assert calls == []

    def test_an_engine_named_column_is_bound_on_a_stub(
        self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        con.table("orders").columns  # pay for the source first  # noqa: B018
        calls = record_binds(con, monkeypatch)
        # Nothing here knows what DuckDB will call this column, so it asks. The
        # question it asks is this step over an empty input, not the whole query.
        frame = con.table("orders").select(sql_expr("amount * 2"))
        assert frame.columns == ["(amount * 2)"]
        assert len(calls) == 2, calls  # the fresh source, then the stub
        assert "WHERE FALSE" in calls[-1]
        assert "NULL::INTEGER" in calls[-1]

    def test_a_stub_does_not_grow_with_the_chain(self, con: duckdb.Connection, monkeypatch: pytest.MonkeyPatch) -> None:
        deep = con.table("orders")
        for _ in range(8):
            deep = deep.filter(col("amount") > 0)
        deep.columns  # noqa: B018
        calls = record_binds(con, monkeypatch)
        assert deep.select(sql_expr("amount * 2")).columns == ["(amount * 2)"]
        # The point of the stub: one step's worth of SQL, however deep the chain.
        assert len(calls) == 1
        assert calls[0].count("SELECT") == 2, calls[0]


class TestJoinRefusesToDuplicateAName:
    """A join carries both sides through, so a shared name would appear twice.

    DuckDB does not catch that once the join is behind a step: it binds the
    later reference to whichever came first and answers. That silent wrong
    answer is what this refuses.
    """

    def test_a_shared_name_is_refused(self, con: duckdb.Connection) -> None:
        with pytest.raises(ValueError, match="both sides of this join have"):
            con.table("orders").join(con.table("orders"), on=col("l.id") == col("r.id"))

    def test_the_message_names_the_columns(self, con: duckdb.Connection) -> None:
        with pytest.raises(ValueError, match=r"'id'.*'country'.*'amount'"):
            con.table("orders").join(con.table("orders"), on=col("l.id") == col("r.id"))

    def test_a_suffix_renames_the_right_side(self, con: duckdb.Connection) -> None:
        joined = con.table("orders").join(con.table("orders"), on=col("l.id") == col("r.id"), suffix="_r")
        assert joined.columns == ["id", "country", "amount", "id_r", "country_r", "amount_r"]
        assert joined.columns == [c.name for c in joined._bind_whole()]

    def test_a_suffixed_column_is_reachable(self, con: duckdb.Connection) -> None:
        joined = con.table("orders").join(con.table("orders"), on=col("l.id") == (col("r.id") - 1), suffix="_next")
        pairs = joined.select("id", "id_next").sort(col("id")).fetchall()
        assert pairs == [(1, 2), (2, 3), (3, 4), (4, 5)]

    def test_a_using_key_is_not_a_clash(self, con: duckdb.Connection) -> None:
        # USING folds its key into one column, so it cannot appear twice.
        left = con.table("orders").select("id", "country")
        right = con.table("countries").rename(code="country")
        assert left.join(right, on="country").columns == ["id", "country", "label"]

    def test_a_semi_join_keeps_only_the_left(self, con: duckdb.Connection) -> None:
        # Nothing from the right survives, so nothing can collide.
        joined = con.table("orders").join(con.table("orders"), on=col("l.id") == col("r.id"), how="semi")
        assert joined.columns == ["id", "country", "amount"]

    def test_disjoint_sides_need_no_suffix(self, con: duckdb.Connection) -> None:
        joined = con.table("orders").join(con.table("countries"), on=col("l.country") == col("r.code"))
        assert joined.columns == ["id", "country", "amount", "code", "label"]
        assert "RENAME" not in joined.render()
