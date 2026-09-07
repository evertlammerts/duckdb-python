"""A string names one thing and is never split; qualification, sides and fields are Python structure."""

from __future__ import annotations

import pathlib
import pickle
import sys
from pathlib import Path

import pytest

import duckdb
from duckdb import (
    Side,
    coalesce,
    col,
    compat,
    count_all,
    exceptions,
    fn,
    lit,
    param,
    sql,
    sql_expr,
    star,
    table,
    table_function,
    values,
)


@pytest.fixture(scope="module")
def con() -> duckdb.Connection:
    connection = duckdb.connect()
    connection.run("CREATE SCHEMA s")
    connection.run("CREATE TABLE s.t AS SELECT 1 AS v")
    connection.run('CREATE TABLE "a.b" AS SELECT 2 AS v')
    connection.run("CREATE MACRO s.twice(x) AS x * 2")
    connection.run("CREATE MACRO s.rng(n) AS TABLE SELECT * FROM range(n)")
    connection.run("""CREATE TABLE shadow AS SELECT 21 AS "hello.world", {'world': 7} AS hello""")
    return connection


@pytest.fixture
def dotted() -> duckdb.Frame:
    return values([(21, 1)], columns=["hello.world", "id"])


class TestAStringIsOneName:
    def test_every_verb_takes_a_dotted_name_whole(self, con: duckdb.Connection, dotted: duckdb.Frame) -> None:
        assert dotted.select("hello.world").rows(con) == [(21,)]
        assert dotted.select(col("hello.world")).rows(con) == [(21,)]
        assert dotted.sort("hello.world").rows(con) == [(21, 1)]
        assert dotted.group_by("hello.world").agg(count_all()).rows(con) == [(21, 1)]
        assert dotted["hello.world"].rows(con) == [(21,)]
        assert dotted.rename(**{"hello.world": "h"}).columns(con) == ["h", "id"]
        assert values([([1, 2],)], columns=["a.b"]).unnest("a.b").rows(con) == [(1,), (2,)]

    def test_column_names_round_trip(self, con: duckdb.Connection, dotted: duckdb.Frame) -> None:
        assert dotted.select(*dotted.columns(con)).rows(con) == dotted.rows(con)

    def test_a_struct_named_like_a_prefix_does_not_shadow(self, con: duckdb.Connection) -> None:
        assert table("shadow").select(col("hello.world")).rows(con) == [(21,)]
        assert table("shadow").select(col("hello")["world"]).rows(con) == [(7,)]

    def test_a_column_is_quoted_whole(self) -> None:
        assert col("hello.world").fragment() == '"hello.world"'
        assert col('say "hi"').fragment() == '"say ""hi"""'
        assert col("order").fragment() == '"order"'

    def test_col_takes_one_name(self) -> None:
        with pytest.raises(TypeError, match="one column name"):
            col(("l", "x"))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="column name"):
            col(3)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="empty"):
            col("")

    def test_using_a_dotted_key(self, con: duckdb.Connection) -> None:
        left = values([(1, 5)], columns=["k.1", "x"])
        right = values([(1, 9)], columns=["k.1", "y"])
        assert left.join(right, on="k.1").rows(con) == [(1, 5, 9)]

    def test_drop_takes_any_name_through_exclude(self, con: duckdb.Connection, dotted: duckdb.Frame) -> None:
        # One form for every name: the engine quotes EXCLUDE's list like any
        # other identifier (duckdb/duckdb#25370), so nothing is special-cased.
        assert dotted.drop("id").render().splitlines()[-1] == 'SELECT * EXCLUDE ("id") FROM "_s0"'
        assert dotted.drop("id").rows(con) == [(21,)]
        assert dotted.drop("hello.world").render().splitlines()[-1] == 'SELECT * EXCLUDE ("hello.world") FROM "_s0"'
        assert dotted.drop("hello.world").rows(con) == [(1,)]
        assert dotted.drop("HELLO.WORLD").rows(con) == [(1,)]
        assert dotted.drop("hello.world").columns(con) == ["id"]
        quoted = values([(1, 2)], columns=['say "hi"', "x"])
        assert quoted.drop('say "hi"').rows(con) == [(2,)]
        with pytest.raises(ValueError, match="drop: no column 'nope'"):
            dotted.drop("nope").rows(con)
        with pytest.raises(ValueError, match="drop: no column 'nope'"):
            dotted.drop("hello.world", "nope").rows(con)

    def test_rename_takes_any_name_through_rename(self, con: duckdb.Connection, dotted: duckdb.Frame) -> None:
        renamed = dotted.rename(**{"hello.world": "h"})
        assert renamed.render().splitlines()[-1] == 'SELECT * RENAME ("hello.world" AS "h") FROM "_s0"'
        assert renamed.columns(con) == ["h", "id"]
        assert renamed.rows(con) == [(21, 1)]
        assert renamed.select("h").rows(con) == [(21,)]
        assert repr(renamed).startswith("<Frame WITH")
        quoted = values([(1, 2)], columns=['say "hi"', "x"])
        assert quoted.rename(**{'say "hi"': "s"}).columns(con) == ["s", "x"]
        assert quoted.rename(**{'say "hi"': "s"}).rows(con) == [(1, 2)]
        with pytest.raises(ValueError, match="rename: no column 'nope'"):
            dotted.rename(nope="x").rows(con)
        con.create_macro("renamed_rows", [], renamed)
        assert sql("SELECT * FROM renamed_rows()").columns(con) == ["h", "id"]

    def test_star_exclude_and_rename_take_any_name(self, con: duckdb.Connection, dotted: duckdb.Frame) -> None:
        assert dotted.select(star(exclude=["id"])).rows(con) == [(21,)]
        assert dotted.select(star(exclude=["hello.world"])).rows(con) == [(1,)]
        assert dotted.select(star(rename={"hello.world": "h"})).columns(con) == ["h", "id"]


class TestQualifiedNamesAreTuples:
    def test_a_tuple_qualifies_a_table(self, con: duckdb.Connection) -> None:
        assert table(("s", "t")).rows(con) == [(1,)]
        assert table(("s", "t")).render() == 'SELECT * FROM "s"."t"'
        assert table(("memory", "s", "t")).rows(con) == [(1,)]

    def test_a_dotted_string_is_one_table(self, con: duckdb.Connection) -> None:
        assert table("a.b").rows(con) == [(2,)]
        with pytest.raises(exceptions.CatalogError, match=r"s\.t"):
            table("s.t").rows(con)

    def test_a_file_path_is_one_name(self, con: duckdb.Connection, tmp_path: Path) -> None:
        path = tmp_path / "v1.2" / "report.2026.csv"
        path.parent.mkdir()
        path.write_text("id\n8\n")
        assert table(str(path)).render() == f'SELECT * FROM "{path}"'
        assert table(str(path)).rows(con) == [(8,)]

    def test_names_are_validated_when_built(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            table(())
        with pytest.raises(ValueError, match="empty"):
            table(("s", ""))
        with pytest.raises(TypeError, match="tuple of strings"):
            table(("s", 1))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tuple of strings"):
            table(1)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="empty"):
            fn(("s", ""), 1)

    def test_create_and_insert_into_take_tuples(self, con: duckdb.Connection) -> None:
        rows = values([(1,)], columns=["v"])
        assert rows.create(con, ("s", "made")) == 1
        assert rows.insert_into(con, ("s", "made")) == 1
        assert table(("s", "made")).count(con) == 2
        assert rows.create(con, "x.y") == 1
        assert sql('SELECT * FROM "x.y"').rows(con) == [(1,)]

    def test_create_macro_takes_a_tuple(self, con: duckdb.Connection) -> None:
        con.create_macro(("s", "thrice"), ["x"], col("x") * 3)
        assert sql("SELECT s.thrice(4)").rows(con) == [(12,)]

    def test_function_names_are_written_as_the_engine_writes_identifiers(self, con: duckdb.Connection) -> None:
        assert fn("sum", col("x")).fragment() == 'sum("x")'
        assert fn("my func", col("x")).fragment() == '"my func"("x")'
        assert fn("my.func", col("x")).fragment() == '"my.func"("x")'
        assert fn(("util", "f"), col("x")).fragment() == 'util.f("x")'
        assert col("s").str().upper().fragment() == 'upper("s")'
        assert values([(1,)], columns=["v"]).select(fn(("s", "twice"), col("v") + 3)).rows(con) == [(8,)]

    def test_a_keyword_as_a_function_name_is_quoted(self, con: duckdb.Connection) -> None:
        assert fn("select", col("x")).fragment() == '"select"("x")'
        assert fn("filter", col("x")).fragment() == '"filter"("x")'
        con.run('CREATE MACRO "select"(x) AS x + 1')
        assert values([(1,)], columns=["v"]).select(fn("select", col("v"))).rows(con) == [(2,)]

    def test_coalesce_is_syntax_not_a_function(self, con: duckdb.Connection) -> None:
        nulls = values([(None,)], columns=[("v", "INTEGER")])
        assert coalesce(col("v"), 0).fragment() == 'COALESCE("v", 0)'
        assert nulls.select(coalesce(col("v"), 0)).rows(con) == [(0,)]
        with pytest.raises(exceptions.CatalogError, match="coalesce"):
            nulls.select(fn("coalesce", col("v"), 0)).rows(con)

    def test_table_functions_follow_the_same_rule(self, con: duckdb.Connection) -> None:
        assert table_function("range", 3).render() == 'SELECT * FROM "range"(3)'
        assert table_function("range", 3).rows(con) == [(0,), (1,), (2,)]
        assert table_function(("s", "rng"), 2).rows(con) == [(0,), (1,)]

    def test_the_keyword_module_is_current(self, con: duckdb.Connection) -> None:
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
        from gen_keywords import render

        committed = (pathlib.Path(__file__).parent.parent / "src" / "duckdb" / "_keywords.py").read_text()
        assert committed == render(con), "run scripts/gen_keywords.py"

    def test_a_function_registered_from_python_has_a_bare_name(self, con: duckdb.Connection) -> None:
        with pytest.raises(TypeError, match="bare name"):
            con.create_function(("s", "f"), lambda x: x, ["BIGINT"], "BIGINT")  # type: ignore[arg-type]


class TestBracketsReachIntoValues:
    def test_struct_list_and_map(self, con: duckdb.Connection) -> None:
        source = sql("SELECT {'a': {'b': 1}, 'x.y': 5} AS st, [10, 20] AS xs, map {'k': 7} AS m")
        picked = source.select(col("st")["a"]["b"], col("st")["x.y"], col("xs")[1], col("m")["k"])
        assert picked.rows(con) == [(1, 5, 10, 7)]
        assert col("st")["a"]["b"].fragment() == "\"st\"['a']['b']"
        assert col("xs")[1].fragment() == '"xs"[1]'

    def test_keys_are_names_or_positions(self) -> None:
        with pytest.raises(TypeError, match="field name or a position"):
            col("x")[1.5]
        with pytest.raises(TypeError, match="field name or a position"):
            col("x")[True]

    def test_an_expression_is_not_a_sequence(self) -> None:
        with pytest.raises(TypeError, match="not iterable"):
            list(col("x"))
        with pytest.raises(TypeError, match="not iterable"):
            assert "a" in col("x")

    def test_a_plan_bracket_still_narrows_the_plan(self, con: duckdb.Connection, dotted: duckdb.Frame) -> None:
        narrowed = dotted["hello.world"]
        assert isinstance(narrowed, duckdb.Frame)
        assert narrowed.columns(con) == ["hello.world"]

    def test_an_index_pickles(self) -> None:
        plan = sql("SELECT {'a': 1} AS st").select(col("st")["a"])
        assert pickle.loads(pickle.dumps(plan)).render() == plan.render()


class TestJoinSidesAreTheConditionsArguments:
    def test_the_condition_is_a_function_of_both_sides(self, con: duckdb.Connection) -> None:
        orders = values([(1, "acme")], columns=["id", "customer"])
        customers = values([("acme", 7)], columns=["name", "score"])
        joined = orders.join(customers, on=lambda left, right: left["customer"] == right["name"])
        assert joined.rows(con) == [(1, "acme", "acme", 7)]
        assert 'ON ("l"."customer" = "r"."name")' in joined.render()

    def test_sides_take_dotted_names_and_fields(self, con: duckdb.Connection) -> None:
        orders = values([(1, {"zip": "1011"})], columns=["id", "shipping"])
        zones = values([("1011", "centre")], columns=["zip.code", "zone"])
        joined = orders.join(zones, on=lambda left, right: left["shipping"]["zip"] == right["zip.code"])
        assert joined.select("id", "zone").rows(con) == [(1, "centre")]

    def test_the_condition_runs_once_on_side_objects(self, con: duckdb.Connection) -> None:
        calls: list[tuple[type, type]] = []

        def same_id(left: Side, right: Side) -> duckdb.Expr:
            calls.append((type(left), type(right)))
            return left["id"] == right["id"]

        rows = values([(1,)], columns=["id"])
        joined = rows.join(rows, on=same_id, suffix="_r")
        assert calls == [(Side, Side)]
        assert joined.rows(con) == [(1, 1)]
        assert joined.rows(con) == [(1, 1)]
        assert calls == [(Side, Side)]

    def test_a_side_has_brackets_and_nothing_else(self) -> None:
        rows = values([(1,)], columns=["id"])
        with pytest.raises(AttributeError, match=r"\['id'\]"):
            rows.join(rows, on=lambda left, right: left.id == right.id)
        with pytest.raises(TypeError, match="column name"):
            rows.join(rows, on=lambda left, right: left[0] == right[0])
        assert Side("l").alias == "l"

    def test_using_takes_names_not_expressions(self) -> None:
        rows = values([(1,)], columns=["id"])
        with pytest.raises(TypeError, match="USING join takes column names"):
            rows.join(rows, on=[col("id")])  # type: ignore[list-item]

    def test_a_suffixed_join_on_a_dotted_key(self, con: duckdb.Connection) -> None:
        left = values([(1, 5)], columns=["k.1", "x"])
        right = values([(1, 9)], columns=["k.1", "x"])
        joined = left.join(right, on="k.1", suffix="_r")
        assert joined.columns(con) == ["k.1", "x", "x_r"]
        assert joined.rows(con) == [(1, 5, 9)]

    def test_the_condition_must_be_an_expression(self) -> None:
        rows = values([(1,)], columns=["id"])
        with pytest.raises(TypeError, match="not an expression"):
            rows.join(rows, on=lambda left, right: True)

    def test_a_value_from_outside_binds(self, con: duckdb.Connection) -> None:
        floor = 5
        left = values([(1, 3), (2, 9)], columns=["id", "v"])
        right = values([(1,), (2,)], columns=["id"])
        joined = left.join(right, on=lambda a, b: (a["id"] == b["id"]) & (a["v"] > floor), suffix="_r")
        assert joined.select("id").rows(con) == [(2,)]

    def test_a_dotted_side_name_is_a_column_now(self, con: duckdb.Connection) -> None:
        rows = values([(1,)], columns=["id"])
        with pytest.raises(exceptions.Error, match=r"l\.id"):
            rows.join(rows, on=col("l.id") == col("r.id"), suffix="_r").rows(con)

    def test_raw_sql_names_the_sides_itself(self, con: duckdb.Connection) -> None:
        rows = values([(1,)], columns=["id"])
        assert rows.join(rows, on=sql_expr("l.id = r.id"), suffix="_r").rows(con) == [(1, 1)]

    def test_a_joined_plan_pickles(self) -> None:
        left = values([(1,)], columns=["id"])
        right = values([(1,)], columns=["k"])
        plan = left.join(right, on=lambda a, b: a["id"] == b["k"])
        assert pickle.loads(pickle.dumps(plan)).render() == plan.render()


class TestCompatKeepsTheOldSplit:
    @staticmethod
    def rendered(text: str) -> str:
        return compat.ColumnExpression(text)._value().fragment()

    def test_a_dotted_column_expression_is_qualified(self) -> None:
        assert self.rendered("a.b") == '"a"."b"'
        assert self.rendered('"a.b"') == '"a.b"'
        assert self.rendered('a."b.c".d') == '"a"."b.c"."d"'
        assert self.rendered('"x""y"') == '"x""y"'
        assert self.rendered("w.x.y.z") == '"w"."x"."y"."z"'
        assert compat.ColumnExpression("a", "b")._value().fragment() == '"a"."b"'
        assert compat.ColumnExpression("*")._value().fragment() == "*"

    def test_the_engine_grammar_and_its_words(self) -> None:
        for text, words in [
            ('a"b', "Unexpected quote in the middle"),
            ('"a', "Unterminated quote"),
            ('""', "Zero-length delimited identifier"),
            ('"a"b', "Unexpected character after a quoted identifier"),
        ]:
            with pytest.raises(exceptions.ParserError, match=words):
                compat.ColumnExpression(text)
        with pytest.raises(exceptions.InvalidInputError, match="needs a column name"):
            compat.ColumnExpression()
        with pytest.raises(exceptions.InvalidInputError, match="empty"):
            compat.ColumnExpression("")
        assert compat.ColumnExpression("a", None)._value().fragment() == '"a"."None"'  # type: ignore[arg-type]

    def test_a_malformed_name_raises_an_old_client_error(self) -> None:
        connection = compat.connect()
        connection.run("CREATE TABLE t AS SELECT 1 AS v")
        with pytest.raises(exceptions.Error):
            connection.table("a..b").fetchall()
        made = connection.sql("SELECT 1 AS v")
        assert made is not None
        with pytest.raises(exceptions.InvalidInputError, match="empty"):
            made.create("a..b")

    def test_update_says_the_closed_words_first(self) -> None:
        connection = compat.connect()
        connection.run("CREATE TABLE t AS SELECT 1 AS v")
        relation = connection.table("t")
        connection.close()
        with pytest.raises(compat.ConnectionException, match="closed"):
            relation.update({"v": 2})

    def test_a_dotted_table_name_is_qualified(self) -> None:
        connection = compat.connect()
        connection.run("CREATE SCHEMA s")
        connection.run("CREATE TABLE s.t AS SELECT 1 AS v")
        connection.run('CREATE TABLE "a.b" AS SELECT 2 AS v')
        connection.run("CREATE MACRO s.rng(n) AS TABLE SELECT * FROM range(n)")
        assert connection.table("s.t").fetchall() == [(1,)]
        assert connection.table('"a.b"').fetchall() == [(2,)]
        assert connection.table_function("s.rng", [2]).fetchall() == [(0,), (1,)]
        made = connection.sql("SELECT 3 AS v")
        assert made is not None
        made.create("s.made")
        made.insert_into("s.made")
        assert connection.table("s.made").fetchall() == [(3,), (3,)]


class TestNamesCompareAsTheEngineDoes:
    def test_case_is_not_a_difference(self, con: duckdb.Connection) -> None:
        rows = values([(1, 2)], columns=["Total", "id"])
        assert rows.select("TOTAL").rows(con) == [(1,)]
        assert rows.drop("ID").columns(con) == ["Total"]
        assert rows.drop("ID").rows(con) == [(1,)]
        assert rows.rename(TOTAL="t").columns(con) == ["t", "id"]
        assert rows.with_columns(total=col("id") + 1).columns(con) == ["Total", "id"]

    def test_two_spellings_of_one_name_are_refused(self, con: duckdb.Connection) -> None:
        with pytest.raises(ValueError, match="Total"):
            values([(1, 2)], columns=["Total", "total"]).columns(con)
        left = values([(1, 2)], columns=["id", "Total"])
        right = values([(1, 3)], columns=["id", "total"])
        with pytest.raises(ValueError, match="suffix"):
            left.join(right, on="id").columns(con)
        assert left.join(right, on="id", suffix="_r").columns(con) == ["id", "Total", "total_r"]

    def test_only_ascii_case_folds(self) -> None:
        from duckdb.expr import fold_name

        assert fold_name("PRICE.USD") == "price.usd"
        assert fold_name("STRASSE") == "strasse"
        assert fold_name("straße") == "straße"
        assert fold_name("ÉCOLE") == "École".replace("É", "É")


class TestFloatsAreDoubles:
    def test_a_float_is_a_double_inline_and_bound(self, con: duckdb.Connection) -> None:
        one = sql("SELECT 1 AS id")
        for value in (1.5, 0.1, [1.5], {"a": 1.5}):
            plan = one.select(lit(value).alias("v"))
            declared = plan.types(con)[0]
            assert "DOUBLE" in declared, (value, declared)
            got = plan.rows(con)[0][0]
            leaf = got[0] if isinstance(got, list) else got["a"] if isinstance(got, dict) else got
            assert isinstance(leaf, float), (value, got)
        assert (col("p") * 1.21).fragment() == '("p" * 1.21::DOUBLE)'
        assert one.select(param("x").alias("v")).rows(con, parameters={"x": 1.5}) == [(1.5,)]


class TestTheParameterizedForm:
    def test_render_parameterized_is_what_executes(self, con: duckdb.Connection) -> None:
        orders = values([(1, "F"), (2, "O")], columns=["id", "status"])
        plan = orders.filter((col("status") == "F") & (col("id") == param("wanted")))
        text, bound = plan.render_parameterized(con, parameters={"wanted": 1})
        assert text.splitlines()[-1] == 'SELECT * FROM "_s0" WHERE (("status" = $3) AND ("id" = $4))'
        assert bound == ["F", "O", "F", 1]
        assert plan.on(con).render_parameterized(parameters={"wanted": 1}) == (text, bound)
        assert "$" not in plan.render(con)
        with pytest.raises(ValueError, match="no value for parameter 'wanted'"):
            plan.render_parameterized(con)
