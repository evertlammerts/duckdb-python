"""The migration face: the old client's connection vocabulary over the new seam.

Parity with the old client is measured by the adopted suite under compat/;
these tests gate the face itself.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import pytest

import duckdb
from duckdb import compat, exceptions

if TYPE_CHECKING:
    from pathlib import Path


def rel(source: compat.CompatConnection, query: str) -> compat.CompatRelation:
    relation = source.sql(query)
    assert relation is not None
    return relation


class TestExecuteAndFetch:
    def test_execute_fetch_family(self) -> None:
        con = compat.connect()
        assert con.execute("SELECT 42").fetchall() == [(42,)]
        con.execute("SELECT i FROM range(5) t(i)")
        assert con.fetchone() == (0,)
        assert con.fetchmany(2) == [(1,), (2,)]
        assert con.fetchall() == [(3,), (4,)]
        assert con.fetchone() is None

    def test_execute_returns_self_and_replaces_the_held_result(self) -> None:
        con = compat.connect()
        assert con.execute("SELECT 1") is con
        con.execute("SELECT 2")
        assert con.fetchall() == [(2,)]

    def test_fetchmany_defaults_to_one(self) -> None:
        con = compat.connect()
        con.execute("SELECT i FROM range(3) t(i)")
        assert con.fetchmany() == [(0,)]
        assert con.fetchmany(0) == []

    def test_fetchnumpy_after_fetchone_returns_the_remaining_rows(self) -> None:
        con = compat.connect()
        con.execute("SELECT i FROM range(5) t(i)")
        assert con.fetchone() == (0,)
        assert list(con.fetchnumpy()["i"]) == [1, 2, 3, 4]

    def test_a_statement_without_rows_applies_at_execute(self) -> None:
        # The old client's INSERT took effect at execute, with no fetch step.
        con = compat.connect()
        con.execute("CREATE TABLE t (v INTEGER)")
        con.execute("INSERT INTO t VALUES (1), (2)")
        assert con.execute("SELECT count(*) FROM t").fetchone() == (2,)

    def test_executemany(self) -> None:
        con = compat.connect()
        con.execute("CREATE TABLE t (v INTEGER)")
        con.executemany("INSERT INTO t VALUES (?)", [[1], [2], [3]])
        assert con.execute("SELECT sum(v) FROM t").fetchone() == (6,)

    def test_parameter_names_coerce_the_old_way(self) -> None:
        # The old client stringified any parameter-dict key; the native seam
        # refuses non-strings, so the leniency lives in this face alone.
        con = compat.connect()
        assert con.execute("SELECT $1 AS a", {1: 5}).fetchall() == [(5,)]
        assert con.execute("SELECT $1 AS a", {b"1": 5}).fetchall() == [(5,)]

    def test_fetch_without_a_result_is_refused(self) -> None:
        con = compat.connect()
        with pytest.raises(exceptions.InvalidInputError, match="No open result"):
            con.fetchall()

    def test_description_of_a_select(self) -> None:
        con = compat.connect()
        con.execute("SELECT 1 AS a, 'x' AS b")
        assert con.description is not None
        assert [d[0] for d in con.description] == ["a", "b"]
        assert con.description[0][1] == "INTEGER"

    def test_no_description_and_no_rowcount_after_ddl(self) -> None:
        con = compat.connect()
        con.execute("CREATE TABLE t (v INTEGER)")
        assert con.description is None
        # The old contract: never a real count. run() and dbapi carry those.
        assert con.rowcount == -1


class TestOldConnectionShape:
    def test_cursor_is_an_independent_duplicate(self) -> None:
        # The old client's cursor() semantics, kept deliberately: its own
        # transaction, unlike duckdb.dbapi's shared-transaction cursors.
        con = compat.connect()
        cur = con.cursor()
        assert isinstance(cur, compat.CompatConnection)
        con.execute("CREATE TABLE t (v INTEGER)")
        con.execute("BEGIN")
        con.execute("INSERT INTO t VALUES (1)")
        assert cur.execute("SELECT count(*) FROM t").fetchone() == (0,)
        con.execute("COMMIT")
        assert cur.execute("SELECT count(*) FROM t").fetchone() == (1,)

    def test_plans_run_on_a_compat_connection(self) -> None:
        # A subclass of the native connection, so the new surface stays whole.
        con = compat.connect()
        con.execute("CREATE TABLE t AS SELECT 7 AS v")
        assert duckdb.table("t").rows(con) == [(7,)]
        with con.transaction():
            con.run("INSERT INTO t VALUES (8)")
        assert duckdb.table("t").count(con) == 2

    def test_read_only_flag(self, tmp_path: Path) -> None:
        path = str(tmp_path / "ro.db")
        setup = compat.connect(path)
        setup.execute("CREATE TABLE t (v INTEGER)")
        setup.close()
        readonly = compat.connect(path, read_only=True)
        assert readonly.execute("SELECT count(*) FROM t").fetchone() == (0,)
        with pytest.raises(exceptions.Error):
            readonly.execute("CREATE TABLE nope (v INTEGER)")

    def test_config_dict(self) -> None:
        con = compat.connect(":memory:", config={"threads": 3})
        assert con.execute("SELECT current_setting('threads')").fetchone() == (3,)

    def test_a_closed_connection_is_refused(self) -> None:
        con = compat.connect()
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            con.execute("SELECT 1")
        # The old fetch answer on a closed connection: the result speaks first.
        with pytest.raises(exceptions.InvalidInputError, match="No open result set"):
            con.fetchall()

    def test_close_releases_a_held_result(self, tmp_path: Path) -> None:
        path = str(tmp_path / "held.db")
        con = compat.connect(path)
        con.execute("CREATE TABLE t AS SELECT * FROM range(10)")
        con.execute("SELECT * FROM t")
        con.close()
        assert compat.connect(path).execute("SELECT count(*) FROM t").fetchone() == (10,)

    def test_close_reaches_every_cursor_past_one_that_raises(self) -> None:
        con = compat.connect()
        first = con.cursor()
        second = con.cursor()
        broken = first if next(iter(con._cursors)) is first else second
        intact = second if broken is first else first

        def boom() -> None:
            message = "cursor close failed"
            raise RuntimeError(message)

        broken.close = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="cursor close failed"):
            con.close()
        # The failure did not stop the sweep: the other cursor and the
        # connection itself are closed all the same.
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            intact.execute("SELECT 1")
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            con.execute("SELECT 1")

    def test_closing_a_connection_closes_its_cursors(self) -> None:
        # The old lifetime coupling, kept: cursors die with their parent.
        con = compat.connect()
        cursor = con.cursor()
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            cursor.execute("SELECT 1")

    def test_transaction_methods(self) -> None:
        con = compat.connect()
        con.execute("CREATE TABLE t (v INTEGER)")
        con.begin()
        con.execute("INSERT INTO t VALUES (1)")
        con.rollback()
        assert con.execute("SELECT count(*) FROM t").fetchone() == (0,)
        con.begin()
        con.execute("INSERT INTO t VALUES (1)")
        con.commit()
        assert con.execute("SELECT count(*) FROM t").fetchone() == (1,)


class TestCompatRelation:
    def test_old_fetch_model(self) -> None:
        # A bare fetchall re-runs each call; fetchone and fetchmany hold and
        # drain one result; execute() resets it.
        con = compat.connect()
        relation = rel(con, "SELECT * FROM range(3)")
        assert relation.fetchall() == [(0,), (1,), (2,)]
        assert relation.fetchall() == [(0,), (1,), (2,)]
        assert relation.fetchone() == (0,)
        assert relation.fetchmany(2) == [(1,), (2,)]
        assert relation.fetchone() is None
        assert relation.fetchall() == []
        relation.execute()
        assert relation.fetchone() == (0,)

    def test_verbs_compose_as_text(self) -> None:
        con = compat.connect()
        con.execute("CREATE TABLE t AS SELECT i, i % 2 AS parity FROM range(10) r(i)")
        odd = con.table("t").filter("parity = 1").project("i")
        assert odd.aggregate("sum(i)").fetchone() == (25,)
        aliased = con.table("t").set_alias("s").filter("s.i < 3")
        assert len(aliased.fetchall()) == 3
        renamed = rel(con, "SELECT 1 AS a").query("v", "SELECT a + 1 FROM v")
        assert renamed is not None
        assert renamed.fetchone() == (2,)

    def test_close_speaks_the_old_words(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT 1").execute()
        relation.close()
        with pytest.raises(exceptions.InvalidInputError, match="result closed"):
            relation.fetchone()
        with pytest.raises(exceptions.InvalidInputError, match="result closed"):
            relation.fetchall()

    def test_description_without_running(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT 1 AS a, 'x' AS b")
        assert [d[:2] for d in relation.description] == [("a", "INTEGER"), ("b", "VARCHAR")]

    def test_a_relation_keeps_its_cursor_alive(self) -> None:
        con = compat.connect()
        assert rel(con.cursor(), "SELECT 1 AS foo").fetchall() == [(1,)]

    def test_fetchdf_after_fetchone_returns_the_remaining_rows(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT i FROM range(5) t(i)").execute()
        assert relation.fetchone() == (0,)
        assert list(relation.fetchdf()["i"]) == [1, 2, 3, 4]

    def test_a_failed_conversion_does_not_orphan_the_held_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from duckdb import _numpy

        def boom(result: object) -> dict[str, object]:
            message = "conversion failed"
            raise RuntimeError(message)

        monkeypatch.setattr(_numpy, "fetch_numpy", boom)
        con = compat.connect()
        relation = rel(con, "SELECT 1 AS x").execute()
        with pytest.raises(RuntimeError, match="conversion failed"):
            relation.fetchnumpy()
        # The engine allows one live result per connection: an orphaned open
        # result would refuse this statement.
        assert con.execute("SELECT 2").fetchone() == (2,)

    def test_a_connection_close_marks_relations_closed_the_compat_way(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT 42 AS x").execute()
        con.close()
        with pytest.raises(exceptions.InvalidInputError, match="result closed"):
            relation.fetchall()
        with pytest.raises(exceptions.InvalidInputError, match="result closed"):
            relation.fetchone()

    def test_description_after_close_raises_like_the_fetches(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT 1 AS a")
        relation.close()
        with pytest.raises(exceptions.InvalidInputError, match="result closed"):
            _ = relation.description


class TestSharedInstances:
    def test_two_connections_to_one_file_share_the_database(self, tmp_path: Path) -> None:
        path = str(tmp_path / "shared.db")
        first = compat.connect(path)
        second = compat.connect(path)
        first.execute("CREATE TABLE a AS SELECT 1 AS v")
        second.execute("CREATE TABLE b AS SELECT 2 AS v")
        assert first.execute("SELECT count(*) FROM b").fetchone() == (1,)
        first.close()
        second.close()
        # The instance died with its last connection, so this opens afresh.
        third = compat.connect(path)
        assert third.execute("SELECT count(*) FROM a").fetchone() == (1,)

    def test_a_different_configuration_is_refused_in_the_old_words(self, tmp_path: Path) -> None:
        path = str(tmp_path / "config.db")
        held = compat.connect(path)
        with pytest.raises(compat.ConnectionException, match="different configuration than existing"):
            compat.connect(path, read_only=True)
        held.close()

    def test_a_cursor_keeps_the_shared_instance_alive(self, tmp_path: Path) -> None:
        path = str(tmp_path / "held.db")
        con = compat.connect(path)
        con.execute("CREATE TABLE t AS SELECT 1 AS v")
        cursor = con.cursor()
        del con
        gc.collect()
        # The cursor holds the instance, so this joins it instead of hitting
        # the engine's already-attached refusal.
        again = compat.connect(path)
        assert again.execute("SELECT count(*) FROM t").fetchone() == (1,)
        again.close()
        cursor.close()

    def test_a_cursor_keeps_a_named_memory_database_alive(self) -> None:
        con = compat.connect(":memory:compat_cursor_hold")
        con.execute("CREATE TABLE t AS SELECT 7 AS v")
        cursor = con.cursor()
        del con
        gc.collect()
        # A dead instance here would silently hand back a fresh empty database.
        again = compat.connect(":memory:compat_cursor_hold")
        assert again.execute("SELECT v FROM t").fetchone() == (7,)
        again.close()
        cursor.close()

    def test_a_path_and_the_equal_string_share_the_database(self, tmp_path: Path) -> None:
        # One instance per file, however the file was spelled: a second
        # instance would be refused by the engine as already open.
        path = tmp_path / "typed.db"
        by_string = compat.connect(str(path))
        by_path = compat.connect(path)
        by_string.execute("CREATE TABLE a AS SELECT 1 AS v")
        assert by_path.execute("SELECT count(*) FROM a").fetchone() == (1,)
        by_string.close()
        by_path.close()

    def test_named_memory_is_shared_and_plain_memory_is_not(self) -> None:
        named = compat.connect(":memory:compat_gate")
        sibling = compat.connect(":memory:compat_gate")
        named.execute("CREATE TABLE t AS SELECT 1 AS v")
        assert sibling.execute("SELECT count(*) FROM t").fetchone() == (1,)
        loner = compat.connect(":memory:")
        with pytest.raises(exceptions.CatalogError):
            loner.execute("SELECT count(*) FROM t")
        named.close()
        sibling.close()


class TestModuleSurface:
    def test_execute_rides_one_default_connection(self) -> None:
        compat.execute("CREATE OR REPLACE TABLE compat_default AS SELECT 7 AS v")
        assert compat.execute("SELECT v FROM compat_default").fetchone() == (7,)
        assert rel(compat.default_connection(), "SELECT v FROM compat_default").fetchone() == (7,)
        module_query = compat.query("SELECT 1")
        assert module_query is not None
        assert module_query.fetchone() == (1,)
        module_from = compat.from_query("SELECT 2")
        assert module_from is not None
        assert module_from.fetchone() == (2,)
        assert compat.description() is not None

    def test_the_old_exception_names_point_at_the_new_classes(self) -> None:
        assert compat.CatalogException is exceptions.CatalogError
        assert compat.ConnectionException is exceptions.InterfaceError
        assert compat.InvalidInputException is exceptions.InvalidInputError
        assert compat.InterruptException is exceptions.InterruptError

    def test_the_connection_announces_the_old_client_api(self) -> None:
        con = compat.connect()
        row = con.execute("SELECT value FROM duckdb_settings() WHERE name = 'duckdb_api'").fetchone()
        assert row is not None
        assert row[0].startswith("python/")


class TestParityReport:
    def test_a_teardown_failure_is_not_counted_as_a_pass(self, tmp_path: Path) -> None:
        import importlib.util
        from pathlib import Path as RuntimePath

        script = RuntimePath(__file__).resolve().parents[1] / "scripts" / "compat_report.py"
        spec = importlib.util.spec_from_file_location("compat_report_under_test", script)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        target = tmp_path / "test_teardown_fails.py"
        target.write_text(
            "import pytest\n"
            "@pytest.fixture\n"
            "def broken():\n"
            "    yield\n"
            "    raise RuntimeError('teardown boom')\n"
            "def test_passes_then_teardown_fails(broken):\n"
            "    assert True\n"
            "def test_clean():\n"
            "    assert True\n"
        )
        recorder = module.Recorder()
        pytest.main([str(target), "-q", "--tb=no", "-p", "no:cacheprovider"], plugins=[recorder])
        assert recorder.outcomes["passed"] == 1
        assert recorder.outcomes["behavior"] == 1
        assert any("teardown boom" in signature for signature in recorder.signatures)


class TestStatementDispatch:
    # sql() classifies by the parsed statement's own type, never by text.

    def test_a_cte_fronted_insert_executes_once(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        assert con.sql("WITH x AS (SELECT 42 AS v) INSERT INTO t SELECT * FROM x") is None
        assert duckdb.table("t").count(con) == 1

    def test_returning_inside_a_string_literal_is_data(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE s (x VARCHAR)")
        con.run("INSERT INTO s VALUES ('old')")
        assert con.sql("UPDATE s SET x = 'x returning y'") is None
        assert duckdb.table("s").rows(con) == [("x returning y",)]

    def test_returning_dml_materializes_and_runs_once(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        relation = con.sql("INSERT INTO t VALUES (5) RETURNING v")
        assert relation is not None
        assert relation.type == "MATERIALIZED_RELATION"
        assert relation.fetchall() == [(5,)]
        assert relation.fetchall() == [(5,)]
        assert duckdb.table("t").count(con) == 1
        # Verbs compose over the materialized rows, as they did on the old
        # client's result-backed relations.
        assert relation.filter("v > 1").fetchall() == [(5,)]

    def test_selects_stay_lazy_and_leave_no_result_open(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT 1 AS a")
        assert relation is not None
        # A second statement runs immediately: sql() closed its
        # classification result before returning.
        assert con.run("SELECT 1") == 0

    def test_ddl_and_transaction_control_run_on_the_spot(self) -> None:
        con = compat.connect()
        assert con.sql("CREATE TABLE u (x INTEGER)") is None
        assert con.sql("BEGIN") is None
        con.run("INSERT INTO u VALUES (1)")
        assert con.sql("ROLLBACK") is None
        assert duckdb.table("u").count(con) == 0

    def test_sql_discards_an_untouched_held_result_like_the_old_client(self) -> None:
        con = compat.connect()
        con.execute("SELECT * FROM range(10000)")
        relation = con.sql("SELECT 1")
        assert relation is not None
        assert relation.fetchall() == [(1,)]
        # The old client's silent discard: nothing had been fetched, so the
        # held result is gone rather than the statement refused.
        with pytest.raises(exceptions.InvalidInputError, match="No open result set"):
            con.fetchall()

    def test_sql_refuses_loudly_over_a_touched_held_result(self) -> None:
        # A recorded divergence: the old client kept a touched result because
        # it had materialized it; this client streams and says so.
        con = compat.connect()
        con.execute("SELECT * FROM range(10000)")
        assert con.fetchone() == (0,)
        # The engine reports it under RESOURCE_IN_USE, the code a file held
        # by another connection gets, which maps to OperationalError.
        with pytest.raises(exceptions.OperationalError, match="live result"):
            con.sql("SELECT 1")
        # The held result is untouched by the refusal.
        assert con.fetchone() == (1,)

    def test_returning_decimals_keep_width_and_scale(self) -> None:
        import decimal

        con = compat.connect()
        con.run("CREATE TABLE d (v DECIMAL(38,6))")
        relation = con.sql("INSERT INTO d VALUES (-0.000001) RETURNING v")
        assert relation is not None
        assert relation.fetchall() == [(decimal.Decimal("-0.000001"),)]

    def test_executemany_refuses_an_empty_set_in_the_old_words(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE t (v INTEGER)")
        with pytest.raises(exceptions.InvalidInputError, match="non-empty list of parameter sets"):
            con.executemany("INSERT INTO t VALUES (?)", [])


class TestRelationVerbs:
    # One gating behavior per verb family; the adopted suite measures, this gates.

    def test_aggregate_shorthand_with_groups_and_projection(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE a AS SELECT unnest([1, 1, 2]) AS g, unnest([10, 20, 30]) AS v")
        rows = con.table("a").sum("v", groups="g", projected_columns="g").order("g").fetchall()
        assert rows == [(1, 30), (2, 30)]

    def test_window_shorthand(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE w AS SELECT unnest([1, 1, 2]) AS g, unnest([3, 1, 2]) AS v")
        rows = con.table("w").row_number("over (partition by g order by v)", "g, v").order("g, v").fetchall()
        assert rows == [(1, 1, 1), (1, 3, 2), (2, 2, 1)]

    def test_join_using_on_and_literal_condition(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE l AS SELECT 1 AS i")
        con.run("CREATE TABLE r AS SELECT 1 AS i, 2 AS j")
        using = con.table("l").join(con.table("r").set_alias("rr"), "i").fetchall()
        assert using == [(1, 2)]
        on = con.table("l").set_alias("a").join(con.table("r").set_alias("b"), "a.i = b.i").fetchall()
        assert on == [(1, 1, 2)]
        # A boolean literal is a condition, never a USING column.
        cross_ish = con.table("l").set_alias("c").join(con.table("r").set_alias("d"), "true").fetchall()
        assert cross_ish == [(1, 1, 2)]

    def test_union_keeps_duplicates(self) -> None:
        con = compat.connect()
        one = con.sql("SELECT 1 AS v")
        assert one is not None
        assert sorted(one.union(one.set_alias("again")).fetchall()) == [(1,), (1,)]

    def test_describe_shape(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE d AS SELECT 1 AS n, 'x' AS s")
        rows = con.table("d").describe().fetchall()
        assert [row[0] for row in rows] == ["count", "mean", "stddev", "min", "max", "median"]
        assert rows[0][1] == 1.0
        assert rows[0][2] == "1"

    def test_create_insert_into_and_insert(self) -> None:
        con = compat.connect()
        source = con.sql("SELECT 7 AS v")
        assert source is not None
        source.create("made")
        source.insert_into("made")
        con.table("made").insert([8])
        assert sorted(con.table("made").fetchall()) == [(7,), (7,), (8,)]

    def test_explain_analyze_differs_from_standard(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT sum(range) FROM range(100)")
        assert relation is not None
        standard = relation.explain()
        analyzed = relation.explain(type="analyze")
        assert standard != analyzed

    def test_view_relations_say_so(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE vt AS SELECT 1 AS v")
        con.run("CREATE VIEW vv AS SELECT * FROM vt")
        assert con.view("vv").type == "VIEW_RELATION"
        assert con.table("vt").type == "TABLE_RELATION"

    def test_comments_in_aggregate_operands_fail_loudly(self) -> None:
        # A recorded divergence: the old parser round-trip discarded the
        # comment silently; here the operand quote-falls-back and the bind
        # names it. Literals containing -- are data and keep working.
        con = compat.connect()
        relation = con.sql("SELECT 'a--b' AS s, 1 AS v")
        assert relation is not None
        with pytest.raises(exceptions.ProgrammingError, match="not found"):
            relation.sum("v -- a trailing comment")
        assert relation.max("case when s = 'a--b' then 1 else 0 end").fetchall() == [(1,)]


class TestDogfoodedVerbs:
    # The writers ride the native COPY sinks; the expression face rides the
    # native expression layer. These gate the translation boundaries.

    def test_write_csv_roundtrips_through_the_native_sink(self, tmp_path: Path) -> None:
        con = compat.connect()
        con.run("CREATE TABLE w AS SELECT i AS a, 'v' || i AS b FROM range(3) t(i)")
        target = str(tmp_path / "out.csv")
        con.table("w").write_csv(target, sep="|", header=True)
        back = con.table_function("read_csv", [target]).order("a").fetchall()
        assert back == [(0, "v0"), (1, "v1"), (2, "v2")]

    def test_write_parquet_roundtrips(self, tmp_path: Path) -> None:
        con = compat.connect()
        con.run("CREATE TABLE p AS SELECT i AS a FROM range(4) t(i)")
        target = str(tmp_path / "out.parquet")
        con.table("p").to_parquet(target, compression="zstd")
        assert con.table_function("read_parquet", [target]).sum("a").fetchone() == (6,)

    def test_update_with_expression_and_condition(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE u AS SELECT i AS a, 0 AS b FROM range(4) t(i)")
        con.table("u").update({"b": compat.ColumnExpression("a") + 10}, condition=compat.SQLExpression("a >= 2"))
        assert con.table("u").order("a").fetchall() == [(0, 0), (1, 0), (2, 12), (3, 13)]

    def test_select_types_and_contains(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT 1::INTEGER AS i, 'x' AS s, 2.5::DOUBLE AS d")
        assert relation is not None
        assert relation.select_types(["INTEGER", "DOUBLE"]).columns == ["i", "d"]
        assert "s" in relation
        assert "nope" not in relation

    def test_fetch_df_chunk_walks_the_held_result(self) -> None:
        con = compat.connect()
        con.execute("SELECT i FROM range(5000) t(i)")
        first = con.fetch_df_chunk()
        rest = con.fetch_df_chunk(10)
        tail = con.fetch_df_chunk()
        assert len(first) == 2048
        assert len(first) + len(rest) == 5000
        assert len(tail) == 0
        assert list(tail.columns) == ["i"]


class TestExpressionCompat:
    def test_expression_join_condition(self) -> None:
        con = compat.connect()
        con.run("CREATE TABLE ea AS SELECT 1 AS x")
        con.run("CREATE TABLE eb AS SELECT 1 AS x, 9 AS y")
        expr = compat.ColumnExpression("ea.x") == compat.ColumnExpression("eb.x")
        assert con.table("ea").join(con.table("eb"), expr).fetchall() == [(1, 1, 9)]

    def test_strings_are_values_like_the_old_expression_objects(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT 'active' AS state")
        assert relation is not None
        keep = relation.filter(compat.ColumnExpression("state") == "active")
        assert keep.fetchall() == [("active",)]

    def test_case_expression_chains(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT unnest([1, 2, 3]) AS v")
        assert relation is not None
        expr = (
            compat.CaseExpression(compat.ColumnExpression("v") == 1, "one")
            .when(compat.ColumnExpression("v") == 2, "two")
            .otherwise("many")
            .alias("label")
        )
        assert relation.project(expr).fetchall() == [("one",), ("two",), ("many",)]

    def test_star_exclude_and_membership(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT 1 AS a, 2 AS b, 3 AS c")
        assert relation is not None
        assert relation.project(compat.StarExpression(exclude=["b"])).columns == ["a", "c"]
        picky = relation.filter(compat.ColumnExpression("a").isin(1, 5))
        assert picky.fetchall() == [(1, 2, 3)]
        assert relation.filter(compat.ColumnExpression("a").isnotnull()).fetchall() == [(1, 2, 3)]

    def test_sort_takes_expressions(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT unnest([2, 1, 3]) AS v")
        assert relation is not None
        assert relation.sort(compat.ColumnExpression("v").desc()).fetchall() == [(3,), (2,), (1,)]

    def test_function_and_coalesce(self) -> None:
        con = compat.connect()
        relation = con.sql("SELECT NULL::INTEGER AS a, 4 AS b")
        assert relation is not None
        picked = relation.project(
            compat.CoalesceOperator(compat.ColumnExpression("a"), compat.ColumnExpression("b")).alias("c"),
            compat.FunctionExpression("greatest", compat.ColumnExpression("b"), 7).alias("g"),
        )
        assert picked.fetchall() == [(4, 7)]
        assert picked.columns == ["c", "g"]

    def test_project_keeps_an_alias(self) -> None:
        # The rows came out right while the name was lost: the alias rode
        # the expression but never reached the SELECT list.
        con = compat.connect()
        con.run("CREATE TABLE orders AS SELECT 1 AS total")
        relation = con.table("orders").project(compat.ColumnExpression("total").alias("t"))
        assert relation.columns == ["t"]
        assert [column[0] for column in relation.description] == ["t"]
        assert ' AS "t"' in relation.sql_query()
        assert relation.fetchall() == [(1,)]

    def test_aggregate_keeps_an_alias(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT unnest([1, 2, 3]) AS v")
        summed = relation.aggregate([compat.FunctionExpression("sum", compat.ColumnExpression("v")).alias("s")])
        assert summed.columns == ["s"]
        assert [column[0] for column in summed.description] == ["s"]
        assert summed.fetchall() == [(6,)]

    def test_aliased_and_unaliased_columns_mix(self) -> None:
        con = compat.connect()
        relation = rel(con, "SELECT 1 AS a, 2 AS b")
        mixed = relation.project(
            compat.ColumnExpression("a"),
            (compat.ColumnExpression("b") + 1).alias("c"),
            "b",
        )
        assert mixed.columns == ["a", "c", "b"]
        assert mixed.fetchall() == [(1, 3, 2)]

    def test_filter_still_takes_an_aliased_expression(self) -> None:
        # A WHERE cannot carry AS, so the alias is left off there rather
        # than rendered into a syntax error.
        con = compat.connect()
        relation = rel(con, "SELECT unnest([1, 2, 3]) AS v")
        kept = relation.filter((compat.ColumnExpression("v") > 1).alias("big"))
        assert kept.fetchall() == [(2,), (3,)]


class TestCreateFunction:
    def test_explicit_types(self) -> None:
        con = compat.connect()
        assert con.create_function("plus_one", lambda x: x + 1, ["BIGINT"], "BIGINT") is con
        assert con.execute("SELECT plus_one(41)").fetchall() == [(42,)]

    def test_types_come_from_annotations(self) -> None:
        con = compat.connect()

        def shout(text: str, times: int) -> str:
            return text.upper() * times

        con.create_function("shout", shout)
        assert con.execute("SELECT shout('ha', 2)").fetchall() == [("HAHA",)]

    def test_nested_annotations(self) -> None:
        con = compat.connect()

        def total(values: list[int]) -> int:
            return sum(values)

        con.create_function("total", total)
        assert con.execute("SELECT total([1, 2, 3])").fetchall() == [(6,)]

    def test_optional_annotation_unwraps(self) -> None:
        con = compat.connect()

        def maybe(x: int) -> int | None:
            return x if x % 2 else None

        con.create_function("maybe", maybe, null_handling="special")
        assert con.execute("SELECT maybe(2), maybe(3)").fetchall() == [(None, 3)]

    def test_a_string_annotation_is_a_sql_type(self) -> None:
        # Deferred by this module's __future__ import, the annotation arrives
        # as text that evaluates to the string "VARCHAR"; the text is the
        # type, not a Python name to look up.
        con = compat.connect()

        def shout(text: "VARCHAR") -> "VARCHAR":  # type: ignore[name-defined]  # noqa: F821, UP037
            return text.upper()

        con.create_function("shout_sql", shout)
        assert con.execute("SELECT shout_sql('ha')").fetchall() == [("HA",)]

    def test_string_sql_types_mix_with_python_types(self) -> None:
        con = compat.connect()

        def widen(x: int, scale: "DECIMAL(4,1)") -> float:  # type: ignore[valid-type]  # noqa: F821, UP037
            return x * float(scale)

        con.create_function("widen", widen)
        assert con.execute("SELECT widen(2, 1.5)").fetchall() == [(3.0,)]

    def test_undeferred_string_annotations_are_sql_types_too(self) -> None:
        # Without the __future__ import an annotation is the bare string, which
        # inspect's own evaluation turned into a NameError before the text
        # could be read as a type; "INTEGER[]" is not even Python syntax.
        con = compat.connect()

        def total(values: object) -> object:
            return sum(values)  # type: ignore[call-overload]

        total.__annotations__ = {"values": "INTEGER[]", "return": "BIGINT"}
        con.create_function("total_sql", total)
        assert con.execute("SELECT total_sql([1, 2, 3])").fetchall() == [(6,)]

        def label(x: object) -> object:
            return str(x)

        label.__annotations__ = {"x": "int", "return": "VARCHAR"}
        con.create_function("label_sql", label)
        assert con.execute("SELECT label_sql(7)").fetchall() == [("7",)]

    def test_missing_annotations_are_refused_with_directions(self) -> None:
        con = compat.connect()
        with pytest.raises(exceptions.InvalidInputError, match="return_type explicitly"):
            con.create_function("f", lambda x: x)

        def half(x) -> float:  # type: ignore[no-untyped-def]  # noqa: ANN001
            return float(x) / 2

        with pytest.raises(exceptions.InvalidInputError, match="parameters explicitly"):
            con.create_function("half", half)

    def test_exception_handling_return_null(self) -> None:
        con = compat.connect()

        def brittle(x: int) -> int:
            return 100 // x

        con.create_function("brittle", brittle, exception_handling="return_null")
        got = con.execute("SELECT brittle(x) FROM (VALUES (4), (0), (10)) t(x)").fetchall()
        assert got == [(25,), (None,), (10,)]

    def test_null_handling_special(self) -> None:
        con = compat.connect()
        con.create_function("backfill", lambda x: -1 if x is None else x, ["BIGINT"], "BIGINT", null_handling="special")
        assert con.execute("SELECT backfill(NULL::BIGINT)").fetchall() == [(-1,)]

    def test_a_none_return_under_default_nulls_is_refused(self) -> None:
        con = compat.connect()
        con.create_function("swallow", lambda x: None, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match="The UDF is not expected to return NULL values"):
            con.execute("SELECT swallow(1)").fetchall()

    def test_a_none_return_under_special_nulls_is_null(self) -> None:
        con = compat.connect()
        con.create_function("swallow", lambda x: None, ["BIGINT"], "BIGINT", null_handling="special")
        assert con.execute("SELECT swallow(1)").fetchall() == [(None,)]

    def test_a_pandas_na_return_follows_the_same_rule(self) -> None:
        pd = pytest.importorskip("pandas")
        con = compat.connect()
        con.create_function("na_out", lambda x: pd.NA, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match="The UDF is not expected to return NULL values"):
            con.execute("SELECT na_out(1)").fetchall()
        con.create_function("na_ok", lambda x: pd.NA, ["BIGINT"], "BIGINT", null_handling="special")
        assert con.execute("SELECT na_ok(1)").fetchall() == [(None,)]

    def test_error_words_match_the_old_client(self) -> None:
        con = compat.connect()

        def kapow(x: int) -> int:
            message = "kapow"
            raise ValueError(message)

        con.create_function("kapow", kapow)
        with pytest.raises(
            exceptions.InvalidInputError, match="Python exception occurred while executing the UDF"
        ) as info:
            con.execute("SELECT kapow(1)").fetchall()
        assert "ValueError: kapow" in str(info.value)

    def test_arrow_udfs_are_refused(self) -> None:
        con = compat.connect()
        with pytest.raises(exceptions.NotSupportedError, match="Arrow"):
            con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT", type="arrow")

    def test_remove_function_points_at_re_registration(self) -> None:
        con = compat.connect()
        con.create_function("f", lambda x: x, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.NotSupportedError, match="registering the same name again"):
            con.remove_function("f")
        con.create_function("f", lambda x: x * 10, ["BIGINT"], "BIGINT")
        assert con.execute("SELECT f(2)").fetchall() == [(20,)]

    def test_module_level_uses_the_default_connection(self) -> None:
        compat.create_function("mod_double", lambda x: x * 2, ["BIGINT"], "BIGINT")
        try:
            assert compat.execute("SELECT mod_double(21)").fetchall() == [(42,)]
        finally:
            compat.close()

    def test_udf_in_a_relation(self) -> None:
        con = compat.connect()
        con.create_function("bump", lambda x: x + 1, ["BIGINT"], "BIGINT")
        relation = con.sql("SELECT i FROM range(3) t(i)")
        assert relation is not None
        assert relation.project("bump(i)").fetchall() == [(1,), (2,), (3,)]
