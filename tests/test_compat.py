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
