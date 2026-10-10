"""Eager polars egress: a plan's rows as a polars DataFrame over the Arrow stream."""

from __future__ import annotations

import datetime
import subprocess
import sys
import textwrap
import threading
from decimal import Decimal

import pytest

from duckdb import exceptions, frame

#: The engine's default Arrow batch size; past it the stream delivers more than one batch.
BATCH = 131072


@pytest.fixture
def con() -> frame.Connection:
    return frame.connect()


@pytest.mark.requires("polars")
class TestToPolars:
    def test_rows_round_trip(self, con: frame.Connection) -> None:
        plan = frame.sql(
            "SELECT i::BIGINT AS n, i / 2 AS f, i::DECIMAL(9, 2) AS d, 'v' || i AS s,"
            " DATE '2026-01-01' + i::INT AS day, [i, i + 1] AS l,"
            " {'a': i, 'b': 'x'} AS st, CASE WHEN i % 2 = 0 THEN i END AS maybe"
            " FROM range(5000) t(i) ORDER BY i"
        )
        out = plan.to_polars(con)
        rows = plan.rows(con)
        assert out.height == 5000
        assert out.columns == ["n", "f", "d", "s", "day", "l", "st", "maybe"]
        assert out["n"].to_list() == [r[0] for r in rows]
        assert out["d"][3] == Decimal("3.00")
        assert out["day"][1] == datetime.date(2026, 1, 2)
        assert out["l"][2].to_list() == [2, 3]
        assert out["st"][2] == {"a": 2, "b": "x"}
        assert out["maybe"][1] is None

    def test_batches_arrive_whole_and_in_order(self, con: frame.Connection) -> None:
        count = 2 * BATCH + 123
        plan = frame.sql(f"SELECT i AS n, CASE WHEN i % 3 = 0 THEN i END AS maybe FROM range({count}) t(i) ORDER BY i")
        out = plan.to_polars(con)
        assert out.height == count
        assert out.n_chunks() > 1
        assert out["n"].to_list() == list(range(count))
        # 131073 is divisible by three and 131074 is not, so the mask holds just past the batch boundary.
        assert out["maybe"][BATCH + 1] == BATCH + 1
        assert out["maybe"][BATCH + 2] is None
        assert out["maybe"].null_count() == count - len(range(0, count, 3))

    def test_an_empty_result_keeps_its_columns(self, con: frame.Connection) -> None:
        out = frame.sql("SELECT 1 AS a, 'x' AS b WHERE false").to_polars(con)
        assert out.height == 0
        assert out.columns == ["a", "b"]

    def test_parameters_flow_through(self, con: frame.Connection) -> None:
        plan = frame.sql("SELECT i AS n FROM range(5) t(i)").filter(frame.col("n") >= frame.param("floor"))
        out = plan.to_polars(con, parameters={"floor": 3})
        assert out["n"].to_list() == [3, 4]

    def test_the_bound_forwarder(self, con: frame.Connection) -> None:
        assert frame.sql("SELECT 42 AS n").on(con).to_polars()["n"].to_list() == [42]

    def test_a_closed_connection_is_refused(self, con: frame.Connection) -> None:
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            frame.sql("SELECT 1").to_polars(con)

    def test_an_engine_error_is_raised_typed(self, con: frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="boom"):
            frame.sql("SELECT CASE WHEN i = 4999 THEN error('boom') ELSE 'ok' END FROM range(5000) t(i)").to_polars(con)
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").to_polars(con)["n"].to_list() == [1]

    def test_an_error_past_the_first_batch_is_raised_typed(self, con: frame.Connection) -> None:
        hit = 2 * BATCH
        late = f"SELECT CASE WHEN i = {hit} THEN error('late')::BIGINT ELSE i END FROM range({hit + 1}) t(i)"
        with pytest.raises(exceptions.InvalidInputError, match="late"):
            frame.sql(late).to_polars(con)
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").to_polars(con)["n"].to_list() == [1]

    def test_an_interrupt_lands_typed(self, con: frame.Connection) -> None:
        timer = threading.Timer(0.2, con.interrupt)
        timer.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                frame.sql("SELECT sum(a.range * b.range) AS s FROM range(200000) a, range(200000) b").to_polars(con)
        finally:
            timer.cancel()
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").to_polars(con)["n"].to_list() == [1]

    @pytest.mark.parametrize("query", ["SELECT INTERVAL 1 DAY AS c", "SELECT union_value(a := 1) AS c"])
    def test_a_type_polars_lacks_is_refused_typed(self, con: frame.Connection, query: str) -> None:
        # polars panics out of Rust on these, a BaseException an `except Exception` would miss.
        with pytest.raises(exceptions.NotSupportedError, match="to_arrow"):
            frame.sql(query).to_polars(con)
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").to_polars(con)["n"].to_list() == [1]

    def test_an_empty_result_with_repeated_names_names_polars_own_cause(self, con: frame.Connection) -> None:
        # polars panics rather than raising DuplicateError when the frame it is given is empty.
        with pytest.raises(exceptions.NotSupportedError, match="multiple fields with name 'a'"):
            frame.sql("SELECT 1 AS a, 2 AS a WHERE false").to_polars(con)
        assert not con._live_streams

    def test_a_consumer_error_leaves_the_connection_usable(self, con: frame.Connection) -> None:
        import polars

        # polars refuses duplicate column names; nothing ended the stream, so its own error surfaces.
        with pytest.raises(polars.exceptions.DuplicateError):
            frame.sql("SELECT 1 AS a, 2 AS a").to_polars(con)
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").to_polars(con)["n"].to_list() == [1]

    def test_a_scalar_function_feeds_the_read(self) -> None:
        # In a child with a timeout, so a GIL deadlock between the read and the function fails, not hangs.
        probe = textwrap.dedent(
            """
            import duckdb

            con = duckdb.frame.connect()
            con.create_function("twice", lambda x: x * 2, ["BIGINT"], "BIGINT")
            out = duckdb.frame.sql("SELECT twice(i) AS n FROM range(200000) t(i)").to_polars(con)
            assert out.height == 200000 and out["n"][3] == 6
            print("fed")
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False)
        assert done.returncode == 0, done.stderr
        assert "fed" in done.stdout


@pytest.mark.requires("polars")
class TestLazy:
    """polars trusts a source with every pushdown it hands over and checks none, so each must hold exactly."""

    @pytest.fixture
    def table(self, con: frame.Connection) -> frame.Frame:
        con.run(
            "CREATE TABLE t AS SELECT i AS a, 's' || i AS b, CASE WHEN i % 3 = 0 THEN NULL ELSE i END AS c"
            " FROM range(10) t(i)"
        )
        return frame.table("t")

    def test_the_schema_and_rows_are_the_eager_frames(self, con: frame.Connection) -> None:
        plan = frame.sql(
            "SELECT i::INT AS n, i::DECIMAL(9, 2) AS d, TIMESTAMPTZ '2026-01-01 00:00:00+00' AS ts,"
            " [i] AS l, {'k': i} AS st FROM range(3) t(i)"
        )
        lazy = plan.to_polars(con, lazy=True)
        eager = plan.to_polars(con)
        assert lazy.collect_schema() == eager.schema
        assert lazy.collect().equals(eager)

    def test_nothing_runs_until_collect_and_each_collect_runs_again(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE grows (i INT)")
        lazy = frame.table("grows").to_polars(con, lazy=True)
        con.run("INSERT INTO grows VALUES (1)")
        assert lazy.collect()["i"].to_list() == [1]
        con.run("INSERT INTO grows VALUES (2)")
        assert lazy.collect()["i"].to_list() == [1, 2]

    def test_a_filter_is_applied(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        lazy = table.to_polars(con, lazy=True)
        assert lazy.filter(pl.col("a") > 7).collect()["a"].to_list() == [8, 9]
        # A NULL comparison drops the row, as SQL's WHERE does.
        assert lazy.filter(pl.col("c") > 7).collect()["a"].to_list() == [8]

    def test_a_row_limit_is_applied(self, con: frame.Connection, table: frame.Frame) -> None:
        assert table.to_polars(con, lazy=True).head(3).collect().height == 3

    def test_a_projection_is_applied(self, con: frame.Connection, table: frame.Frame) -> None:
        assert table.to_polars(con, lazy=True).select("b").collect().columns == ["b"]

    def test_a_filter_on_a_column_left_out_of_the_selection(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        out = table.to_polars(con, lazy=True).filter(pl.col("a") > 7).select("b").collect()
        assert out.to_dicts() == [{"b": "s8"}, {"b": "s9"}]

    def test_a_limit_after_a_filter_counts_the_kept_rows(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        assert table.to_polars(con, lazy=True).filter(pl.col("a") > 2).head(2).collect()["a"].to_list() == [3, 4]

    def test_aggregates_sorts_and_top_k(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        lazy = table.to_polars(con, lazy=True)
        assert lazy.select(pl.len()).collect().item() == 10
        for engine in ("in-memory", "streaming"):
            assert lazy.sort("a", descending=True).head(1).collect(engine=engine)["a"].to_list() == [9]
            assert lazy.filter(pl.col("c").is_not_null()).top_k(2, by="a").collect(engine=engine)["a"].to_list() == [
                8,
                7,
            ]

    def test_the_projection_reaches_the_engine(self, con: frame.Connection) -> None:
        # A column left unselected must never be computed, or its error would surface.
        lazy = frame.sql("SELECT i AS a, error('never computed') AS b FROM range(3) t(i)").to_polars(con, lazy=True)
        assert lazy.select("a").collect()["a"].to_list() == [0, 1, 2]

    def test_the_row_limit_reaches_the_engine(self, con: frame.Connection) -> None:
        # Past the limit the engine would reach the failing row; only a pushed LIMIT stops it first.
        late = f"CASE WHEN i = {4 * BATCH} THEN error('past the limit') ELSE i END"
        sql = f"SELECT {late} AS a FROM range({8 * BATCH}) t(i)"
        assert frame.sql(sql).to_polars(con, lazy=True).head(5).collect()["a"].to_list() == [0, 1, 2, 3, 4]

    def test_parameters_flow_through(self, con: frame.Connection) -> None:
        plan = frame.sql("SELECT i AS n FROM range(5) t(i)").filter(frame.col("n") >= frame.param("floor"))
        assert plan.to_polars(con, parameters={"floor": 3}, lazy=True).collect()["n"].to_list() == [3, 4]

    def test_the_bound_forwarder(self, con: frame.Connection) -> None:
        assert frame.sql("SELECT 42 AS n").on(con).to_polars(lazy=True).collect()["n"].to_list() == [42]

    @pytest.mark.parametrize(
        ("sql", "error"),
        [
            ("SELECT * FROM nowhere", exceptions.CatalogError),
            ("SELECT INTERVAL 1 DAY AS c", exceptions.NotSupportedError),
        ],
    )
    def test_a_plan_polars_cannot_take_fails_at_creation(
        self, con: frame.Connection, sql: str, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            frame.sql(sql).to_polars(con, lazy=True)
        assert not con._live_streams

    @pytest.mark.parametrize("where", ["", " WHERE false"], ids=["rows", "empty"])
    def test_repeated_names_fail_at_creation_as_they_do_eagerly(self, con: frame.Connection, where: str) -> None:
        import polars as pl

        with pytest.raises(pl.exceptions.DuplicateError, match="'a' has more than one occurrence"):
            frame.sql(f"SELECT 1 AS a, 2 AS a{where}").to_polars(con, lazy=True)

    def test_names_differing_only_in_case_project_the_right_column(self, con: frame.Connection) -> None:
        import polars as pl

        # polars holds `a` and `A` apart where DuckDB resolves a name regardless of case, to its first match.
        lazy = frame.sql("SELECT 1 AS a, 3 AS A, 5 AS b").to_polars(con, lazy=True)
        assert lazy.collect_schema().names() == ["a", "A", "b"]
        assert lazy.select("A").collect().item() == 3
        assert lazy.select("a").collect().item() == 1
        assert lazy.select("b", "A").collect().row(0) == (5, 3)
        assert lazy.head(1).collect().row(0) == (1, 3, 5)
        assert lazy.filter(pl.col("A") == 3).select("a").collect().item() == 1

    def test_a_projection_of_no_column_needs_no_query(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        assert table.to_polars(con, lazy=True).select(pl.lit(1)).collect().to_dicts() == [{"literal": 1}]

    @pytest.mark.parametrize(
        "change",
        [
            "CREATE OR REPLACE TABLE w AS SELECT 'y' AS b, 2 AS a",
            "CREATE OR REPLACE TABLE w AS SELECT 2::BIGINT AS a, 'y' AS b",
            "ALTER TABLE w DROP COLUMN b",
        ],
        ids=["reordered", "retyped", "dropped"],
    )
    def test_columns_that_changed_since_creation_are_refused(self, con: frame.Connection, change: str) -> None:
        import polars as pl

        con.run("CREATE TABLE w AS SELECT 1 AS a, 'x' AS b")
        lazy = frame.table("w").to_polars(con, lazy=True)
        con.run(change)
        # Read by the old positions, a reordered table would hand back the other column under the old name.
        for query in (lazy, lazy.select("b")):
            with pytest.raises(pl.exceptions.ComputeError, match="columns changed since the LazyFrame was made"):
                query.collect()
        assert not con._live_streams

    def test_a_pivot_names_its_columns_by_running(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE s AS SELECT * FROM (VALUES ('x', 1), ('y', 2)) t(k, v)")
        lazy = frame.sql("PIVOT s ON k USING sum(v)").to_polars(con, lazy=True)
        assert lazy.collect_schema().names() == ["x", "y"]
        assert lazy.select("y").collect().item() == 2
        import polars as pl

        con.run("INSERT INTO s VALUES ('z', 3)")
        with pytest.raises(pl.exceptions.ComputeError, match="columns changed"):
            lazy.collect()

    def test_an_engine_error_surfaces_from_collect(self, con: frame.Connection) -> None:
        import polars as pl

        lazy = frame.sql("SELECT CASE WHEN i = 3 THEN error('boom') ELSE i END AS a FROM range(5) t(i)").to_polars(
            con, lazy=True
        )
        # polars wraps a source's exception in its own type, keeping the message.
        with pytest.raises(pl.exceptions.ComputeError, match=r"InvalidInputError.*boom"):
            lazy.collect()
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").rows(con) == [(1,)]

    def test_a_closed_connection_surfaces_from_collect(self, con: frame.Connection) -> None:
        import polars as pl

        lazy = frame.sql("SELECT 1 AS n").to_polars(con, lazy=True)
        con.close()
        with pytest.raises(pl.exceptions.ComputeError, match="closed"):
            lazy.collect()

    def test_two_frames_of_one_connection_in_one_query_are_refused_with_the_remedy(
        self, con: frame.Connection, table: frame.Frame
    ) -> None:
        import polars as pl

        lazy = table.to_polars(con, lazy=True)
        # polars overlaps the two scans only when it gets to them together, so the join may also just succeed;
        # it must never succeed wrongly or fail any other way.
        outcome: object
        try:
            outcome = lazy.join(lazy, on="a").sort("a").collect()["a"].to_list()
        except pl.exceptions.ComputeError as error:
            outcome = str(error)
        assert outcome == list(range(10)) or "duplicate()" in str(outcome)
        other = frame.sql("SELECT i AS a, i * 10 AS d FROM range(5) t(i)").to_polars(con.duplicate(), lazy=True)
        assert lazy.join(other, on="a").sort("a").collect()["d"].to_list() == [0, 10, 20, 30, 40]

    def test_a_frame_scanned_inside_its_own_connections_query_errors_rather_than_hangs(self) -> None:
        # In a child with a timeout: re-entering the connection from inside its own query must fail, not wait.
        probe = textwrap.dedent(
            """
            import duckdb
            from duckdb import frame

            con = frame.connect()
            con.register("own", frame.sql("SELECT 1 AS a").to_polars(con, lazy=True))
            print("started", flush=True)
            try:
                frame.sql("SELECT count(*) FROM own").rows(con)
            except duckdb.exceptions.Error as error:
                assert "duplicate()" in str(error), error
                print("refused")
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
        assert done.returncode == 0, done.stderr
        assert "refused" in done.stdout


def test_missing_polars_io_plugins_are_named() -> None:
    probe = textwrap.dedent(
        """
        import sys

        import polars.io

        # Importing polars already set the attribute, which a later import would read instead of the module.
        del polars.io.plugins
        sys.modules["polars.io.plugins"] = None
        import duckdb

        con = duckdb.frame.connect()
        try:
            duckdb.frame.sql("SELECT 1").to_polars(con, lazy=True)
        except ImportError as error:
            assert "IO plugins" in str(error), error
            print("named")
        """
    )
    pytest.importorskip("polars")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "named" in done.stdout


def test_missing_polars_is_named() -> None:
    # Unmarked on purpose: this must hold exactly where polars is not installed.
    probe = textwrap.dedent(
        """
        import sys

        sys.modules["polars"] = None
        import duckdb

        con = duckdb.frame.connect()
        try:
            duckdb.frame.sql("SELECT 1").to_polars(con)
        except ImportError as error:
            assert "polars" in str(error), error
            assert not con._live_streams, "a stream was opened before the import failed"
            print("named")
        """
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "named" in done.stdout
