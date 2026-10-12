"""Eager polars egress: a plan's rows as a polars DataFrame over the Arrow stream."""

from __future__ import annotations

import contextlib
import datetime
import functools
import gc
import importlib.metadata
import operator
import pathlib
import re
import subprocess
import sys
import textwrap
import threading
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal

import pytest

from duckdb import exceptions, frame

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import polars as pl
    from _pytest.mark import ParameterSet

#: The engine's default Arrow batch size; past it the stream delivers more than one batch.
BATCH = 131072


def _polars_major() -> int:
    import polars as pl

    return int(pl.__version__.split(".")[0])


def _lazy_capable() -> bool:
    """Whether the installed polars, if any, is one to_polars(lazy=True) accepts."""
    try:
        version = importlib.metadata.version("polars")
    except importlib.metadata.PackageNotFoundError:
        return True
    return tuple(int(part) for part in version.split(".")[:2] if part.isdigit()) >= (1, 33)


#: The lazy source refuses older polars, whose expressions also lack what its cases are written in.
lazy_only = pytest.mark.skipif(not _lazy_capable(), reason="to_polars(lazy=True) needs polars 1.33")


def _zoned_literals_push() -> bool:
    """Whether polars hands a source an aware datetime literal it can push, as it does from 1.36."""
    import polars as pl

    return tuple(int(part) for part in pl.__version__.split(".")[:2]) >= (1, 36)


def received(predicate: pl.Expr, schema: dict[str, Any]) -> pl.Expr:
    """The predicate as polars hands it to a source, typed by its optimizer, whatever shape this version gives."""
    import polars as pl
    from polars.io.plugins import register_io_source

    seen: list[pl.Expr] = []

    def source(
        _columns: list[str] | None, given: pl.Expr | None, _rows: int | None, _size: int | None
    ) -> Iterator[pl.DataFrame]:
        assert given is not None
        seen.append(given)
        yield pl.DataFrame(schema=schema)

    register_io_source(source, schema=schema).filter(predicate).collect()
    return seen[0]


def raised_by_collect(error: type[Exception], match: str) -> contextlib.AbstractContextManager[object]:
    """What collect raises for a failed scan: polars 2.0 lets the source's exception through, older ones wrap it."""
    import polars as pl

    if _polars_major() >= 2:
        return pytest.raises(error, match=match)
    return pytest.raises(pl.exceptions.ComputeError, match=rf"{error.__name__}.*{match}")


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

    def test_an_engine_error_polars_panics_on_is_raised_typed(
        self, con: frame.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import polars as pl

        real = pl.DataFrame

        def panicking(data: Any) -> pl.DataFrame:  # noqa: ANN401
            # As polars 1.36 and older do on a stream that ends with an error.
            try:
                return real(data)
            except Exception:
                message = "called `Result::unwrap()` on an `Err` value"
                raise pl.exceptions.PanicException(message) from None

        monkeypatch.setattr(pl, "DataFrame", panicking)
        plan = frame.sql("SELECT CASE WHEN i = 3 THEN error('boom') ELSE i END AS a FROM range(5) t(i)")
        with pytest.raises(exceptions.InvalidInputError, match="boom"):
            plan.to_polars(con)
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
@lazy_only
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
        out = table.to_polars(con, lazy=True).filter(pl.col("c") > 7).select("b", "a").collect()
        assert out.rows() == [("s8", 8)]

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_a_limit_before_a_filter_takes_its_rows_first(
        self, con: frame.Connection, table: frame.Frame, engine: str
    ) -> None:
        import polars as pl

        lazy, eager = table.to_polars(con, lazy=True), table.to_polars(con)
        # One filter DuckDB applies and one polars does; each has a match past the first five rows.
        for predicate in (pl.col("a") > 2, pl.col("b").str.ends_with("3") | pl.col("b").str.ends_with("8")):
            assert lazy.head(5).filter(predicate).collect(engine=engine).equals(eager.head(5).filter(predicate))

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_a_sort_and_limit_keep_the_filter_below_them(self, con: frame.Connection, engine: str) -> None:
        import polars as pl

        # The smallest x fails the filter, so a filter lost under the top-k would show as the wrong row; polars
        # 1.39 to 1.42 join a dynamic hint to the filter it hands the scan.
        plan = frame.sql("SELECT * FROM (VALUES (3, 100, 'c'), (1, 5, 'a'), (2, 50, 'b')) v(x, y, s)")
        lazy = plan.to_polars(con, lazy=True)
        assert lazy.sort("x").head(1).collect(engine=engine).rows() == [(1, 5, "a")]
        assert lazy.filter(pl.col("y") > 15).sort("x").head(1).collect(engine=engine).rows() == [(2, 50, "b")]
        kept = pl.col("s").str.starts_with("b") | pl.col("s").str.starts_with("c")
        assert lazy.filter(kept).sort("x").head(1).collect(engine=engine).rows() == [(2, 50, "b")]

    def test_what_polars_applies_above_the_scan(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        lazy, eager = table.to_polars(con, lazy=True), table.to_polars(con).lazy()
        queries: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
            lambda f: f.with_row_index().filter(pl.col("index") > 6),
            lambda f: f.with_row_index().filter(pl.col("a") > 2).head(2),
            lambda f: f.slice(3, 4),
            lambda f: f.tail(3),
            lambda f: f.filter(pl.col("a") > 2).head(4).head(2),
            lambda f: f.filter(pl.lit(False)),
            lambda f: f.filter(pl.lit(True)),
            lambda f: f.group_by("c").agg(pl.len()).sort("c"),
        ]
        for query in queries:
            assert query(lazy).collect().equals(query(eager).collect())

    def test_an_empty_answer_keeps_the_schema(self, con: frame.Connection, table: frame.Frame) -> None:
        import polars as pl

        lazy = table.to_polars(con, lazy=True)
        for predicate in (pl.col("a") > 100, pl.col("b").str.starts_with("none")):
            out = lazy.filter(predicate).collect()
            assert out.height == 0
            assert out.schema == lazy.collect_schema()
        empty = frame.sql("SELECT 1::INT AS a, 'x' AS b WHERE false").to_polars(con, lazy=True)
        assert empty.collect().schema == empty.collect_schema() == pl.Schema({"a": pl.Int32, "b": pl.String})

    @pytest.mark.parametrize("name", ["x:y", "x y", "x.y", '"xy"', "a\" , error('x') AS \"b", "日本", "_c2"])
    def test_a_name_that_needs_quoting_projects_and_filters(self, con: frame.Connection, name: str) -> None:
        import polars as pl

        # `_c2` is also the alias a filter reads the second column by.
        con.register("named", pl.DataFrame({"other": [10, 20, 30], name: [1, 2, 3]}))
        lazy = frame.table("named").to_polars(con, lazy=True)
        assert lazy.collect_schema().names() == ["other", name]
        assert lazy.select(name).collect()[name].to_list() == [1, 2, 3]
        assert lazy.filter(pl.col(name) == 2).collect().rows() == [(20, 2)]
        assert lazy.filter(pl.col(name).cast(pl.Float64) > 1.5).select(name).collect()[name].to_list() == [2, 3]

    def test_a_scan_past_one_batch_reads_whole(self, con: frame.Connection) -> None:
        import polars as pl

        count = 2 * BATCH + 123
        con.run(f"CREATE TABLE big AS SELECT i AS a, i % 7 AS b, 's' || i AS s FROM range({count}) t(i)")
        lazy, eager = frame.table("big").to_polars(con, lazy=True), frame.table("big").to_polars(con)
        queries: list[Callable[[pl.LazyFrame], pl.LazyFrame]] = [
            lambda f: f,
            lambda f: f.filter(pl.col("a") >= BATCH).select("a"),
            lambda f: f.filter(pl.col("s").str.ends_with("3")),
            lambda f: f.group_by("b").agg(pl.len(), pl.col("a").max()).sort("b"),
            lambda f: f.select(pl.col("b").unique().sort()),
            lambda f: f.select(pl.len()),
        ]
        for engine in ("in-memory", "streaming"):
            for query in queries:
                assert query(lazy).collect(engine=engine).equals(query(eager.lazy()).collect())

    def test_a_frame_keeps_its_connection_open(self, con: frame.Connection) -> None:
        def made() -> pl.LazyFrame:
            local = frame.connect()
            local.run("CREATE TABLE z AS SELECT 1 AS foo, 2 AS bar")
            lazy: pl.LazyFrame = frame.table("z").to_polars(local, lazy=True)
            return lazy

        def duplicated() -> pl.LazyFrame:
            lazy: pl.LazyFrame = frame.sql("SELECT 1 AS foo").to_polars(con.duplicate(), lazy=True)
            return lazy

        lazy, other = made(), duplicated()
        gc.collect()
        assert lazy.collect().to_dicts() == [{"foo": 1, "bar": 2}]
        assert other.collect().to_dicts() == [{"foo": 1}]

    def test_explain_and_the_schema_run_nothing(self, con: frame.Connection) -> None:
        import polars as pl

        lazy = frame.sql("SELECT CASE WHEN i >= 0 THEN error('ran') END AS a FROM range(3) t(i)").to_polars(
            con, lazy=True
        )
        assert lazy.collect_schema().names() == ["a"]
        assert "PYTHON SCAN" in lazy.filter(pl.col("a") == 1).explain()
        with raised_by_collect(exceptions.InvalidInputError, "ran"):
            lazy.collect()

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
        lazy = frame.sql(sql).to_polars(con, lazy=True)
        assert lazy.head(5).collect()["a"].to_list() == [0, 1, 2, 3, 4]
        # It reaches the engine with a filter polars applies after it too.
        import polars as pl

        assert lazy.head(5).filter(pl.col("a").cast(pl.String).str.ends_with("3")).collect()["a"].to_list() == [3]

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
        con.run("CREATE TABLE w AS SELECT 1 AS a, 'x' AS b")
        lazy = frame.table("w").to_polars(con, lazy=True)
        con.run(change)
        # Read by the old positions, a reordered table would hand back the other column under the old name.
        for query in (lazy, lazy.select("b")):
            with raised_by_collect(exceptions.InvalidInputError, "columns changed since the LazyFrame was made"):
                query.collect()
        assert not con._live_streams

    def test_a_dtype_a_session_setting_changed_is_refused(self, con: frame.Connection) -> None:
        import polars as pl

        # The engine's type text stays TIMESTAMP WITH TIME ZONE while polars' dtype follows the session zone.
        con.run("SET TimeZone = 'UTC'")
        lazy = frame.sql("SELECT TIMESTAMPTZ '2026-11-01 05:30:00+00' AS t, 1 AS i").to_polars(con, lazy=True)
        con.run("SET TimeZone = 'America/New_York'")
        for query in (lazy, lazy.select("t"), lazy.filter(pl.col("i") > 0)):
            with raised_by_collect(exceptions.InvalidInputError, "columns changed since the LazyFrame was made"):
                query.collect()
        assert lazy.select("i").collect()["i"].to_list() == [1]

    @pytest.mark.parametrize(
        "text", ["SELECT 1 AS i;", "SELECT 1 AS i ;;\n", "SELECT 1 AS i -- note", "SELECT 1 AS i /* note */;"]
    )
    def test_sql_closing_with_a_terminator_or_a_comment(
        self, con: frame.Connection, text: str, tmp_path: pathlib.Path
    ) -> None:
        import polars as pl

        plan = frame.sql(text)
        assert plan.to_polars(con, lazy=True).filter(pl.col("i") > 0).head(1).collect().rows() == [(1,)]
        assert plan.select(frame.col("i")).rows(con) == [(1,)]
        plan.to_csv(con, tmp_path / "out.csv")
        assert (tmp_path / "out.csv").read_text().split() == ["i", "1"]

    def test_a_pivot_names_its_columns_by_running(self, con: frame.Connection) -> None:
        con.run("CREATE TABLE s AS SELECT * FROM (VALUES ('x', 1), ('y', 2)) t(k, v)")
        lazy = frame.sql("PIVOT s ON k USING sum(v)").to_polars(con, lazy=True)
        assert lazy.collect_schema().names() == ["x", "y"]
        assert lazy.select("y").collect().item() == 2
        con.run("INSERT INTO s VALUES ('z', 3)")
        with raised_by_collect(exceptions.InvalidInputError, "columns changed"):
            lazy.collect()

    def test_an_engine_error_surfaces_from_collect(self, con: frame.Connection) -> None:
        lazy = frame.sql("SELECT CASE WHEN i = 3 THEN error('boom') ELSE i END AS a FROM range(5) t(i)").to_polars(
            con, lazy=True
        )
        with raised_by_collect(exceptions.InvalidInputError, "boom"):
            lazy.collect()
        assert not con._live_streams
        assert frame.sql("SELECT 1 AS n").rows(con) == [(1,)]

    def test_a_closed_connection_surfaces_from_collect(self, con: frame.Connection) -> None:
        lazy = frame.sql("SELECT 1 AS n").to_polars(con, lazy=True)
        con.close()
        with raised_by_collect(exceptions.InterfaceError, "closed"):
            lazy.collect()

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_a_frame_used_twice_in_one_query_is_read_once(
        self, con: frame.Connection, table: frame.Frame, monkeypatch: pytest.MonkeyPatch, engine: str
    ) -> None:
        import polars as pl

        from duckdb.frame import _arrow

        reads: list[str] = []
        real = _arrow.polars_frame

        def spy(connection: frame.Connection, sql: str, values: Sequence[object] | None) -> object:
            reads.append(sql)
            return real(connection, sql, values)

        lazy, eager = table.to_polars(con, lazy=True), table.to_polars(con)
        monkeypatch.setattr(_arrow, "polars_frame", spy)
        cached = lazy.cache()
        assert cached.join(cached, on="a").collect(engine=engine).height == 10
        assert len(reads) == 1
        if engine == "streaming" and _polars_major() >= 2:
            pytest.skip("polars 2.0's streaming engine reads a plain scan once per use; cache() reads it once")
        reads.clear()
        assert lazy.join(lazy, on="a").sort("a").collect(engine=engine)["a"].to_list() == list(range(10))
        assert pl.concat([lazy, lazy]).collect(engine=engine).height == 20
        both = lazy.select("a", "b").join(lazy.select("a", "c"), on="a").sort("a").collect(engine=engine)
        assert both.equals(eager.select("a", "b", "c"))
        assert len(reads) == 3
        # Read once, a plan that differs per run gives one answer for both uses.
        noise = frame.sql("SELECT random() AS r FROM range(3) t(i)").to_polars(con, lazy=True)
        assert noise.join(noise, on="r").collect(engine=engine).height == 3

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_frames_each_used_twice_on_connections_of_their_own(
        self, engine: Literal["in-memory", "streaming"]
    ) -> None:
        def build(left: pl.LazyFrame, right: pl.LazyFrame) -> pl.LazyFrame:
            joined = left.join(right, on=["g", "t"])
            return joined.join(joined.select("g").unique(), on="g").sort("g", "t")

        values = "SELECT i % 7 AS g, i AS t, i * 0.5 AS x FROM range(1000) t(i)"
        keys = "SELECT i % 7 AS g, i AS t FROM range(1000) t(i)"
        expect = build(
            frame.sql(values).to_polars(frame.connect()).lazy(), frame.sql(keys).to_polars(frame.connect()).lazy()
        )
        left = frame.sql(values).to_polars(frame.connect(), lazy=True)
        right = frame.sql(keys).to_polars(frame.connect(), lazy=True)
        assert build(left, right).collect(engine=engine).equals(expect.collect())

    def test_scans_of_one_connection_that_overlap_are_refused_with_the_remedy(
        self, con: frame.Connection, table: frame.Frame
    ) -> None:
        import polars as pl

        lazy = table.to_polars(con, lazy=True)
        # Differently filtered uses of one frame, or two frames of one connection, are separate scans; polars
        # overlaps them only when it gets to them together, so the query may also just succeed, but must never
        # succeed wrongly or fail any other way.
        for left, right, answer in (
            (lazy.filter(pl.col("a") > 2), lazy.filter(pl.col("a") < 8), list(range(3, 8))),
            (lazy, table.to_polars(con, lazy=True), list(range(10))),
        ):
            outcome: object
            try:
                outcome = left.join(right, on="a").sort("a").collect()["a"].to_list()
            except (pl.exceptions.ComputeError, exceptions.OperationalError) as error:
                outcome = str(error)
            assert outcome == answer or "duplicate()" in str(outcome)
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


#: Adversarial rows: NULLs, NaN, signed zero, infinities, integer extremes, case and accent variants, epoch and
#: range-end dates, and instants written in other zones.
PUSHDOWN_TABLE = """
CREATE TABLE p AS SELECT * FROM (VALUES
 (1::INT, 1::TINYINT, 18446744073709551615::UBIGINT, 1.5::DOUBLE, 0.1::FLOAT, 'abc', true, DATE '2026-01-02',
  TIMESTAMP '2026-01-02 03:04:05.000006', TIMESTAMPTZ '2026-01-02 03:04:05+00',
  '00000000-0000-0000-0000-000000000001'::UUID),
 (NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
 (-2147483648, -128, 0, 'nan'::DOUBLE, 'nan'::FLOAT, 'ABC', false, DATE '1970-01-01', TIMESTAMP '1970-01-01 00:00:00',
  TIMESTAMPTZ '1970-01-01 00:00:00+00', NULL),
 (2147483647, 127, 9223372036854775808, '-0.0'::DOUBLE, 'inf'::FLOAT, 'é', NULL, DATE '9999-12-31',
  TIMESTAMP '2262-01-01 00:00:00', TIMESTAMPTZ '2262-01-01 00:00:00+00', NULL),
 (0, 0, 1, 'inf'::DOUBLE, '-inf'::FLOAT, '', true, DATE '0001-01-01', TIMESTAMP '1677-09-22 00:00:00',
  TIMESTAMPTZ '2026-01-02 03:04:05.5+00', NULL),
 (5, 5, 5, 0.0::DOUBLE, 0.0::FLOAT, 'abd', false, DATE '2026-01-03', TIMESTAMP '2026-01-02 03:04:05',
  TIMESTAMPTZ '2026-01-01 23:00:00-04', NULL)
) v(i, ti, u, f, f4, s, b, d, ts, tz, id)
"""


def _pushdown_cases() -> list[ParameterSet]:
    if not _lazy_capable():
        return []
    with contextlib.suppress(ModuleNotFoundError):
        import polars as pl

        c = pl.col
        utc = datetime.UTC
        plus_one = datetime.timezone(datetime.timedelta(hours=1))
        pushed = [
            ("int compare", c("i") > 1),
            ("int extreme", c("i") >= -2147483648),
            ("int not equal", c("i") != 5),
            ("not", ~(c("i") > 1)),
            ("or", (c("i") > 1) | (c("i") < 0)),
            ("is null", c("i").is_null()),
            ("is not null on any type", c("id").is_not_null()),
            ("between", c("i").is_between(0, 5)),
            ("between open", c("i").is_between(0, 5, closed="none")),
            ("tinyint", c("ti") > 100),
            ("unsigned past int64", c("u") > 2**63),
            ("unsigned max", c("u") == 18446744073709551615),
            ("double", c("f") > 1.0),
            ("double zero meets minus zero", c("f") == 0.0),
            ("float", c("f4") > 0.1),
            ("float equal", c("f4") == 0.1),
            ("string equal", c("s") == "abc"),
            ("string order", c("s") < "abd"),
            ("string case", c("s") > "ABC"),
            ("string empty", c("s") == ""),
            ("string not equal", c("s") != "abc"),
            ("string is_in", c("s").is_in(["abc", "é"])),
            ("boolean column", c("b")),
            ("boolean negated", ~c("b")),
            ("date", c("d") < datetime.date(2026, 1, 3)),
            ("date range end", c("d") >= datetime.date(1, 1, 1)),
            ("timestamp", c("ts") > datetime.datetime(2026, 1, 2, 3, 4, 5)),
            ("timestamp microseconds", c("ts") == datetime.datetime(2026, 1, 2, 3, 4, 5, 6)),
            ("conjunction", (c("i") > 0) & (c("b") == True)),  # noqa: E712
            ("or of conjunctions", ((c("i") > 0) & (c("s") == "abc")) | c("b").is_null()),
        ]
        kept = [
            ("cast by polars", c("i") > 1.5),
            ("cast of unsigned by polars", c("u") > -1),
            ("is_in with a null", c("i").is_in([0, None])),
            ("is_in matching nulls", c("s").is_in(["abc", None], nulls_equal=True)),
            ("not is_in matching nulls", ~c("s").is_in(["abc", None], nulls_equal=True)),
            ("is_in nothing", c("s").is_in([])),
            ("not is_in nothing", ~c("s").is_in([])),
            ("typed null literal", c("i") == pl.lit(None, dtype=pl.Int32)),
            ("or with a side kept", (c("i") > 1) | c("s").str.starts_with("a")),
            ("nan literal", c("f") > float("nan")),
            ("infinite literal", c("f") < float("inf")),
            ("is_nan", c("f").is_nan()),
            ("string function", c("s").str.starts_with("ab")),
            ("column to column", c("i") > c("ti")),
        ]
        # polars before 2.0 casts the list to Int64, and the cast stays with polars.
        uncast = pytest.param(c("i").is_in([0, 5, 7]), _polars_major() >= 2, id="int is_in")
        zoned = [
            ("instant", c("tz") == datetime.datetime(2026, 1, 2, 3, 4, 5, tzinfo=utc)),
            ("instant in another zone", c("tz") > datetime.datetime(2026, 1, 2, 4, 4, 5, tzinfo=plus_one)),
        ]
        return [
            *(pytest.param(p, True, id=name) for name, p in pushed),
            *(pytest.param(p, False, id=name) for name, p in kept),
            uncast,
            *(pytest.param(p, _zoned_literals_push(), id=name) for name, p in zoned),
        ]
    return []


def _instants(*hours: int) -> tuple[datetime.datetime, ...]:
    return tuple(datetime.datetime(2026, 1, 1, hour, tzinfo=datetime.UTC) for hour in hours)


#: Per engine type, three ascending values and whether a comparison with one reaches DuckDB.
TYPED_COLUMNS = [
    *((t, (1, 10, 100), True) for t in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "UTINYINT", "USMALLINT")),
    *((t, (1, 10, 100), True) for t in ("UINTEGER", "UBIGINT", "FLOAT", "DOUBLE")),
    *((t, (1, 10, 100), False) for t in ("HUGEINT", "UHUGEINT", "DECIMAL(4,1)", "DECIMAL(9,1)", "DECIMAL(18,4)")),
    *((t, (1, 10, 100), False) for t in ("DECIMAL(20,0)", "DECIMAL(30,12)")),
    ("VARCHAR", ("a", "b", "c"), True),
    ("DATE", (datetime.date(2000, 1, 1), datetime.date(2000, 10, 1), datetime.date(2010, 1, 1)), True),
    ("TIMESTAMP", tuple(datetime.datetime(2008 + year, 1, 1, 0, 0, 1) for year in (0, 2, 12)), True),
    ("TIMESTAMPTZ", _instants(1, 2, 3), None),
    ("TIME", (datetime.time(0, 1), datetime.time(0, 10), datetime.time(1, 0)), False),
    *(
        (t, tuple(datetime.datetime(2008 + y, 1, 1) for y in (0, 2, 12)), False)
        for t in ("TIMESTAMP_S", "TIMESTAMP_MS")
    ),
    ("TIMESTAMP_NS", tuple(datetime.datetime(2008 + year, 1, 1) for year in (0, 2, 12)), False),
    ("BLOB", (b"\x01", b"\x02", b"\x03"), False),
]

#: String values that are SQL syntax, quoting or escapes when written into a query.
AWKWARD_TEXT = [
    "A'dam",
    'answer = "42"',
    "'; DROP TABLE users; --",
    "line1\nline2\ttab",
    "",
    "\\",
    "\\x41",
    "日本",
    "é",
]


@pytest.mark.requires("polars")
@lazy_only
class TestLazyPushdown:
    """A pushed filter must mean exactly what polars' own does, on every row: each case is checked against polars."""

    @pytest.fixture(autouse=True)
    def utc(self, con: frame.Connection) -> None:
        # polars types a datetime aware of UTC or a fixed offset as zone UTC and compares it only with a column of
        # that same zone name, which the session zone sets; an Etc/UTC session would make polars refuse it.
        con.run("SET TimeZone = 'UTC'")

    @pytest.fixture
    def queries(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        from duckdb.frame import _arrow

        seen: list[str] = []
        real = _arrow.polars_frame

        def spy(connection: frame.Connection, sql: str, values: Sequence[object] | None) -> object:
            seen.append(sql)
            return real(connection, sql, values)

        monkeypatch.setattr(_arrow, "polars_frame", spy)
        return seen

    @pytest.fixture
    def table(self, con: frame.Connection) -> frame.Frame:
        con.run(PUSHDOWN_TABLE)
        return frame.table("p")

    @pytest.mark.parametrize(("predicate", "pushed"), _pushdown_cases())
    def test_the_answer_is_polars_own(
        self, con: frame.Connection, table: frame.Frame, queries: list[str], predicate: pl.Expr, *, pushed: bool
    ) -> None:
        expect = table.to_polars(con).filter(predicate)
        lazy = table.to_polars(con, lazy=True)
        queries.clear()
        assert lazy.filter(predicate).collect().equals(expect)
        assert any(" WHERE " in query for query in queries) is pushed

    @pytest.mark.parametrize("collation", ["nocase", "noaccent"])
    def test_strings_compare_byte_wise_under_any_session_collation(
        self, con: frame.Connection, table: frame.Frame, collation: str
    ) -> None:
        import polars as pl

        con.run(f"SET default_collation = '{collation}'")
        lazy = table.to_polars(con, lazy=True)
        eager = table.to_polars(con)
        for predicate in (pl.col("s") == "abc", pl.col("s") < "b", pl.col("s").is_in(["ABC", "e"])):
            assert lazy.filter(predicate).collect().equals(eager.filter(predicate))

    def test_instants_compare_as_instants_in_any_session_zone(self, con: frame.Connection, queries: list[str]) -> None:
        import zoneinfo

        import polars as pl

        con.run("SET TimeZone = 'America/New_York'")
        # Quarter hours across the autumn change, where 01:30 local happens twice.
        start = "TIMESTAMPTZ '2026-11-01 04:00:00+00'"
        con.run(f"CREATE TABLE z AS SELECT {start} + i * INTERVAL 15 MINUTE AS tz FROM range(12) t(i)")
        lazy, eager = frame.table("z").to_polars(con, lazy=True), frame.table("z").to_polars(con)
        assert eager.schema["tz"] == pl.Datetime("us", "America/New_York")
        zone = zoneinfo.ZoneInfo("America/New_York")
        first, second = (datetime.datetime(2026, 11, 1, 1, 30, fold=fold, tzinfo=zone) for fold in (0, 1))
        for predicate in (
            pl.col("tz") > first,
            pl.col("tz") > second,
            pl.col("tz") == second,
            pl.col("tz").is_between(first, second),
        ):
            queries.clear()
            assert lazy.filter(predicate).collect().equals(eager.filter(predicate)), predicate
            assert any(" WHERE " in query for query in queries) is _zoned_literals_push(), predicate

    def test_strings_compare_byte_wise_on_a_collated_column(self, con: frame.Connection) -> None:
        import polars as pl

        # The column's collation is invisible in the result header, so the pushed filter must override it.
        con.run("CREATE TABLE nc (s VARCHAR COLLATE NOCASE)")
        con.run("INSERT INTO nc VALUES ('ABC'), ('abc'), ('b'), ('B')")
        lazy = frame.table("nc").to_polars(con, lazy=True)
        assert lazy.filter(pl.col("s") == "abc").collect()["s"].to_list() == ["abc"]
        assert lazy.filter(pl.col("s") < "b").collect()["s"].to_list() == ["ABC", "abc", "B"]

    def test_names_differing_only_in_case_filter_the_right_column(self, con: frame.Connection) -> None:
        import polars as pl

        lazy = frame.sql("SELECT 1 AS a, 3 AS A, 5 AS b").to_polars(con, lazy=True)
        assert lazy.filter(pl.col("A") == 3).select("b").collect().item() == 5
        assert lazy.filter(pl.col("a") == 3).collect().height == 0

    def test_a_pushed_filter_spares_the_engine_the_rows_it_drops(self, con: frame.Connection) -> None:
        import polars as pl

        failing = "CASE WHEN i = 3 THEN error('dropped row') ELSE i END"
        lazy = frame.sql(f"SELECT i, {failing} AS v FROM range(5) t(i)").to_polars(con, lazy=True)
        assert lazy.filter(pl.col("i") != 3).collect()["v"].to_list() == [0, 1, 2, 4]
        # A filter kept for polars runs after the engine, which computes every row.
        with raised_by_collect(exceptions.InvalidInputError, "dropped row"):
            lazy.filter(pl.col("i").cast(pl.Float64) != 3.0).collect()

    def test_a_partly_pushed_filter_applies_the_rest_in_polars(
        self, con: frame.Connection, table: frame.Frame, queries: list[str]
    ) -> None:
        import polars as pl

        predicate = (pl.col("i") > 0) & pl.col("s").str.starts_with("ab")
        lazy = table.to_polars(con, lazy=True)
        queries.clear()
        assert lazy.filter(predicate).collect().equals(table.to_polars(con).filter(predicate))
        assert any(" WHERE " in query for query in queries)

    @pytest.mark.parametrize(("type_text", "values", "pushed"), TYPED_COLUMNS, ids=[t for t, _, _ in TYPED_COLUMNS])
    def test_each_type_answers_as_polars_does(
        self,
        con: frame.Connection,
        queries: list[str],
        type_text: str,
        values: tuple[Any, Any, Any],
        *,
        pushed: bool | None,
    ) -> None:
        import polars as pl

        c = pl.col
        low, mid, high = values
        con.run(f"CREATE TABLE x (a {type_text}, b {type_text}, c {type_text})")
        con.run("INSERT INTO x VALUES ($1, $1, $1), ($2, $2, $2), ($3, $2, $3), (NULL, NULL, NULL)", [low, mid, high])
        lazy, eager = frame.table("x").to_polars(con, lazy=True), frame.table("x").to_polars(con)
        comparisons = [
            c("a") == low,
            c("a") != low,
            c("a") > low,
            c("a") >= mid,
            c("a") < mid,
            c("a") <= mid,
            (c("a") == mid) & (c("b") == low),
            (c("a") == high) & (c("b") == mid) & (c("c") == high),
            (c("a") == high) | (c("b") == low),
        ]
        # A NULL check means the same on a column of any type; None marks the zoned literals pushed from 1.36.
        reaches_engine = _zoned_literals_push() if pushed is None else pushed
        cases = [*((p, reaches_engine) for p in comparisons), (c("a").is_null(), True), (c("a").is_not_null(), True)]
        for predicate, reaches in cases:
            queries.clear()
            assert lazy.filter(predicate).collect().equals(eager.filter(predicate)), predicate
            assert any(" WHERE " in query for query in queries) is reaches, predicate

    def test_boolean_columns_combine(self, con: frame.Connection, queries: list[str]) -> None:
        import polars as pl

        c = pl.col
        con.run(
            "CREATE TABLE tb AS SELECT * FROM (VALUES (true, true), (true, false), (false, true), (NULL, NULL)) v(a, b)"
        )
        lazy, eager = frame.table("tb").to_polars(con, lazy=True), frame.table("tb").to_polars(con)
        for predicate in (c("a") & c("b"), c("a") | c("b"), ~c("a") | c("b").is_null(), c("a") != False):  # noqa: E712
            queries.clear()
            assert lazy.filter(predicate).collect().equals(eager.filter(predicate)), predicate
            assert any(" WHERE " in query for query in queries), predicate

    def test_string_values_are_never_read_as_syntax(self, con: frame.Connection, queries: list[str]) -> None:
        import polars as pl

        con.run("CREATE TABLE s (s VARCHAR)")
        for text in [*AWKWARD_TEXT, None]:
            con.run("INSERT INTO s VALUES ($1)", [text])
        lazy, eager = frame.table("s").to_polars(con, lazy=True), frame.table("s").to_polars(con)
        predicates = [pl.col("s") == text for text in AWKWARD_TEXT] + [pl.col("s") > text for text in AWKWARD_TEXT]
        for predicate in [*predicates, pl.col("s").is_in(AWKWARD_TEXT)]:
            queries.clear()
            assert lazy.filter(predicate).collect().equals(eager.filter(predicate)), predicate
            assert any(" WHERE " in query for query in queries), predicate

    def test_a_string_value_holding_a_nul_byte(self, con: frame.Connection) -> None:
        import polars as pl

        con.run("CREATE TABLE z (s VARCHAR)")
        con.run("INSERT INTO z VALUES ($1), ('nul')", ["nul\x00byte"])
        lazy = frame.table("z").to_polars(con, lazy=True)
        assert lazy.filter(pl.col("s") == "nul\x00byte").collect()["s"].to_list() == ["nul\x00byte"]
        assert lazy.filter(pl.col("s").is_in(["nul\x00byte", "x"])).collect()["s"].to_list() == ["nul\x00byte"]

    @pytest.mark.parametrize(
        ("combine", "depth", "pushed"),
        [
            (operator.and_, 150, True),
            (operator.or_, 150, True),
            (operator.and_, 1000, False),
            (operator.or_, 1000, False),
        ],
        ids=["and", "or", "deep and", "deep or"],
    )
    def test_a_deeply_nested_predicate_still_answers(
        self,
        con: frame.Connection,
        queries: list[str],
        combine: Callable[[pl.Expr, pl.Expr], pl.Expr],
        depth: int,
        *,
        pushed: bool,
    ) -> None:
        import polars as pl

        # A thousand levels exhaust a recursive walk yet still serialize.
        predicate = functools.reduce(combine, [pl.col("i") != n for n in range(depth)])
        con.run("CREATE TABLE deep AS SELECT i FROM range(1005) t(i)")
        lazy, eager = frame.table("deep").to_polars(con, lazy=True), frame.table("deep").to_polars(con)
        queries.clear()
        assert lazy.filter(predicate).collect().equals(eager.filter(predicate))
        assert any(" WHERE " in query for query in queries) is pushed

    @pytest.mark.parametrize("engine", ["in-memory", "streaming"])
    def test_a_conjunct_left_to_polars_binds_nothing(self, con: frame.Connection, engine: str) -> None:
        import polars as pl

        con.run("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'x'), (2, 'abc'), (3, 'z'), (4, NULL)) v(i, s)")
        # Each translates a value before meeting the part polars keeps, so the value must not stay bound.
        predicates = [
            (pl.col("s") == "x") | pl.col("s").str.starts_with("a"),
            pl.col("s").is_in(["x", None]),
            ((pl.col("s") == "x") | pl.col("s").str.starts_with("a")) & (pl.col("i") > 0),
        ]
        for plan, parameters in (
            (frame.table("t"), None),
            (frame.table("t").filter(frame.col("i") >= frame.param("floor")), {"floor": 1}),
        ):
            lazy = plan.to_polars(con, parameters=parameters, lazy=True)
            eager = plan.to_polars(con, parameters=parameters)
            for predicate in predicates:
                assert lazy.filter(predicate).collect(engine=engine).equals(eager.filter(predicate)), predicate

    @pytest.mark.parametrize(
        ("column", "literal"),
        [
            ("d", ("lit", 2932897, "Date")),
            ("d", ("lit", -719163, "Date")),
            ("ts", ("lit", 253402300800000000, "Datetime")),
            ("d", ("list", 2932897, "Date")),
            ("ts", ("list", 253402300800000000, "Datetime")),
        ],
        ids=["date past 9999", "date before 1", "timestamp past 9999", "date list past 9999", "timestamp list"],
    )
    def test_a_literal_past_pythons_dates_stays_with_polars(
        self, con: frame.Connection, queries: list[str], column: str, literal: tuple[str, int, str]
    ) -> None:
        import polars as pl

        kind, raw, dtype_name = literal
        dtype = pl.Date if dtype_name == "Date" else pl.Datetime("us")
        value = pl.lit(raw, dtype=dtype) if kind == "lit" else pl.lit([raw], dtype=pl.List(dtype))
        plan = frame.sql(
            "SELECT * FROM (VALUES (1, DATE '2000-01-01', TIMESTAMP '2000-01-01'), (2, NULL, NULL)) t(i, d, ts)"
        )
        lazy, eager = plan.to_polars(con, lazy=True), plan.to_polars(con)
        predicate = pl.col(column).is_in(value) if kind == "list" else pl.col(column) < value
        queries.clear()
        assert lazy.filter(predicate).collect().equals(eager.filter(predicate))
        assert not any(" WHERE " in query for query in queries)

    def test_parameters_and_pushdown_together(self, con: frame.Connection) -> None:
        import polars as pl

        plan = frame.sql("SELECT i AS n FROM range(10) t(i)").filter(frame.col("n") >= frame.param("floor"))
        lazy = plan.to_polars(con, parameters={"floor": 3}, lazy=True)
        assert lazy.filter(pl.col("n") < 6).collect()["n"].to_list() == [3, 4, 5]
        # The plan binds a string of its own, so the pushed ones are numbered after both of its values.
        named = frame.sql("SELECT i AS n, 's' || i AS s FROM range(10) t(i)").filter(
            (frame.col("n") >= frame.param("floor")) & (frame.col("s") != "s9")
        )
        lazy = named.to_polars(con, parameters={"floor": 3}, lazy=True)
        assert lazy.filter(pl.col("s").is_in(["s2", "s4", "s9"]) | (pl.col("s") == "s5")).collect().rows() == [
            (4, "s4"),
            (5, "s5"),
        ]

    def test_a_dynamic_hint_is_dropped_and_the_rest_kept_as_polars_expressions(self) -> None:
        import json

        import polars as pl

        from duckdb.frame import _arrow

        # Only polars 1.39 to 1.42 hand a source dynamic predicates, so one is joined to a real predicate here,
        # standing in for polars' own expression: it serializes and pops as polars' does.
        class Hint:
            def __init__(self) -> None:
                self.meta = self

            def serialize(self, format: str) -> str:
                return json.dumps({"Display": {"inputs": [], "fmt_str": "dynamic_pred: 1"}})

        class WithHint:
            def __init__(self, real: pl.Expr) -> None:
                self.meta = self
                self.real = real

            def serialize(self, format: str) -> str:
                real = json.loads(self.real.meta.serialize(format="json"))
                return json.dumps(
                    {"BinaryExpr": {"left": real, "op": "And", "right": json.loads(Hint().serialize(format))}}
                )

            def pop(self) -> list[object]:
                return [Hint(), self.real]

        columns = {"i": (1, pl.Int64, True), "f": (2, pl.Float64, True)}
        typed = received(pl.col("i") > 1, {"i": pl.Int64, "f": pl.Float64})
        assert _arrow._split(pl, WithHint(typed), columns, None) == (['"_c1" > 1'], {1}, [], None)
        # A NaN literal cannot survive a round trip through its serialized form, so the kept part must be the
        # expression itself.
        nan = pl.col("f") > float("nan")
        pushed, _, _, residual = _arrow._split(pl, WithHint(nan), columns, None)
        assert pushed == []
        frame_ = pl.DataFrame({"i": [1, 2], "f": [1.0, float("nan")]})
        assert frame_.filter(residual).equals(frame_.filter(nan))

    @pytest.mark.parametrize(
        "tree",
        [
            {"BinaryExpr": {"left": {"Column": "i"}, "op": ["Gt"], "right": {"Column": "i"}}},
            {"BinaryExpr": {"left": {"Column": "i"}, "op": "Gt"}},
            {"BinaryExpr": {"left": {"Column": "nowhere"}, "op": "Gt", "right": {"Column": "i"}}},
            {"Function": {"input": [{"Column": "i"}] * 3, "function": {"Boolean": {"IsBetween": "Both"}}}},
            {"Function": {"input": [{"Column": "i"}] * 3, "function": {"Boolean": {"IsBetween": {"closed": []}}}}},
            {"Function": {"input": [], "function": {"Boolean": "Not"}}},
            {"Function": {"input": [{"Column": "i"}], "function": ["Boolean"]}},
            {"Column": ["i"]},
            {"Literal": "nonsense"},
            ["not", "a", "node"],
            {"Two": 1, "Keys": 2},
        ],
        ids=lambda tree: str(tree)[:40],
    )
    def test_a_shape_it_does_not_know_is_left_to_polars(self, tree: object) -> None:
        import polars as pl

        from duckdb.frame import _arrow

        # Anything else escaping would fail the query, where the contract is that polars applies what is unknown.
        with pytest.raises(_arrow._Untranslatable):
            _arrow._Predicate(pl, {"i": (1, pl.Int64, True)}, None).sql(tree)

    def test_a_literal_on_the_left_compares_from_the_other_side(self) -> None:
        import json

        import polars as pl

        from duckdb.frame import _arrow

        # polars puts the column first itself, so only a serialized form can say whether the flip holds.
        given = json.loads(received(pl.col("i") > 1, {"i": pl.Int64}).meta.serialize(format="json"))
        literal = given["BinaryExpr"]["right"]
        tree = {"BinaryExpr": {"left": literal, "op": "Lt", "right": {"Column": "i"}}}
        assert _arrow._Predicate(pl, {"i": (1, pl.Int64, True)}, None).sql(tree) == '"_c1" > 1'

    def test_values_bind_after_the_plans_own(self) -> None:
        import polars as pl

        from duckdb.frame import _arrow

        columns = {"s": (2, pl.String, True), "d": (3, pl.Date, True), "i": (1, pl.Int64, True)}
        day = datetime.date(2026, 1, 2)
        schema = {"s": pl.String, "d": pl.Date, "i": pl.Int64}
        predicate = received((pl.col("s") == "x") & pl.col("d").is_in([day]) & (pl.col("i") > 4), schema)
        pushed, used, values, residual = _arrow._split(pl, predicate, columns, [7])
        assert residual is None
        assert values is not None
        assert (used, len(values), values[0]) == ({1, 2, 3}, 3, 7)
        # Numbers are written in as DuckDB types them; text and dates are bound.
        resolved = sorted(re.sub(r"\$(\d+)", lambda n: repr(values[int(n[1]) - 1]), sql) for sql in pushed)
        assert resolved == ['"_c1" > 4', "(\"_c2\" COLLATE C) = 'x'", '("_c3" IN (datetime.date(2026, 1, 2)))']


@pytest.mark.requires("polars")
@pytest.mark.parametrize("version", ["1.21.0", "1.32.1"])
def test_a_polars_too_old_for_lazy_frames_is_named(monkeypatch: pytest.MonkeyPatch, version: str) -> None:
    import polars as pl

    con = frame.connect()
    monkeypatch.setattr(pl, "__version__", version)
    with pytest.raises(ImportError, match=rf"polars 1\.33 or newer.*not {version}"):
        frame.sql("SELECT 1 AS a").to_polars(con, lazy=True)
    # The eager path asks for less.
    assert frame.sql("SELECT 1 AS a").to_polars(con)["a"].to_list() == [1]


@pytest.mark.requires("polars")
def test_a_polars_that_reads_one_batch_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import polars as pl

    con = frame.connect()
    monkeypatch.setattr(pl, "__version__", "1.9.0")
    with pytest.raises(ImportError, match=r"polars 1\.10 or newer.*not 1\.9\.0"):
        frame.sql("SELECT 1 AS a").to_polars(con)
    assert not con._live_streams


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
