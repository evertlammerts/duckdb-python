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
