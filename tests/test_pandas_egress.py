"""pandas egress: the previous duckdb package's dtypes, except where its frames disagreed with its own rows."""

from __future__ import annotations

import contextlib
import datetime
import subprocess
import sys
import textwrap
import threading
import uuid
from decimal import Decimal

import pytest

from duckdb import exceptions, frame
from duckdb.frame import _pandas

with contextlib.suppress(ModuleNotFoundError):
    import numpy as np
    import pandas as pd

#: The engine's vector size; past it a result arrives in more than one batch.
VECTOR = 2048


@pytest.fixture
def con() -> frame.Connection:
    return frame.connect()


def out(con: frame.Connection, sql: str, *, date_as_object: bool = False) -> pd.DataFrame:
    return frame.sql(sql).to_pandas(con, date_as_object=date_as_object)


@pytest.mark.requires("pandas")
class TestNumbers:
    @pytest.mark.parametrize(
        ("type_text", "plain", "nullable"),
        [
            ("BOOLEAN", "bool", "boolean"),
            ("TINYINT", "int8", "Int8"),
            ("SMALLINT", "int16", "Int16"),
            ("INTEGER", "int32", "Int32"),
            ("BIGINT", "int64", "Int64"),
            ("UTINYINT", "uint8", "UInt8"),
            ("USMALLINT", "uint16", "UInt16"),
            ("UINTEGER", "uint32", "UInt32"),
            ("UBIGINT", "uint64", "UInt64"),
        ],
    )
    def test_a_null_switches_to_the_nullable_dtype(
        self, con: frame.Connection, type_text: str, plain: str, nullable: str
    ) -> None:
        clean = out(con, f"SELECT c::{type_text} AS c FROM (VALUES (1), (0)) t(c)")
        assert str(clean["c"].dtype) == plain
        holed = out(con, f"SELECT c::{type_text} AS c FROM (VALUES (1), (NULL)) t(c)")
        assert str(holed["c"].dtype) == nullable
        assert holed["c"][1] is pd.NA

    def test_nullable_integers_keep_every_bit(self, con: frame.Connection) -> None:
        sql = (
            "SELECT * FROM (VALUES (18446744073709551615::UBIGINT, (-9223372036854775808)::BIGINT),"
            " (NULL, 9223372036854775807)) t(u, i)"
        )
        frame_ = out(con, sql)
        assert frame_["u"][0] == 18446744073709551615
        assert frame_["i"].tolist() == [-9223372036854775808, 9223372036854775807]

    def test_a_null_in_a_later_batch_decides_for_the_whole_column(self, con: frame.Connection) -> None:
        count = 3 * VECTOR + 5
        frame_ = out(con, f"SELECT CASE WHEN i = {count - 1} THEN NULL ELSE i END::INT AS c FROM range({count}) t(i)")
        assert str(frame_["c"].dtype) == "Int32"
        assert frame_["c"][: count - 1].tolist() == list(range(count - 1))
        assert frame_["c"][count - 1] is pd.NA

    @pytest.mark.parametrize(("type_text", "dtype"), [("FLOAT", "float32"), ("DOUBLE", "float64")])
    def test_floats_keep_nan_for_null(self, con: frame.Connection, type_text: str, dtype: str) -> None:
        frame_ = out(con, f"SELECT c::{type_text} AS c FROM (VALUES (1.5), (NULL)) t(c)")
        assert str(frame_["c"].dtype) == dtype
        assert frame_["c"][0] == 1.5
        assert np.isnan(frame_["c"][1])

    @pytest.mark.parametrize(
        ("type_text", "literal", "value"),
        [
            ("DECIMAL(4, 1)", "1.5", 1.5),
            ("DECIMAL(18, 3)", "123456789012345.125", 123456789012345.125),
            ("DECIMAL(38, 10)", "12345678901234567890.0123456789", 1.2345678901234567e19),
            ("HUGEINT", "170141183460469231731687303715884105727", 1.7014118346046923e38),
            ("UHUGEINT", "340282366920938463463374607431768211455", 3.402823669209385e38),
        ],
    )
    def test_wide_numbers_become_float64(
        self, con: frame.Connection, type_text: str, literal: str, value: float
    ) -> None:
        frame_ = out(con, f"SELECT c::{type_text} AS c FROM (VALUES ({literal}), (NULL)) t(c)")
        assert str(frame_["c"].dtype) == "float64"
        assert frame_["c"][0] == pytest.approx(value)
        assert np.isnan(frame_["c"][1])


@pytest.mark.requires("pandas")
class TestTemporal:
    def test_date_is_datetime64_us_with_nat(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT * FROM (VALUES (DATE '2026-01-02'), (NULL)) t(c)")
        assert str(frame_["c"].dtype) == "datetime64[us]"
        assert frame_["c"][0] == pd.Timestamp("2026-01-02")
        assert frame_["c"][1] is pd.NaT

    def test_date_as_object(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT * FROM (VALUES (DATE '2026-01-02'), (NULL)) t(c)", date_as_object=True)
        assert frame_["c"][0] == datetime.date(2026, 1, 2)
        assert pd.isna(frame_["c"][1])

    def test_infinite_dates_are_the_ends_rows_give(self, con: frame.Connection) -> None:
        sql = "SELECT 'infinity'::DATE AS top, '-infinity'::DATE AS bottom"
        frame_ = out(con, sql)
        assert frame_["top"][0].date() == datetime.date.max == frame.sql(sql).rows(con)[0][0]
        assert frame_["bottom"][0].date() == datetime.date.min == frame.sql(sql).rows(con)[0][1]

    @pytest.mark.parametrize("day", ["5877641-06-25", "-5877641-06-25"])
    def test_a_date_past_datetime64_us_raises(self, con: frame.Connection, day: str) -> None:
        with pytest.raises(exceptions.ConversionError, match="datetime64"):
            out(con, f"SELECT DATE '{day}' AS c")
        assert out(con, "SELECT 1 AS n")["n"].tolist() == [1]

    def test_a_far_date_behind_a_null_raises_too(self, con: frame.Connection) -> None:
        with pytest.raises(exceptions.ConversionError):
            out(con, "SELECT * FROM (VALUES (NULL), (DATE '5877641-06-25')) t(c)")

    @pytest.mark.parametrize(
        ("type_text", "dtype"),
        [("TIMESTAMP_S", "datetime64[s]"), ("TIMESTAMP_MS", "datetime64[ms]"), ("TIMESTAMP", "datetime64[us]")],
    )
    def test_timestamps_keep_their_unit(self, con: frame.Connection, type_text: str, dtype: str) -> None:
        frame_ = out(con, f"SELECT * FROM (VALUES ({type_text} '2026-01-02 03:04:05'), (NULL)) t(c)")
        assert str(frame_["c"].dtype) == dtype
        assert frame_["c"][0] == pd.Timestamp("2026-01-02 03:04:05")
        assert frame_["c"][1] is pd.NaT

    def test_nanosecond_timestamps_keep_their_nanoseconds(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT TIMESTAMP_NS '2026-01-02 03:04:05.123456789' AS c")
        assert str(frame_["c"].dtype) == "datetime64[ns]"
        assert frame_["c"][0].nanosecond == 789

    def test_infinite_timestamps_are_the_ends_rows_give(self, con: frame.Connection) -> None:
        sql = "SELECT 'infinity'::TIMESTAMP AS top, '-infinity'::TIMESTAMP AS bottom"
        frame_ = out(con, sql)
        rows = frame.sql(sql).rows(con)[0]
        assert frame_["top"][0].to_pydatetime() == rows[0] == datetime.datetime.max
        assert frame_["bottom"][0].to_pydatetime() == rows[1] == datetime.datetime.min

    def test_timestamptz_is_shown_in_the_session_zone(self, con: frame.Connection) -> None:
        con.run("SET TimeZone = 'Europe/Amsterdam'")
        frame_ = out(con, "SELECT * FROM (VALUES (TIMESTAMPTZ '2026-07-02 03:04:05+00'), (NULL)) t(c)")
        assert str(frame_["c"].dtype) == "datetime64[us, Europe/Amsterdam]"
        assert frame_["c"][0] == pd.Timestamp("2026-07-02 03:04:05", tz="UTC")
        assert frame_["c"][0].hour == 5
        assert frame_["c"][1] is pd.NaT

    def test_a_changed_zone_applies_to_the_next_frame(self, con: frame.Connection) -> None:
        sql = "SELECT TIMESTAMPTZ '2026-07-02 03:04:05+00' AS c"
        con.run("SET TimeZone = 'UTC'")
        assert out(con, sql)["c"][0].hour == 3
        con.run("SET TimeZone = 'Asia/Tokyo'")
        assert out(con, sql)["c"][0].hour == 12

    def test_daylight_saving_past_2038_is_right(self, con: frame.Connection) -> None:
        # The previous package's rows went through pytz, which stops applying daylight saving past 2038.
        con.run("SET TimeZone = 'America/New_York'")
        stamp = out(con, "SELECT TIMESTAMPTZ '2040-07-01 12:00:00+00' AS c")["c"][0]
        assert stamp.utcoffset() == datetime.timedelta(hours=-4)

    def test_a_zone_only_icu_knows_refuses_timestamptz_alone(self, con: frame.Connection) -> None:
        # The engine accepts ICU's legacy three-letter IDs; the standard library's zone rules have no PST.
        con.run("SET TimeZone = 'PST'")
        with pytest.raises(exceptions.ConversionError, match="'PST'"):
            out(con, "SELECT TIMESTAMPTZ '2026-07-02 03:04:05+00' AS c")
        assert out(con, "SELECT TIMESTAMP '2026-07-02 03:04:05' AS c")["c"][0].hour == 3

    def test_nanosecond_timestamptz_keeps_its_nanoseconds(self, con: frame.Connection) -> None:
        con.run("SET TimeZone = 'UTC'")
        frame_ = out(con, "SELECT TIMESTAMPTZ_NS '2026-07-02 03:04:05.123456789+00' AS c")
        assert str(frame_["c"].dtype) == "datetime64[ns, UTC]"
        assert frame_["c"][0].nanosecond == 789

    @pytest.mark.parametrize("zone", ["Asia/Tokyo", "America/New_York", "Pacific/Kiritimati", "Pacific/Pago_Pago"])
    @pytest.mark.parametrize("type_text", ["TIMESTAMPTZ", "TIMESTAMPTZ_NS"])
    def test_infinite_timestamptz_stays_readable_in_any_zone(
        self, con: frame.Connection, zone: str, type_text: str
    ) -> None:
        # Clamped in UTC, an infinity lands past the year 9999 or past nanoseconds once shown east or west of it.
        con.run(f"SET TimeZone = '{zone}'")
        frame_ = out(con, f"SELECT 'infinity'::{type_text} AS top, '-infinity'::{type_text} AS bottom")
        text = str(frame_)
        top, bottom = frame_["top"][0], frame_["bottom"][0]
        assert top > pd.Timestamp("2262-01-01", tz="UTC")
        assert bottom < pd.Timestamp("1678-01-01", tz="UTC")
        assert str(top.year) in text

    def test_interval_is_timedelta64_us_with_nat(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT * FROM (VALUES (INTERVAL '1 month 2 days 3 seconds'), (NULL)) t(c)")
        assert str(frame_["c"].dtype) == "timedelta64[us]"
        assert frame_["c"][0] == pd.Timedelta(days=32, seconds=3)
        assert frame_["c"][1] is pd.NaT

    def test_an_interval_past_timedelta64_raises(self, con: frame.Connection) -> None:
        with pytest.raises(exceptions.ConversionError, match="timedelta64"):
            out(con, "SELECT INTERVAL '110000000 days' AS c")


@pytest.mark.requires("pandas")
class TestText:
    def test_varchar_takes_pandas_own_string_dtype(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT * FROM (VALUES ('a'), (NULL), ('é🦆')) t(c)")
        assert frame_["c"].dtype == pd.Series(["x"]).dtype
        assert frame_["c"][0] == "a"
        assert frame_["c"][2] == "é🦆"
        assert pd.isna(frame_["c"][1])

    @pytest.mark.parametrize(
        "sql", ["SELECT 'x' AS c WHERE false", "SELECT NULL::VARCHAR AS c", "SELECT 'x'::JSON AS c WHERE false"]
    )
    def test_an_empty_or_all_null_varchar_keeps_the_string_dtype(self, con: frame.Connection, sql: str) -> None:
        # The previous package left these as object, so a column's dtype depended on its rows.
        assert out(con, sql)["c"].dtype == pd.Series(["x"]).dtype

    def test_strings_cross_batches_in_order(self, con: frame.Connection) -> None:
        count = 2 * VECTOR + 7
        frame_ = out(con, f"SELECT 'v' || i AS c FROM range({count}) t(i)")
        assert frame_["c"].tolist() == [f"v{i}" for i in range(count)]

    def test_the_varchar_type_id_copy_matches_the_engine(self, con: frame.Connection) -> None:
        with con._execute("SELECT 'x' AS c") as live:
            assert live.result.schema_types[0][0] == _pandas._VARCHAR


@pytest.mark.requires("pandas")
class TestObjects:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT '\\x01\\x02'::BLOB AS c",
            "SELECT '0101'::BIT AS c",
            "SELECT '00000000-0000-0000-0000-000000000001'::UUID AS c",
            "SELECT TIME '01:02:03.456789' AS c",
            "SELECT TIMETZ '01:02:03+05' AS c",
            "SELECT 12345678901234567890123::VARINT AS c",
            "SELECT [1, NULL, 3] AS c",
            "SELECT [[1], [2, 3]] AS c",
            "SELECT [1, 2, 3]::INT[3] AS c",
            "SELECT {'a': 1, 'b': 'x'} AS c",
            "SELECT MAP {'k': 1, 'j': 2} AS c",
            "SELECT MAP {1: 'a'} AS c",
            "SELECT union_value(a := 1)::UNION(a INT, b VARCHAR) AS c",
        ],
    )
    def test_cells_are_what_rows_give(self, con: frame.Connection, sql: str) -> None:
        # Where the previous package's frames disagreed with its rows (bytearray, BIT's internal encoding, numpy
        # arrays for lists, TIMETZ and BIGNUM refused), the rows win.
        frame_ = out(con, f"{sql} UNION ALL SELECT NULL")
        rows = frame.sql(sql).rows(con)
        assert frame_["c"].dtype == object
        assert frame_["c"][0] == rows[0][0]
        assert type(frame_["c"][0]) is type(rows[0][0])
        assert frame_["c"][1] is pd.NA

    def test_a_uuid_is_a_uuid(self, con: frame.Connection) -> None:
        assert isinstance(out(con, "SELECT uuid() AS c")["c"][0], uuid.UUID)

    def test_a_decimal_inside_a_list_stays_exact(self, con: frame.Connection) -> None:
        assert out(con, "SELECT [1.25::DECIMAL(4, 2)] AS c")["c"][0] == [Decimal("1.25")]

    def test_a_null_typed_column_is_object(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT NULL AS c")
        assert frame_["c"].dtype == object
        assert frame_["c"][0] is pd.NA


@pytest.mark.requires("pandas")
class TestEnum:
    @pytest.fixture
    def mood(self, con: frame.Connection) -> frame.Connection:
        con.run("CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy')")
        return con

    def test_an_enum_is_an_ordered_categorical_over_every_label(self, mood: frame.Connection) -> None:
        frame_ = out(mood, "SELECT c::mood AS c FROM (VALUES ('happy'), (NULL), ('sad')) t(c)")
        dtype = frame_["c"].dtype
        assert isinstance(dtype, pd.CategoricalDtype)
        assert dtype.ordered
        assert list(dtype.categories) == ["sad", "ok", "happy"]
        assert frame_["c"][0] == "happy"
        assert pd.isna(frame_["c"][1])
        # The declared order, not the text's: 'sad' sorts before 'happy' here.
        assert (frame_["c"] < "happy").tolist() == [False, False, True]

    def test_an_empty_enum_keeps_its_categories(self, mood: frame.Connection) -> None:
        dtype = out(mood, "SELECT 'ok'::mood AS c WHERE false")["c"].dtype
        assert list(dtype.categories) == ["sad", "ok", "happy"]

    def test_a_wide_enum_reads_through_wider_codes(self, con: frame.Connection) -> None:
        labels = ", ".join(f"'l{i}'" for i in range(300))
        con.run(f"CREATE TYPE wide AS ENUM ({labels})")
        frame_ = out(con, "SELECT c::wide AS c FROM (VALUES ('l299'), ('l0')) t(c)")
        assert frame_["c"].tolist() == ["l299", "l0"]


@pytest.mark.requires("pandas")
class TestShape:
    @pytest.mark.parametrize(
        ("sql", "names"),
        [
            ("SELECT 1 AS a, 2 AS a, 3 AS A", ["a", "a_1", "A_2"]),
            ("SELECT 1 AS a_1, 2 AS a, 3 AS a, 4 AS a_2", ["a_1", "a", "a_2", "a_2_1"]),
            ("SELECT 1 AS a, 2 AS b, 3 AS a, 4 AS a, 5 AS b", ["a", "b", "a_1", "a_2", "b_1"]),
        ],
    )
    def test_repeated_names_take_the_engine_suffix(self, con: frame.Connection, sql: str, names: list[str]) -> None:
        frame_ = out(con, sql)
        assert list(frame_.columns) == names
        assert frame_.iloc[0].tolist() == list(range(1, len(names) + 1))

    def test_the_index_is_a_range(self, con: frame.Connection) -> None:
        frame_ = out(con, "SELECT i FROM range(3) t(i)")
        assert isinstance(frame_.index, pd.RangeIndex)
        assert list(frame_.index) == [0, 1, 2]

    def test_an_empty_result_keeps_every_dtype(self, con: frame.Connection) -> None:
        frame_ = out(
            con,
            "SELECT 1::INT AS i, 1.5::DOUBLE AS f, DATE '2026-01-01' AS d, TIMESTAMP '2026-01-01' AS t, [1] AS l"
            " WHERE false",
        )
        assert len(frame_) == 0
        assert [str(t) for t in frame_.dtypes] == ["int32", "float64", "datetime64[us]", "datetime64[us]", "object"]

    def test_a_many_batch_result_matches_rows(self, con: frame.Connection) -> None:
        sql = (
            "SELECT i, CASE WHEN i % 7 = 0 THEN NULL ELSE i * 1.5 END AS f, 's' || i AS s"
            f" FROM range({4 * VECTOR + 11}) t(i) ORDER BY i"
        )
        frame_ = out(con, sql)
        rows = frame.sql(sql).rows(con)
        assert frame_["i"].tolist() == [r[0] for r in rows]
        assert frame_["s"].tolist() == [r[2] for r in rows]
        assert [None if np.isnan(v) else v for v in frame_["f"]] == [r[1] for r in rows]

    def test_parameters_flow_through(self, con: frame.Connection) -> None:
        plan = frame.sql("SELECT i AS n FROM range(5) t(i)").filter(frame.col("n") >= frame.param("floor"))
        assert plan.to_pandas(con, parameters={"floor": 3})["n"].tolist() == [3, 4]

    def test_the_bound_forwarder(self, con: frame.Connection) -> None:
        bound = frame.sql("SELECT DATE '2026-01-02' AS d").on(con)
        assert bound.to_pandas()["d"][0] == pd.Timestamp("2026-01-02")
        assert bound.to_pandas(date_as_object=True)["d"][0] == datetime.date(2026, 1, 2)


@pytest.mark.requires("pandas")
class TestFailures:
    def test_an_engine_error_is_raised_typed_and_frees_the_connection(self, con: frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="boom"):
            out(con, f"SELECT CASE WHEN i = {3 * VECTOR} THEN error('boom') END FROM range({4 * VECTOR}) t(i)")
        assert out(con, "SELECT 1 AS n")["n"].tolist() == [1]

    def test_a_closed_connection_is_refused(self, con: frame.Connection) -> None:
        con.close()
        with pytest.raises(exceptions.InterfaceError, match="closed"):
            out(con, "SELECT 1")

    def test_an_interrupt_lands_typed(self, con: frame.Connection) -> None:
        timer = threading.Timer(0.2, con.interrupt)
        timer.start()
        try:
            with pytest.raises(exceptions.InterruptError):
                out(con, "SELECT sum(a.range * b.range) AS s FROM range(200000) a, range(200000) b")
        finally:
            timer.cancel()
        assert out(con, "SELECT 1 AS n")["n"].tolist() == [1]

    def test_a_scalar_function_feeds_the_conversion(self) -> None:
        # In a child with a timeout, so a GIL deadlock between the fetch and the function fails rather than hangs.
        probe = textwrap.dedent(
            """
            import duckdb

            con = duckdb.frame.connect()
            con.create_function("twice", lambda x: x * 2, ["BIGINT"], "BIGINT")
            out = duckdb.frame.sql("SELECT twice(i) AS n FROM range(200000) t(i)").to_pandas(con)
            assert len(out) == 200000 and out["n"][3] == 6
            print("fed")
            """
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False)
        assert done.returncode == 0, done.stderr
        assert "fed" in done.stdout


def test_missing_pandas_is_named() -> None:
    probe = textwrap.dedent(
        """
        import sys

        sys.modules["pandas"] = None
        import duckdb

        con = duckdb.frame.connect()
        try:
            duckdb.frame.sql("SELECT 1").to_pandas(con)
        except ImportError as error:
            assert "pandas" in str(error), error
            print("named")
        assert duckdb.frame.sql("SELECT 1 AS n").rows(con) == [(1,)]
        """
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert "named" in done.stdout
