"""Engine values become the right Python objects, including at the edges."""

from __future__ import annotations

import datetime
import decimal
import uuid
from typing import Any

import pytest

from duckdb import _duckdb, exceptions


@pytest.fixture(scope="module")
def con() -> _duckdb.Connection:
    return _duckdb.Database(":memory:").connect()


def scalar(con: _duckdb.Connection, sql: str) -> Any:  # noqa: ANN401
    # Any is the honest type: the point of these tests is that the converter
    # returns a different Python type per SQL type.
    return con.execute(sql).fetch_all()[0][0]


ROUNDTRIP = [
    ("SELECT true", True),
    ("SELECT 1::TINYINT", 1),
    ("SELECT 9223372036854775807::BIGINT", 9223372036854775807),
    ("SELECT 18446744073709551615::UBIGINT", 18446744073709551615),
    ("SELECT 'héllo'", "héllo"),
    ("SELECT '\\xDE\\xAD'::BLOB", b"\xde\xad"),
    ("SELECT DATE '2026-08-27'", datetime.date(2026, 8, 27)),
    ("SELECT TIME '13:45:06.123456'", datetime.time(13, 45, 6, 123456)),
    ("SELECT TIMESTAMP '2026-08-27 13:45:06.123456'", datetime.datetime(2026, 8, 27, 13, 45, 6, 123456)),
    ("SELECT TIMESTAMP_S '2026-08-27 13:45:06'", datetime.datetime(2026, 8, 27, 13, 45, 6)),
    ("SELECT TIMESTAMP_MS '2026-08-27 13:45:06.123'", datetime.datetime(2026, 8, 27, 13, 45, 6, 123000)),
    ("SELECT INTERVAL '1 month 2 days 3 hours'", datetime.timedelta(days=32, seconds=10800)),
    ("SELECT [1, NULL, 3]", [1, None, 3]),
    ("SELECT {'a': 1, 'b': 'x'}", {"a": 1, "b": "x"}),
    ("SELECT MAP{'k': 1}", {"k": 1}),
    ("SELECT []::INT[]", []),
    ("SELECT [{'a': [1,2]}, {'a': [3]}]", [{"a": [1, 2]}, {"a": [3]}]),
    ("SELECT NULL::INTEGER", None),
    ("SELECT union_value(n := 2)", 2),
]


@pytest.mark.parametrize(("sql", "expected"), ROUNDTRIP)
def test_value_converts(con: _duckdb.Connection, sql: str, expected: object) -> None:
    assert scalar(con, sql) == expected


def test_hugeint_extremes_are_exact(con: _duckdb.Connection) -> None:
    # Wider than any C integer, so the conversion goes through the text form.
    top = 170141183460469231731687303715884105727
    assert scalar(con, f"SELECT {top}::HUGEINT") == top
    assert scalar(con, f"SELECT ({-top - 1})::HUGEINT") == -top - 1


def test_decimal_keeps_its_scale(con: _duckdb.Connection) -> None:
    # Deliberately not float: the scale is part of the value.
    value = scalar(con, "SELECT 0.10::DECIMAL(18,2)")
    assert value == decimal.Decimal("0.10")
    assert str(value) == "0.10"


def test_uuid_becomes_a_uuid(con: _duckdb.Connection) -> None:
    assert scalar(con, "SELECT '10203040-5060-7080-90a0-b0c0d0e0f000'::UUID") == uuid.UUID(
        "10203040-5060-7080-90a0-b0c0d0e0f000"
    )


def test_nanosecond_timestamps_truncate_to_microseconds(con: _duckdb.Connection) -> None:
    # Python's datetime resolves to microseconds. A deliberate divergence, so
    # pin it rather than let a future change round instead of truncate.
    assert scalar(con, "SELECT TIMESTAMP_NS '2026-08-27 13:45:06.123456789'") == datetime.datetime(
        2026, 8, 27, 13, 45, 6, 123456
    )


def test_time_tz_carries_its_offset(con: _duckdb.Connection) -> None:
    value = scalar(con, "SELECT TIMETZ '13:45:06+02:00'")
    assert value.utcoffset() == datetime.timedelta(hours=2)


class TestInfinity:
    """DuckDB's infinite dates have no Python counterpart.

    They are clamped to date/datetime min and max, matching the previous client
    and the adopted suite. A deliberate divergence: a clamped value no longer
    round-trips as infinite.
    """

    def test_positive_date(self, con: _duckdb.Connection) -> None:
        assert scalar(con, "SELECT 'infinity'::DATE") == datetime.date.max

    def test_negative_date(self, con: _duckdb.Connection) -> None:
        assert scalar(con, "SELECT '-infinity'::DATE") == datetime.date.min

    def test_positive_timestamp(self, con: _duckdb.Connection) -> None:
        assert scalar(con, "SELECT 'infinity'::TIMESTAMP") == datetime.datetime.max

    def test_negative_timestamp(self, con: _duckdb.Connection) -> None:
        assert scalar(con, "SELECT '-infinity'::TIMESTAMP") == datetime.datetime.min

    def test_stays_aware_for_tz_columns(self, con: _duckdb.Connection) -> None:
        # A naive limit here would raise TypeError the moment anyone compared
        # it against an aware datetime.
        value = scalar(con, "SELECT 'infinity'::TIMESTAMPTZ")
        assert value.tzinfo is not None
        assert value > datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)

    def test_inside_a_list(self, con: _duckdb.Connection) -> None:
        assert scalar(con, "SELECT ['infinity'::DATE, DATE '2020-01-01']") == [
            datetime.date.max,
            datetime.date(2020, 1, 1),
        ]


def test_dates_beyond_python_report_a_conversion_error(con: _duckdb.Connection) -> None:
    # DuckDB reaches year 5874897; Python stops at 9999. The engine value is
    # valid, so this must name the value rather than leak OverflowError with an
    # internal day count.
    with pytest.raises(exceptions.ConversionError, match="10000-01-01"):
        scalar(con, "SELECT '10000-01-01'::DATE")


def test_unknown_types_degrade_to_text(con: _duckdb.Connection) -> None:
    # The open tail: a type with no mapping returns its SQL text rather than
    # failing the whole fetch. This is what keeps future engine types readable.
    assert scalar(con, "SELECT '101'::BIT") == "101"


def test_schema_reports_names_and_types(con: _duckdb.Connection) -> None:
    result = con.execute("SELECT 1::INTEGER AS a, 'x' AS b")
    assert result.schema == [("a", "INTEGER"), ("b", "VARCHAR")]


def test_engine_errors_surface_as_typed_exceptions(con: _duckdb.Connection) -> None:
    with pytest.raises(exceptions.CatalogError):
        con.execute("SELECT * FROM no_such_table").fetch_all()
    with pytest.raises(exceptions.ParserError):
        con.execute("SELECT FROM WHERE").fetch_all()
