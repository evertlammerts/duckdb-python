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


class TestBulkRowConversion:
    # The row path converts column-at-a-time off the flattened vectors; these
    # pin the indexing edges that per-value conversion never had.

    def test_lists_with_null_and_empty_rows_interleaved(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT CASE WHEN i % 3 = 0 THEN NULL WHEN i % 3 = 1 THEN [] ELSE [i, i + 1] END AS l FROM range(7) t(i)"
        ).fetch_all()
        assert rows == [(None,), ([],), ([2, 3],), (None,), ([],), ([5, 6],), (None,)]

    def test_a_partial_fetch_slices_nested_children_correctly(self, con: _duckdb.Connection) -> None:
        result = con.execute("SELECT [i, i * 2] AS l FROM range(5) t(i)")
        assert result.fetch_rows(2) == [([0, 0],), ([1, 2],)]
        assert result.fetch_rows(0) == [([2, 4],), ([3, 6],), ([4, 8],)]

    def test_structs_with_null_rows_and_nested_fields(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT CASE WHEN i = 1 THEN NULL ELSE {'x': i, 'y': [i, NULL], 'z': 'v' || i} END AS st FROM range(3) t(i)"
        ).fetch_all()
        assert rows == [
            ({"x": 0, "y": [0, None], "z": "v0"},),
            (None,),
            ({"x": 2, "y": [2, None], "z": "v2"},),
        ]

    def test_maps_with_null_and_empty_rows(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT CASE WHEN i = 0 THEN NULL WHEN i = 1 THEN MAP([], []) "
            "ELSE MAP(['a', 'b'], [i, i + 1]) END AS m FROM range(4) t(i)"
        ).fetch_all()
        assert rows == [(None,), ({},), ({"a": 2, "b": 3},), ({"a": 3, "b": 4},)]

    def test_arrays_with_null_rows_and_nesting(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT a FROM (VALUES ([0, 1, 2]::INTEGER[3]), (NULL), ([2, 3, 4]::INTEGER[3])) v(a)"
        ).fetch_all()
        assert rows == [([0, 1, 2],), (None,), ([2, 3, 4],)]
        nested = con.execute("SELECT [[1, 2], [3, 4]]::INTEGER[2][2] AS a").fetch_all()
        assert nested == [([[1, 2], [3, 4]],)]

    def test_deep_nesting_and_null_elements(self, con: _duckdb.Connection) -> None:
        assert con.execute("SELECT {'m': MAP(['a'], [[1, 2]])} AS deep").fetch_all() == [({"m": {"a": [1, 2]}},)]
        assert con.execute("SELECT MAP(['a', 'b'], [NULL, {'x': NULL}]) AS m").fetch_all() == [
            ({"a": None, "b": {"x": None}},)
        ]

    def test_nested_values_survive_chunk_boundaries(self, con: _duckdb.Connection) -> None:
        rows = con.execute("SELECT i, [i, NULL, i * 2] AS l, {'k': 'v' || i} AS s FROM range(10000) t(i)").fetch_all()
        assert len(rows) == 10000
        assert rows[9999] == (9999, [9999, None, 19998], {"k": "v9999"})

    def test_int128_values_are_exact_at_the_extremes(self, con: _duckdb.Connection) -> None:
        # The UHUGEINT operand is a literal: an expression like 2 ^ 127 binds
        # as DOUBLE and would test nothing.
        top = 170141183460469231731687303715884105727
        umax = 340282366920938463463374607431768211455
        rows = con.execute(f"SELECT {top}::HUGEINT AS h, (-{top})::HUGEINT AS n, {umax}::UHUGEINT AS u").fetch_all()
        assert rows == [(top, -top, umax)]

    def test_coarse_timestamp_infinities_clamp_like_the_others(self, con: _duckdb.Connection) -> None:
        for unit in ("TIMESTAMP_S", "TIMESTAMP_MS", "TIMESTAMP_NS"):
            rows = con.execute(f"SELECT 'infinity'::{unit} AS p, '-infinity'::{unit} AS n").fetch_all()
            assert rows == [(datetime.datetime.max, datetime.datetime.min)], unit

    def test_an_enum_wider_than_a_byte_uses_its_codes(self, con: _duckdb.Connection) -> None:
        values = ", ".join(f"'value_{i}'" for i in range(300))
        con.execute(f"CREATE TYPE wide_mood AS ENUM ({values})").drain()
        rows = con.execute("SELECT unnest(['value_0', 'value_299', NULL]::wide_mood[]) AS m").fetch_all()
        assert rows == [("value_0",), ("value_299",), (None,)]

    def test_arrays_slice_correctly_through_a_partial_fetch(self, con: _duckdb.Connection) -> None:
        result = con.execute("SELECT [i, i + 1, i + 2]::INTEGER[3] AS a FROM range(4) t(i)")
        assert result.fetch_rows(1) == [([0, 1, 2],)]
        assert result.fetch_rows(0) == [([1, 2, 3],), ([2, 3, 4],), ([3, 4, 5],)]

    def test_filtered_nested_vectors_flatten_before_reading(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT l, s FROM (SELECT [i, NULL, i * 2] AS l, {'k': 'v' || i} AS s, i FROM range(100) t(i)) "
            "WHERE i % 7 = 0 ORDER BY i"
        ).fetch_all()
        assert len(rows) == 15
        assert rows[1] == ([7, None, 14], {"k": "v7"})

    def test_wide_decimals_keep_value_and_scale(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT 12345678901234567890.123456::DECIMAL(38,6) AS d, 1.250::DECIMAL(28,3) AS t, "
            "(-0.000001)::DECIMAL(38,6) AS n"
        ).fetch_all()
        assert rows[0][0] == decimal.Decimal("12345678901234567890.123456")
        assert str(rows[0][1]) == "1.250"
        assert rows[0][2] == decimal.Decimal("-0.000001")

    def test_uuid_bytes_round_the_flipped_sign_bit(self, con: _duckdb.Connection) -> None:
        rows = con.execute(
            "SELECT '550e8400-e29b-41d4-a716-446655440000'::UUID AS u, "
            "'00000000-0000-0000-0000-000000000000'::UUID AS z, "
            "'ffffffff-ffff-ffff-ffff-ffffffffffff'::UUID AS f"
        ).fetch_all()
        assert rows == [
            (
                uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),
                uuid.UUID(int=0),
                uuid.UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"),
            )
        ]
