"""Python values bind as parameters and come back unchanged."""

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


def roundtrip(con: _duckdb.Connection, value: object) -> Any:  # noqa: ANN401
    # Any is the honest type: the point is that each Python type survives.
    return con.execute("SELECT $1", [value]).fetch_all()[0][0]


ROUNDTRIP = [
    None,
    True,
    False,
    0,
    -1,
    9223372036854775807,
    -9223372036854775808,
    3.14,
    "héllo",
    b"\xde\xad",
    datetime.date(2026, 8, 27),
    datetime.datetime(2026, 8, 27, 13, 45, 6, 123456),
    datetime.time(13, 45, 6, 123456),
    datetime.timedelta(days=3, seconds=10800, microseconds=5),
    uuid.UUID("10203040-5060-7080-90a0-b0c0d0e0f000"),
    [1, 2, 3],
    [1, None, 3],
    [],
    {"a": 1, "b": "x"},
]


@pytest.mark.parametrize("value", ROUNDTRIP, ids=repr)
def test_value_survives_a_roundtrip(con: _duckdb.Connection, value: object) -> None:
    assert roundtrip(con, value) == value


def test_bool_is_not_bound_as_an_integer(con: _duckdb.Connection) -> None:
    # In Python a bool IS an int, so a converter that tests int first binds
    # True as 1 and the type is silently lost.
    assert roundtrip(con, True) is True
    assert con.execute("SELECT typeof($1)", [True]).fetch_all()[0][0] == "BOOLEAN"


@pytest.mark.parametrize(
    "value",
    [
        170141183460469231731687303715884105727,  # HUGEINT max
        -170141183460469231731687303715884105728,  # HUGEINT min
        340282366920938463463374607431768211455,  # UHUGEINT max
    ],
    ids=["hugeint_max", "hugeint_min", "uhugeint_max"],
)
def test_integers_wider_than_64_bits(con: _duckdb.Connection, value: int) -> None:
    # Past int64 the value goes through its text form, which is exact for
    # integers of any width.
    assert roundtrip(con, value) == value


def test_aware_datetime_keeps_its_offset(con: _duckdb.Connection) -> None:
    aware = datetime.datetime(2026, 8, 27, 13, 45, 6, tzinfo=datetime.UTC)
    assert roundtrip(con, aware) == aware
    assert con.execute("SELECT typeof($1)", [aware]).fetch_all()[0][0] == "TIMESTAMP WITH TIME ZONE"


def test_naive_datetime_stays_naive(con: _duckdb.Connection) -> None:
    assert con.execute("SELECT typeof($1)", [datetime.datetime(2026, 8, 27)]).fetch_all()[0][0] == "TIMESTAMP"


class TestDecimal:
    """Width and scale come from the value, not from a fixed default.

    A fixed DECIMAL(38,10) would repad, turning Decimal("123.456") into
    Decimal("123.4560000000") and losing the scale the caller chose.
    """

    @pytest.mark.parametrize("text", ["123.456", "0.10", "-5", "0", "-0.001", "99999999999999999999"])
    def test_scale_is_preserved_exactly(self, con: _duckdb.Connection, text: str) -> None:
        value = decimal.Decimal(text)
        assert repr(roundtrip(con, value)) == repr(value)

    def test_positive_exponent_widens_instead_of_truncating(self, con: _duckdb.Connection) -> None:
        # Decimal("1E+2") carries one digit and an exponent; the trailing zeroes
        # are not in the digit tuple, so the width has to account for them.
        assert roundtrip(con, decimal.Decimal("1E+2")) == decimal.Decimal(100)

    @pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_is_refused_clearly(self, con: _duckdb.Connection, text: str) -> None:
        # These have no DECIMAL counterpart. Refuse with a message about the
        # Decimal rather than letting a cast fail deeper down.
        with pytest.raises(exceptions.InvalidInputError, match="non-finite"):
            roundtrip(con, decimal.Decimal(text))


def test_null_casts_into_any_target(con: _duckdb.Connection) -> None:
    # A NULL needs a carrier type, but must not pin the statement to it: the
    # binder has to stay free to land wherever the query wants.
    assert con.execute("SELECT $1::VARCHAR, $1::INTEGER, $1::DATE", [None]).fetch_all()[0] == (
        None,
        None,
        None,
    )


def test_named_parameters(con: _duckdb.Connection) -> None:
    assert con.execute("SELECT $a + $b", {"a": 2, "b": 3}).fetch_all()[0][0] == 5


def test_positional_parameters_bind_in_order(con: _duckdb.Connection) -> None:
    assert con.execute("SELECT $1, $2", ["first", "second"]).fetch_all()[0] == ("first", "second")


def test_dict_maps_to_struct_or_map_by_its_keys(con: _duckdb.Connection) -> None:
    # A dict is the only Python type mapping onto two DuckDB types, so the rule
    # is stated rather than guessed: string keys are a STRUCT, anything else a
    # MAP. Assert the shape, not the engine's exact rendering of it.
    struct_type = con.execute("SELECT typeof($1)", [{"a": 1}]).fetch_all()[0][0]
    assert struct_type.startswith("STRUCT")
    map_type = con.execute("SELECT typeof($1)", [{1: "a"}]).fetch_all()[0][0]
    assert map_type.startswith("MAP")


def test_unsupported_type_names_itself(con: _duckdb.Connection) -> None:
    class Unbindable:
        pass

    with pytest.raises(exceptions.InvalidInputError, match="Unbindable"):
        roundtrip(con, Unbindable())


def test_parameters_reject_multiple_statements(con: _duckdb.Connection) -> None:
    # Binding across a statement boundary has no defined meaning, so refuse it
    # rather than silently binding into the first.
    with pytest.raises(exceptions.InvalidInputError, match="exactly one statement"):
        con.execute("SELECT $1; SELECT $1", [1])


class TestAwareTime:
    """An aware `time` binds as TIME_TZ rather than losing its offset.

    The read direction already returns TIME_TZ aware, so binding it back
    naively made the round trip asymmetric and silently dropped the offset.
    """

    def test_offset_survives_a_roundtrip(self, con: _duckdb.Connection) -> None:
        aware = datetime.time(13, 45, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))
        back = roundtrip(con, aware)
        assert back.utcoffset() == datetime.timedelta(hours=2)
        assert back.replace(tzinfo=None) == aware.replace(tzinfo=None)

    def test_binds_as_time_with_time_zone(self, con: _duckdb.Connection) -> None:
        aware = datetime.time(13, 45, 6, tzinfo=datetime.UTC)
        assert "TIME" in con.execute("SELECT typeof($1)", [aware]).fetch_all()[0][0]

    def test_negative_offset(self, con: _duckdb.Connection) -> None:
        aware = datetime.time(9, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=-5)))
        assert roundtrip(con, aware).utcoffset() == datetime.timedelta(hours=-5)

    def test_naive_time_stays_naive(self, con: _duckdb.Connection) -> None:
        naive = datetime.time(13, 45, 6)
        back = roundtrip(con, naive)
        assert back.tzinfo is None
        assert back == naive

    def test_offset_beyond_the_range_is_refused(self, con: _duckdb.Connection) -> None:
        # TIME_TZ tops out just under 16 hours; refuse rather than wrap.
        too_far = datetime.time(12, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=20)))
        with pytest.raises(exceptions.InvalidInputError, match="offset"):
            roundtrip(con, too_far)


def test_database_options_accepts_none(con: _duckdb.Connection) -> None:
    # The stub allows None, so the binding has to as well.
    assert _duckdb.Database(":memory:", None).connect() is not None
