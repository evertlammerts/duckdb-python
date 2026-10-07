"""Python values bind as parameters and come back unchanged."""

from __future__ import annotations

import collections
import datetime
import decimal
import itertools
import os
import re
import subprocess
import sys
import types
import uuid
import zoneinfo
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import duckdb.frame
from duckdb import _duckdb, exceptions
from duckdb._expressions.expr import render_literal
from duckdb.frame import fn, lit, sql

from ._support import Emptying, Failing, NoOffset

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


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
    {"a": 1, "b": "x"},
]


@pytest.mark.parametrize("value", ROUNDTRIP, ids=repr)
def test_value_survives_a_roundtrip(con: _duckdb.Connection, value: object) -> None:
    assert roundtrip(con, value) == value


def test_bool_is_not_bound_as_an_integer(con: _duckdb.Connection) -> None:
    # A bool is an int in Python, so checking for int first would bind True as 1 and lose the type.
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
    # Past 64 bits the value travels as text, which is exact for integers of any width.
    assert roundtrip(con, value) == value


def test_aware_datetime_keeps_its_offset(con: _duckdb.Connection) -> None:
    aware = datetime.datetime(2026, 8, 27, 13, 45, 6, tzinfo=datetime.UTC)
    assert roundtrip(con, aware) == aware
    assert con.execute("SELECT typeof($1)", [aware]).fetch_all()[0][0] == "TIMESTAMP WITH TIME ZONE"


def test_naive_datetime_stays_naive(con: _duckdb.Connection) -> None:
    assert con.execute("SELECT typeof($1)", [datetime.datetime(2026, 8, 27)]).fetch_all()[0][0] == "TIMESTAMP"


class TestDecimal:
    """Width and scale come from the value: a fixed DECIMAL(38,10) would repad and lose the caller's scale."""

    @pytest.mark.parametrize("text", ["123.456", "0.10", "-5", "0", "-0.001", "99999999999999999999"])
    def test_scale_is_preserved_exactly(self, con: _duckdb.Connection, text: str) -> None:
        value = decimal.Decimal(text)
        assert repr(roundtrip(con, value)) == repr(value)

    def test_positive_exponent_widens_instead_of_truncating(self, con: _duckdb.Connection) -> None:
        # Decimal("1E+2") keeps its zeroes in the exponent, not the digits, so the width must allow for them.
        assert roundtrip(con, decimal.Decimal("1E+2")) == decimal.Decimal(100)

    @pytest.mark.parametrize("text", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_is_refused_clearly(self, con: _duckdb.Connection, text: str) -> None:
        # These have no DECIMAL counterpart, and a message about the Decimal beats a cast failing deeper down.
        with pytest.raises(exceptions.InvalidInputError, match="non-finite"):
            roundtrip(con, decimal.Decimal(text))


def test_null_casts_into_any_target(con: _duckdb.Connection) -> None:
    # A NULL needs some type to travel as, but must not pin the statement to it.
    assert con.execute("SELECT $1::VARCHAR, $1::INTEGER, $1::DATE", [None]).fetch_all()[0] == (
        None,
        None,
        None,
    )


def test_a_non_string_parameter_name_is_refused_clearly(con: _duckdb.Connection) -> None:
    # nanobind's failed cast used to surface as a raw std::bad_cast.
    with pytest.raises(exceptions.InvalidInputError, match="parameter names must be strings"):
        con.execute("SELECT $1", {1: 5})  # type: ignore[dict-item]


def test_named_parameters(con: _duckdb.Connection) -> None:
    assert con.execute("SELECT $a + $b", {"a": 2, "b": 3}).fetch_all()[0][0] == 5


@pytest.mark.parametrize(
    "mapping",
    [collections.UserDict({"a": 2, "b": 3}), types.MappingProxyType({"a": 2, "b": 3})],
    ids=lambda mapping: type(mapping).__name__,
)
def test_any_mapping_binds_by_name(con: _duckdb.Connection, mapping: Mapping[str, int]) -> None:
    assert con.execute("SELECT $a + $b", mapping).fetch_all()[0][0] == 5


def test_a_mapping_with_a_missing_name_is_refused(con: _duckdb.Connection) -> None:
    with pytest.raises(exceptions.Error):
        con.execute("SELECT $a + $b", collections.UserDict({"a": 2}))


def test_a_mapping_with_a_non_string_name_is_refused_clearly(con: _duckdb.Connection) -> None:
    with pytest.raises(exceptions.InvalidInputError, match="parameter names must be strings"):
        con.execute("SELECT $1", collections.UserDict({1: 5}))  # type: ignore[dict-item]


def test_a_generator_still_binds_positionally(con: _duckdb.Connection) -> None:
    values = (v for v in ("first", "second"))
    assert con.execute("SELECT $1, $2", values).fetch_all()[0] == ("first", "second")  # type: ignore[arg-type]


def test_positional_parameters_bind_in_order(con: _duckdb.Connection) -> None:
    assert con.execute("SELECT $1, $2", ["first", "second"]).fetch_all()[0] == ("first", "second")


def test_dict_maps_to_struct_or_map_by_its_keys(con: _duckdb.Connection) -> None:
    # A dict maps onto two DuckDB types: text keys a STRUCT, anything else a MAP, however it is spelled.
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
    # A value spanning two statements has no meaning, so it is refused rather than applied to the first.
    with pytest.raises(exceptions.InvalidInputError, match="exactly one statement"):
        con.execute("SELECT $1; SELECT $1", [1])


class TestAwareTime:
    """An aware `time` goes in as TIME_TZ, since reading one back already keeps its offset."""

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
    # The type stubs allow None, so the code has to as well.
    assert _duckdb.Database(":memory:", None).connect() is not None


NAIVE = datetime.datetime(2020, 1, 1, 12)
AWARE = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC)


class TestTemporalScalars:
    """numpy's and pandas' date and time scalars bind in their own unit, up to the microseconds a parameter holds.

    What that cannot hold is refused rather than rounded.
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.datetime64("2020-01-02", "D"), ("DATE", "2020-01-02")),
            (np.datetime64("2020-01-02T03", "h"), ("TIMESTAMP_S", "2020-01-02 03:00:00")),
            (np.datetime64("2020-01-02T03:04:05", "s"), ("TIMESTAMP_S", "2020-01-02 03:04:05")),
            (np.datetime64("2020-01-02T03:04:05.123456", "us"), ("TIMESTAMP", "2020-01-02 03:04:05.123456")),
            (np.datetime64("2020-01-02T03:04:05.123456000", "ns"), ("TIMESTAMP", "2020-01-02 03:04:05.123456")),
            (np.datetime64("1969-12-31T23:59:59.999999", "us"), ("TIMESTAMP", "1969-12-31 23:59:59.999999")),
            (np.timedelta64(3, "D"), ("INTERVAL", "72:00:00")),
            (pd.Timestamp("2020-01-02 03:04:05").as_unit("s"), ("TIMESTAMP_S", "2020-01-02 03:04:05")),
            (pd.Timestamp("2020-01-02 03:04:05.123456").as_unit("ns"), ("TIMESTAMP", "2020-01-02 03:04:05.123456")),
        ],
        ids=str,
    )
    def test_binds_in_its_own_unit(self, con: _duckdb.Connection, value: object, expected: tuple[str, str]) -> None:
        assert con.execute("SELECT typeof($1), $1::VARCHAR", [value]).fetch_all() == [expected]

    def test_an_aware_pandas_timestamp_binds_as_its_instant(self, con: _duckdb.Connection) -> None:
        stamp = pd.Timestamp("2020-01-02 03:04:05.123456", tz="Europe/Amsterdam").as_unit("ns")
        bound = con.execute("SELECT typeof($1), $1", [stamp]).fetch_all()
        assert bound == [("TIMESTAMP WITH TIME ZONE", stamp.to_pydatetime())]

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (
                np.datetime64("2020-01-02T03:04:05.123456789", "ns"),
                "has nanoseconds, and a query parameter holds microseconds at most",
            ),
            (
                pd.Timestamp("2020-01-02 03:04:05.123456789"),
                "has nanoseconds, and a query parameter holds microseconds at most",
            ),
            (np.datetime64(1000, "ps"), "has nanoseconds, and a query parameter holds microseconds at most"),
            (np.datetime64(1, "ps"), "is finer than the engine's nanoseconds"),
            (np.datetime64(2**63 - 1, "us"), "is beyond the range of its engine type"),
            (np.datetime64(2**42, "M"), "is beyond the range of its engine type"),
        ],
        ids=["ns", "pandas ns", "ps of whole ns", "ps", "reserved", "past days"],
    )
    def test_what_a_parameter_cannot_hold_is_refused(
        self, con: _duckdb.Connection, value: object, message: str
    ) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=re.escape(message)):
            con.execute("SELECT $1", [value])

    @pytest.mark.parametrize(
        "missing",
        [pd.NaT, np.datetime64("NaT", "D"), np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns")],
        ids=["pandas", "numpy days", "numpy ns", "numpy duration"],
    )
    def test_a_missing_value_binds_as_null(self, con: _duckdb.Connection, missing: object) -> None:
        assert roundtrip(con, missing) is None

    def test_a_missing_value_in_a_list_is_null(self, con: _duckdb.Connection) -> None:
        assert roundtrip(con, [NAIVE, pd.NaT]) == [NAIVE, None]

    @pytest.mark.parametrize(
        "values",
        [["a", None], [None, "a"], [datetime.date(2020, 1, 1), None], [[1], None]],
        ids=["text", "null first", "date", "nested"],
    )
    def test_a_missing_list_element_takes_the_type_of_the_others(
        self, con: _duckdb.Connection, values: list[object]
    ) -> None:
        assert roundtrip(con, values) == values

    def test_a_pandas_timedelta_drops_nanoseconds_toward_zero(self, con: _duckdb.Connection) -> None:
        assert roundtrip(con, pd.Timedelta(1500, "ns")) == datetime.timedelta(microseconds=1)
        assert roundtrip(con, pd.Timedelta(-1500, "ns")) == datetime.timedelta(microseconds=-1)

    def test_a_pandas_timedelta_is_laid_out_as_a_timedelta(self, con: _duckdb.Connection) -> None:
        days = [pd.Timedelta("5 days"), datetime.timedelta(days=5)]
        assert con.execute("SELECT date_part('day', $1), date_part('day', $2)", days).fetch_all() == [(5, 5)]


class TestParameterZones:
    """A parameter never needs a time zone it does not name."""

    @pytest.mark.parametrize(
        "values",
        [
            [NAIVE, AWARE],
            [AWARE, NAIVE],
            [None, NAIVE, AWARE],
            [datetime.date(2020, 1, 1), AWARE],
            [datetime.time(12), datetime.time(12, tzinfo=datetime.UTC)],
        ],
        ids=["naive first", "aware first", "after a null", "date and aware", "times"],
    )
    def test_a_list_of_both_kinds_is_refused(self, con: _duckdb.Connection, values: list[object]) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="with and without a time zone"):
            con.execute("SELECT $1", [values])

    def test_a_list_of_one_kind_binds(self, con: _duckdb.Connection) -> None:
        assert roundtrip(con, [NAIVE, NAIVE]) == [NAIVE, NAIVE]
        assert roundtrip(con, [AWARE, None]) == [AWARE, None]


class TestNestedZones:
    """A value inside a list, struct or map needs no time zone it does not name either."""

    def test_a_struct_keeps_each_field_as_it_is(self, con: _duckdb.Connection) -> None:
        assert roundtrip(con, {"a": NAIVE, "b": AWARE}) == {"a": NAIVE, "b": AWARE}

    def test_a_field_named_by_the_empty_string_meets_its_siblings_by_name(self, con: _duckdb.Connection) -> None:
        # A struct is unnamed only when no field has a name, so these pair by name and the zones never meet.
        bound = con.execute("SELECT typeof($1)", [[{"": NAIVE, "y": 1}, {"a": AWARE, "y": 2}]]).fetch_all()
        assert bound == [('STRUCT("" TIMESTAMP, y BIGINT, a TIMESTAMP WITH TIME ZONE)[]',)]

    @pytest.mark.parametrize(
        "value", [[{"": NAIVE}, {"a": AWARE}], [{"a": NAIVE}, {"": AWARE}]], ids=["unnamed first", "named first"]
    )
    def test_an_unnamed_struct_meets_a_named_one_by_position(self, con: _duckdb.Connection, value: object) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="with and without a time zone"):
            con.execute("SELECT $1", [value])

    def test_a_datetime_whose_zone_gives_no_offset_is_naive(self, con: _duckdb.Connection) -> None:
        value = datetime.datetime(2020, 1, 1, 12, tzinfo=NoOffset())
        assert con.execute("SELECT typeof($1), $1", [value]).fetch_all() == [("TIMESTAMP", NAIVE)]

    @pytest.mark.parametrize("keys", [["a", "b"], [1, 2], None], ids=["a struct", "a map", "a list"])
    def test_a_value_its_own_tzinfo_empties_binds_as_given(
        self, con: _duckdb.Connection, keys: list[object] | None
    ) -> None:
        # Only the list or dict holds its values, so emptying it while one converts would cut the rest or free them.
        stamps = [datetime.datetime(2020 + i, 1, 1, tzinfo=datetime.UTC) for i in range(2)]
        value: object
        expected: object
        if keys is None:
            elements: list[datetime.datetime] = []
            elements.extend(stamp.replace(tzinfo=Emptying(elements)) for stamp in stamps)
            value, expected = elements, stamps
        else:
            entries: dict[object, datetime.datetime] = {}
            entries.update(
                {key: stamp.replace(tzinfo=Emptying(entries)) for key, stamp in zip(keys, stamps, strict=True)}
            )
            value, expected = entries, dict(zip(keys, stamps, strict=True))
        assert con.execute("SELECT $1", [value]).fetch_all() == [(expected,)]

    def test_a_tzinfo_that_empties_the_parameters_dict_binds_them_as_given(self) -> None:
        # In its own interpreter under the debug allocator, since binding freed entries is a crash, not an error.
        probe = (
            "import datetime\n"
            "from duckdb import _duckdb\n"
            "class Emptying(datetime.tzinfo):\n"
            "    def utcoffset(self, moment):\n"
            "        params.clear()\n"
            "        return datetime.timedelta(0)\n"
            "    def dst(self, moment):\n"
            "        return None\n"
            "con = _duckdb.Database(':memory:').connect()\n"
            "params = {}\n"
            "params['a'] = datetime.datetime(2020, 1, 1, 12, tzinfo=Emptying())\n"
            "params['b'] = 7\n"
            "out = con.execute('SELECT $a, $b', params).fetch_all()\n"
            "print(out == [(datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.UTC), 7)])\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONMALLOC": "debug"},
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "True", (done.stdout, done.stderr)


class TestTemporalScalarEdges:
    @pytest.mark.parametrize("days", [106_751_992, -106_751_992])
    def test_a_pandas_timedelta_past_int64_microseconds_keeps_its_days(
        self, con: _duckdb.Connection, days: int
    ) -> None:
        value = pd.Timedelta(days * 86_400, unit="s")
        assert con.execute("SELECT date_part('day', $1)", [value]).fetch_all() == [(days,)]

    def test_a_datetime_or_timedelta_with_look_alike_pandas_attributes_binds_as_itself(
        self, con: _duckdb.Connection
    ) -> None:
        class Stamp(datetime.datetime):
            asm8 = 5

        class Span(datetime.timedelta):
            nanoseconds = "x"

        values = [Stamp(2020, 1, 1), Span(days=-1)]
        assert con.execute("SELECT $1, $2", values).fetch_all() == [
            (NAIVE.replace(hour=0), datetime.timedelta(days=-1))
        ]

    def test_nanoseconds_past_their_span_are_refused_for_the_nanoseconds(self, con: _duckdb.Connection) -> None:
        value = np.array(2**62 + 1, dtype="datetime64[1500ns]")[()]
        with pytest.raises(exceptions.InvalidInputError, match="has nanoseconds, and a query parameter holds"):
            con.execute("SELECT $1", [value])

    @pytest.mark.parametrize(
        ("count", "dtype"),
        [(2**40, "datetime64[2147483647s]"), (2**62, "datetime64[W]"), (2**40, "timedelta64[2147483647s]")],
        ids=["stepped", "weeks", "a stepped duration"],
    )
    def test_a_value_numpy_cannot_print_is_named_by_its_count(
        self, con: _duckdb.Connection, count: int, dtype: str
    ) -> None:
        # numpy scales these past int64 before printing, which wraps to a wrong value on some platforms.
        value = np.array(count, dtype=dtype)[()]
        message = f"the value {count} of {value.dtype.str} is beyond the range of its engine type"
        with pytest.raises(exceptions.InvalidInputError, match=re.escape(message)):
            con.execute("SELECT $1", [value])

    def test_a_tzinfo_that_raises_surfaces_its_error_and_the_connection_survives(self, con: _duckdb.Connection) -> None:
        with pytest.raises(ValueError, match="no offset today"):
            con.execute("SELECT $1", [datetime.datetime(2020, 1, 1, tzinfo=Failing(ValueError, "no offset today"))])
        assert con.execute("SELECT 1").fetch_all() == [(1,)]


class TestTypelessParts:
    """A NULL or an empty list has no type of its own, so it takes the one its siblings have at the same place."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([[], [NAIVE]], "TIMESTAMP[][]"),
            ([None, [NAIVE]], "TIMESTAMP[][]"),
            ([[None], ["x"]], "VARCHAR[][]"),
            ([[[]], [[NAIVE]]], "TIMESTAMP[][][]"),
            ([{"a": None}, {"a": "x"}], "STRUCT(a VARCHAR)[]"),
            ([None, {"a": None}, {"a": [None]}, {"a": [NAIVE]}], "STRUCT(a TIMESTAMP[])[]"),
            ({1: [], 2: [NAIVE]}, "MAP(BIGINT, TIMESTAMP[])"),
            ([[1], [None], [1.5]], "DOUBLE[][]"),
            ([{}, None], "STRUCT[]"),
            ([{"a": {}, "b": datetime.date(2020, 1, 1)}, {"a": None, "b": None}], "STRUCT(a STRUCT, b DATE)[]"),
            ({1: None, 2: "x"}, "MAP(BIGINT, VARCHAR)"),
            ([{"": 1, "x": "a"}, {"": 2, "x": None}], 'STRUCT("" BIGINT, x VARCHAR)[]'),
        ],
        ids=[
            "empty list",
            "null",
            "null element",
            "deeper",
            "struct field",
            "every depth",
            "map values",
            "widened",
            "beside an empty struct",
            "an empty struct field",
            "a null map value",
            "beside a field named by the empty string",
        ],
    )
    def test_a_missing_part_takes_its_siblings_type(
        self, con: _duckdb.Connection, value: object, expected: str
    ) -> None:
        assert con.execute("SELECT typeof($1), $1", [value]).fetch_all() == [(expected, value)]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([{"": 1}, {"": 2}], "[(1,), (2,)]"),
            ([None, {"": datetime.date(2020, 1, 1)}], "[NULL, (2020-01-01,)]"),
        ],
        ids=["two unnamed", "a null struct"],
    )
    def test_unnamed_fields_meet_by_position(self, con: _duckdb.Connection, value: object, expected: str) -> None:
        # A dict whose one key is empty makes the struct unnamed, a TUPLE, so the engine pairs its fields by position.
        assert con.execute("SELECT $1::VARCHAR", [value]).fetch_all() == [(expected,)]

    @pytest.mark.parametrize(
        "value",
        [
            [{"": "x"}, {"": None}],
            [{"": "x"}, {"a": None}],
            [{"a": "x"}, {"": None}],
            [{"a": None}, {"A": NAIVE}],
        ],
        ids=["a null field", "a named sibling", "an unnamed sibling", "a case twin"],
    )
    def test_a_part_only_an_exact_name_sibling_types_is_refused_untyped(
        self, con: _duckdb.Connection, value: object
    ) -> None:
        # A sibling types a struct field only by its exact name: pairing by position or across case is the engine's
        # own folding, and guessing it could put a type in the wrong field. The stand-in reaches the engine, which
        # refuses the combination itself.
        with pytest.raises(exceptions.NotSupportedError, match="Cannot combine types"):
            con.execute("SELECT $1::VARCHAR", [value])

    def test_a_named_sibling_after_an_unnamed_one_meets_it_by_name(self, con: _duckdb.Connection) -> None:
        # The unnamed struct takes its named neighbour's name, which the third then meets.
        value = [{"a": "x"}, {"": "y"}, {"a": None}]
        bound = con.execute("SELECT typeof($1), $1", [value]).fetch_all()
        assert bound == [("STRUCT(a VARCHAR)[]", [{"a": "x"}, {"a": "y"}, {"a": None}])]


class TestStandInsTakeTheStatementsType:
    """A place no sibling types takes the type the binder expects for the parameter.

    The expectation is only a hint: what it cannot type cleanly goes to the engine as the stand-in, and only a
    stand-in at the statement's first free position, where nothing could type it, is refused.
    """

    @pytest.fixture
    def typed(self) -> _duckdb.Connection:
        connection = _duckdb.Database(":memory:").connect()
        connection.execute("CREATE TABLE places (i INTEGER, l VARCHAR[])").fetch_all()
        connection.execute("INSERT INTO places VALUES (2, ['x'])").fetch_all()
        return connection

    def test_a_free_parameter_earlier_does_not_shadow_a_typed_one(self, typed: _duckdb.Connection) -> None:
        # The binder reports UNKNOWN for every parameter after one it cannot resolve, so the fill must not trust
        # that silence: the stand-in goes to the engine, whose own cast takes it.
        typed.execute("INSERT INTO places (i, l) VALUES ($1 + 1, $2)", [1, []]).fetch_all()
        assert typed.execute("SELECT l FROM places WHERE i = 2 ORDER BY ALL").fetch_all() == [([],), (["x"],)]
        assert typed.execute("SELECT $1 + 1, $2::VARCHAR[]", [1, [None]]).fetch_all() == [(2, [None])]
        assert typed.execute("SELECT i FROM places WHERE i + $1 > 2 AND l = $2", [1, ["x"]]).fetch_all() == [(2,)]

    @pytest.mark.parametrize(
        ("column", "value", "back"),
        [
            ("m", {"k": None}, {"k": None}),
            ("s", {"A": None, "b": 1}, {"a": None, "b": 1}),
            ("v", [None], "[NULL]"),
            ("u", [None], [None]),
        ],
        ids=["a dict for a map", "a case-twin field", "a list for text", "a list for a union"],
    )
    def test_an_expectation_of_another_form_is_left_to_the_engines_cast(
        self, column: str, value: object, back: object
    ) -> None:
        # The fill pairs forms and exact names only; whatever it cannot fill binds as the stand-in, which the
        # engine casts exactly as it casts the same SQL literal.
        db = _duckdb.Database(":memory:").connect()
        db.execute(
            "CREATE TABLE shapes (m MAP(VARCHAR, VARCHAR), s STRUCT(a VARCHAR, b INTEGER), v VARCHAR, "
            "u UNION(ai INTEGER[], b VARCHAR))"
        ).fetch_all()
        db.execute(f"INSERT INTO shapes ({column}) VALUES ($1)", [value]).fetch_all()
        assert db.execute(f"SELECT {column} FROM shapes").fetch_all() == [(back,)]

    def test_an_expectation_holding_any_is_left_to_the_engine(self, con: _duckdb.Connection) -> None:
        # The binder expects ANY[] here; a NULL of ANY cannot be built, and the engine's own message says more.
        assert con.execute("SELECT array_resize($1, $2)", [[], 3]).fetch_all() == [([None, None, None],)]
        with pytest.raises(exceptions.ProgrammingError, match="No function matches"):
            con.execute("SELECT array_resize($1, $2)", [None, 3])

    def test_an_insert_types_each_stand_in_from_its_column(self) -> None:
        db = _duckdb.Database(":memory:").connect()
        db.execute(
            "CREATE TABLE stand_ins (l VARCHAR[], d VARCHAR[][], s STRUCT(a VARCHAR)[], m MAP(BIGINT, VARCHAR))"
        ).fetch_all()
        db.execute(
            "INSERT INTO stand_ins VALUES ($1, $2, $3, $4)", [[None], [[]], [{"a": None}], {1: None}]
        ).fetch_all()
        assert db.execute("SELECT * FROM stand_ins").fetch_all() == [([None], [[]], [{"a": None}], {1: None})]

    def test_a_function_argument_types_the_stand_in(self, con: _duckdb.Connection) -> None:
        assert con.execute("SELECT list_contains($1, 'x')", [[]]).fetch_all() == [(False,)]

    def test_a_sibling_in_the_statement_types_the_stand_in(self, con: _duckdb.Connection) -> None:
        assert con.execute("SELECT typeof([$1, ['x']])", [[None]]).fetch_all() == [("VARCHAR[][]",)]

    def test_a_named_parameter_fills_whatever_its_case(self, con: _duckdb.Connection) -> None:
        assert con.execute("SELECT $tags = ['x']", {"TAGS": []}).fetch_all() == [(False,)]

    @pytest.mark.parametrize(
        "value",
        [[], [None], [{"a": None}], {1: []}],
        ids=["an empty list", "a null element", "a null field", "an empty map value"],
    )
    def test_where_the_statement_expects_nothing_binding_refuses_with_a_cast(
        self, con: _duckdb.Connection, value: object
    ) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=r"cast it to say its type, like \$1::VARCHAR\[\]"):
            con.execute("SELECT $1", [value])

    def test_the_suggested_cast_fixes_the_refusal(self, con: _duckdb.Connection) -> None:
        assert con.execute("SELECT $1::VARCHAR[]", [[]]).fetch_all() == [([],)]
        assert con.execute("SELECT $1::VARCHAR[]", [[None]]).fetch_all() == [([None],)]

    def test_a_parameter_under_two_casts_is_free_so_a_stand_in_refuses(self, con: _duckdb.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="holds an untyped value"):
            con.execute("SELECT $1::VARCHAR[], $1::INTEGER[]", [[]])

    def test_a_bare_null_still_binds_where_nothing_is_expected(self, con: _duckdb.Connection) -> None:
        # A bare NULL means NULL whatever its type, so it needs no refusal, as every DBAPI driver binds one.
        assert con.execute("SELECT $1 IS NULL", [None]).fetch_all() == [(True,)]

    def test_a_bare_null_takes_an_expected_type(self, con: _duckdb.Connection) -> None:
        assert con.execute("SELECT typeof([$1, 'x'])", [None]).fetch_all() == [("VARCHAR[]",)]

    def test_a_known_value_is_never_cast_to_the_expected_type(self, typed: _duckdb.Connection) -> None:
        # Cast to the column's INTEGER, 2.5 would round to 2 and match; the comparison must run on its own DOUBLE.
        assert typed.execute("SELECT i FROM places WHERE i = $1", [2.5]).fetch_all() == []
        assert typed.execute("SELECT i FROM places WHERE i = $1", [2.0]).fetch_all() == [(2,)]

    def test_the_fill_leaves_a_known_field_beside_the_stand_in_alone(self) -> None:
        # The expected type reaches the fill only because of the None in field a; field b must keep its own DOUBLE.
        db = _duckdb.Database(":memory:").connect()
        db.execute("CREATE TABLE k (s STRUCT(a VARCHAR, b INTEGER))").fetch_all()
        db.execute("INSERT INTO k VALUES ({'a': 'z', 'b': 1})").fetch_all()
        bound = db.execute("SELECT typeof([$1, s]) FROM k", [{"a": None, "b": 2.5}]).fetch_all()
        assert bound == [("STRUCT(a VARCHAR, b DOUBLE)[]",)]


# A list nested a thousand deep, alone or beside a NULL at every depth, bound and read back as text on a thread with
# room for that depth of recursion.
DEEP = """
import threading
from duckdb import _duckdb

def bind():
    connection = _duckdb.Database(":memory:").connect()
    alone = paired = 1
    for _ in range(1000):
        alone, paired = [alone], [paired, None]
    for value, text in ((alone, "[" * 1000 + "1" + "]" * 1000), (paired, "[" * 1000 + "1" + ", NULL]" * 1000)):
        print(connection.execute("SELECT $1::VARCHAR", [value]).fetch_all() == [(text,)])

# The default stack: an enlarged one would hide growth in the stack a nesting level costs.
worker = threading.Thread(target=bind)
worker.start()
worker.join()
"""


def test_a_list_nested_a_thousand_deep_binds_within_seconds() -> None:
    # Far longer than a thousand depths need, and far shorter than work cubic in the depth takes.
    result = subprocess.run([sys.executable, "-c", DEEP], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]


# Times that tell the naive and the aware value apart in text.
TEXT_NAIVE = datetime.datetime(2020, 1, 1, 12)
TEXT_AWARE = datetime.datetime(2020, 6, 1, 8, tzinfo=datetime.UTC)

#: SQL cannot name a struct field by the empty string, so a literal names it this instead; the engine pairs named
#: fields by name alone, so its reading differs from the bound value's only in that name.
EMPTY_NAME = "empty_name_stand_in"


def sql_literal(value: object) -> str:
    """`value` written as SQL that the engine combines by the same rules as the bound parameter."""
    if value is None:
        return "NULL"
    if value is TEXT_NAIVE:
        return "TIMESTAMP '2020-01-01 12:00:00'"
    if value is TEXT_AWARE:
        return "TIMESTAMPTZ '2020-06-01 08:00:00+00'"
    if isinstance(value, int):
        return f"{value}::BIGINT"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(sql_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            return "MAP {" + ", ".join(f"{sql_literal(key)}: {sql_literal(item)}" for key, item in value.items()) + "}"
        # ROW makes the engine's unnamed struct, as a dict whose one key is empty does.
        if list(value) == [""]:
            return f"ROW({sql_literal(value[''])})"
        return "{" + ", ".join(f"'{name or EMPTY_NAME}': {sql_literal(item)}" for name, item in value.items()) + "}"
    raise TypeError(value)


def naive_count(value: object) -> int:
    if isinstance(value, (list, tuple)):
        return sum(naive_count(item) for item in value)
    if isinstance(value, dict):
        return sum(naive_count(key) + naive_count(item) for key, item in value.items())
    return int(value is TEXT_NAIVE)


def holds_untyped(value: object) -> bool:
    """Whether a None or an empty container anywhere leaves a place with no type of its own."""
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return not value or any(holds_untyped(item) for item in value)
    if isinstance(value, dict):
        return not value or any(holds_untyped(key) or holds_untyped(item) for key, item in value.items())
    return False


def disagreement(con: _duckdb.Connection, value: object) -> str | None:
    """How binding `value` differs from the engine's own reading of it written in SQL, or None if it does not."""
    literal = sql_literal(value)
    try:
        engine = con.execute(f"SELECT typeof(x), x::VARCHAR FROM (SELECT {literal} AS x)").fetch_all()[0]
    except exceptions.Error:
        engine = None
    try:
        bound = con.execute("SELECT typeof($1), $1::VARCHAR", [value]).fetch_all()[0]
        error = None
    except exceptions.Error as raised:
        bound, error = None, raised
    if engine is None:
        return None if error else f"{literal}: the engine refuses it, binding gave {bound}"
    if (
        bound is None
        and holds_untyped(value)
        and ("holds an untyped value" in str(error) or "Cannot combine" in str(error))
    ):
        # Only an exact-name sibling types a place a value leaves unsaid: where the engine's own folding across
        # names, cases or positions would be needed, binding refuses instead of guessing it.
        return None
    if len(re.findall(r"2020-01-01 12:00:00(?![+-]\d)", engine[1])) < naive_count(value):
        refused = isinstance(error, exceptions.InvalidInputError) and "time zone" in str(error)
        return (
            None if refused else f"{literal}: the engine gives a naive value a time zone, binding gave {bound or error}"
        )
    if bound is None:
        return f"{literal}: the engine reads {engine}, binding failed: {error}"
    # A NULL no sibling types is INTEGER when bound, the engine's own NULL type in SQL; an unnamed struct is a TUPLE.
    expected = (
        engine[0].replace('"NULL"', "INTEGER").replace("TUPLE(", "STRUCT(").replace(f"{EMPTY_NAME} ", '"" '),
        engine[1].replace(f"'{EMPTY_NAME}':", "'':"),
    )
    actual = (bound[0].replace("TUPLE(", "STRUCT("), bound[1])
    return None if actual == expected else f"{literal}: the engine reads {expected}, binding gave {actual}"


@pytest.fixture(scope="module")
def utc() -> _duckdb.Connection:
    connection = _duckdb.Database(":memory:").connect()
    connection.execute("SET TimeZone='UTC'").fetch_all()
    assert connection.execute("SELECT current_setting('TimeZone')").fetch_all() == [("UTC",)]
    return connection


class TestStructSiblingsAgreeWithTheEngine:
    """Every list of three from a few struct shapes binds as the same list written in SQL reads.

    Where the engine would give a naive value a time zone, binding refuses instead. The engine combines a list's
    elements left to right, renaming as it goes, and casts each to the result.
    """

    @pytest.mark.parametrize(
        "shapes",
        [
            [
                {"a": TEXT_NAIVE, "b": TEXT_AWARE},
                {"b": TEXT_NAIVE, "a": TEXT_AWARE},
                {"b": TEXT_AWARE, "a": TEXT_NAIVE},
                {"": TEXT_NAIVE, "x": TEXT_AWARE},
                {"": TEXT_AWARE, "x": TEXT_NAIVE},
                {"A": TEXT_NAIVE, "b": None},
                {"a": None, "b": None},
                {"a": TEXT_NAIVE},
                {"": TEXT_NAIVE},
            ],
            [
                {"a": [TEXT_NAIVE], "b": [TEXT_AWARE]},
                {"a": [], "b": [TEXT_AWARE]},
                {"a": [None], "b": []},
                {"b": [TEXT_NAIVE], "a": [TEXT_AWARE]},
                {"": [TEXT_NAIVE], "x": [TEXT_AWARE]},
                {"": [], "x": None},
                {"a": None},
                {"a": [TEXT_NAIVE], "b": [TEXT_AWARE], "c": [TEXT_NAIVE]},
            ],
            [
                {"a": [None, TEXT_NAIVE], "b": None},
                {"a": [None, TEXT_AWARE], "b": None},
                {"a": [TEXT_AWARE], "b": []},
                {"a": [TEXT_NAIVE], "b": [None, TEXT_AWARE]},
                {"a": None, "b": [None, TEXT_NAIVE]},
                {"a": [], "b": [None]},
                {"": [None, TEXT_NAIVE], "x": None},
                {"b": [TEXT_AWARE, None], "a": []},
            ],
            [
                {1: {"a": [None, TEXT_NAIVE], "b": None}},
                {1: {"a": [TEXT_AWARE], "b": []}},
                {2: {"a": None, "b": [None, TEXT_AWARE]}, 3: None},
                {1: None},
                {1: {"a": [], "b": [TEXT_NAIVE]}},
            ],
            [
                {(None, TEXT_NAIVE): 1},
                {(TEXT_AWARE,): 1},
                {(): 1},
                {(None,): 2},
            ],
        ],
        ids=["timestamp fields", "list fields", "lists beside a NULL", "map values", "map keys"],
    )
    def test_three_siblings_in_every_order(self, utc: _duckdb.Connection, shapes: list[dict[str, object]]) -> None:
        found = [disagreement(utc, list(triple)) for triple in itertools.product(shapes, repeat=3)]
        assert [line for line in found if line] == []

    def test_lists_of_siblings_combine_inside_out(self, utc: _duckdb.Connection) -> None:
        # Each inner list combines on its own first, so an unnamed struct there takes its inner neighbour's names.
        named = {"a": TEXT_NAIVE, "b": TEXT_AWARE}
        reversed_named = {"b": TEXT_AWARE, "a": TEXT_NAIVE}
        unnamed = {"": TEXT_NAIVE, "x": TEXT_AWARE}
        # A NULL typed by its neighbour here keeps that type when an outer sibling types what the neighbour leaves open.
        open_naive = {"a": TEXT_NAIVE, "b": None}
        open_aware = {"a": TEXT_AWARE, "b": [TEXT_AWARE]}
        inner = [
            [named],
            [unnamed, reversed_named],
            [reversed_named, unnamed],
            [unnamed],
            [],
            [None, named],
            [None, open_naive],
            [{"a": None, "b": None}, open_naive],
            [open_aware],
        ]
        found = [disagreement(utc, list(pair)) for pair in itertools.product(inner, repeat=2)]
        assert [line for line in found if line] == []


class TestATypedNullKeepsItsType:
    """A NULL whose type its own neighbours fixed keeps it when a sibling further out types what they leave open.

    Retyping it would cast the neighbour beside it, so a naive value mixed with an aware one would pass unrefused.
    """

    @pytest.mark.parametrize(
        "value",
        [
            [{"a": [TEXT_AWARE], "b": 1}, {"a": [None, TEXT_NAIVE], "b": None}],
            [[{"a": TEXT_AWARE, "b": 1}], [None, {"a": TEXT_NAIVE, "b": None}]],
            [[{"a": TEXT_AWARE, "b": 1}], [{"a": None, "b": None}, {"a": TEXT_NAIVE, "b": None}]],
            [{1: {"a": [TEXT_AWARE], "b": 1}}, {1: {"a": [None, TEXT_NAIVE], "b": None}}],
        ],
        ids=["beside an open field", "a NULL element", "a NULL field", "in a map"],
    )
    @pytest.mark.parametrize("entry", ["bound", "written in", "lit", "returned", "returned in a union"])
    def test_its_naive_neighbour_is_refused_beside_an_aware_value(self, entry: str, value: object) -> None:
        # UTC, where an implementation comparing instants would wrongly accept; another zone adds no signal.
        con = duckdb.frame.connect()
        con._execute("SET TimeZone = 'UTC'").drain()
        assert sql("SELECT current_setting('TimeZone')").rows(con) == [("UTC",)]
        con.create_function("returned", lambda: value, [], "VARCHAR")
        con.create_function("chosen", lambda: value, [], "UNION(i INTEGER, s VARCHAR)")

        def run(query: str, parameters: list[object] | None = None) -> None:
            with con._execute(query, parameters) as result:
                result.fetch_all()

        attempts: dict[str, Callable[[], object]] = {
            "bound": lambda: run("SELECT $1", [value]),
            "written in": lambda: run(f"SELECT {render_literal(value)}"),
            "lit": lambda: sql("SELECT 1").select(lit(value)).rows(con),
            "returned": lambda: run("SELECT returned()"),
            "returned in a union": lambda: run("SELECT chosen()"),
        }
        with pytest.raises(exceptions.InvalidInputError, match="time zone"):
            attempts[entry]()


class TestKeysDifferingOnlyInCase:
    """A dict whose keys differ only in case binds as a STRUCT holding both, as the engine builds it."""

    @pytest.mark.parametrize(
        "value",
        [{"a": 1, "A": 2}, [{"a": 1, "A": 2}], [{"a": 1, "A": 2}, {"a": 3, "A": 4}], {"s": {"x": 1, "X": 2}}],
        ids=["alone", "in a list", "two alike in a list", "nested"],
    )
    def test_both_keys_bind(self, value: object) -> None:
        con = duckdb.frame.connect()
        with con._execute("SELECT $1", [value]) as result:
            assert result.fetch_all() == [(value,)]

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="the engine compares struct field names ignoring case, so it takes the two dicts' types as one and puts "
        "the second dict's values in the first one's order",
    )
    def test_dicts_ordering_them_differently_keep_their_values(self) -> None:
        value = [{"a": 1, "A": 2}, {"A": 3, "a": 4}]
        con = duckdb.frame.connect()
        try:
            with con._execute("SELECT $1", [value]) as result:
                back = result.fetch_all()
        except exceptions.Error:
            return
        assert back == [(value,)]


class TestEveryEntryPointAgrees:
    """A value means one thing bound, written in, collected by a frame, returned by a UDF and held by an object cell."""

    @staticmethod
    def through_each(value: object, declared: str) -> list[Callable[[], list[tuple[object, ...]]]]:
        con = duckdb.frame.connect()
        con._execute("SET TimeZone = 'UTC'").drain()
        assert sql("SELECT current_setting('TimeZone')").rows(con) == [("UTC",)]
        con.create_function("returned", lambda: value, [], declared)
        con.register("cells", pd.DataFrame({"v": pd.Series([value], dtype=object)}))

        def run(query: str, parameters: list[object] | None = None) -> list[tuple[object, ...]]:
            with con._execute(query, parameters) as result:
                return result.fetch_all()

        return [
            lambda: run("SELECT typeof($1), $1::VARCHAR", [value]),
            lambda: run(f"SELECT typeof(v), v::VARCHAR FROM (SELECT {render_literal(value)} AS v)"),
            lambda: sql("SELECT 1").select(fn("typeof", lit(value)), lit(value).cast("VARCHAR")).rows(con),
            lambda: run("SELECT typeof(returned()), returned()::VARCHAR"),
            lambda: run("SELECT typeof(v), v::VARCHAR FROM cells"),
        ]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime.datetime(2020, 1, 2, 3, 4, 5, 6), ("TIMESTAMP", "2020-01-02 03:04:05.000006")),
            (
                datetime.datetime(2020, 1, 2, 3, 4, 5, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
                ("TIMESTAMP WITH TIME ZONE", "2020-01-02 01:04:05.000006+00"),
            ),
            (datetime.datetime(2020, 1, 2, 3, 4, 5, 6, tzinfo=NoOffset()), ("TIMESTAMP", "2020-01-02 03:04:05.000006")),
            (datetime.date(1, 1, 1), ("DATE", "0001-01-01")),
            (
                datetime.time(3, 4, 5, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=-3))),
                ("TIME WITH TIME ZONE", "03:04:05.000006-03"),
            ),
            (datetime.timedelta(days=-3, microseconds=7), ("INTERVAL", "-3 days 00:00:00.000007")),
            (
                np.datetime64("2020-01-02T03:04:05.123", "ms"),
                ("TIMESTAMP_MS", "2020-01-02 03:04:05.123", "TIMESTAMP"),
            ),
            (np.datetime64("2020-03", "M"), ("DATE", "2020-03-01")),
            (np.zeros(1, dtype="datetime64[0M]")[0], ("DATE", "1970-01-01")),
            (np.zeros(1, dtype="datetime64[0s]")[0], ("TIMESTAMP_S", "1970-01-01 00:00:00", "TIMESTAMP")),
            (np.array(2**62, dtype="datetime64[1500ns]")[()], ("TIMESTAMP", "221177-10-08 09:00:41.081856")),
            (np.timedelta64(-1500, "ns"), ("INTERVAL", "-00:00:00.000001")),
            (
                pd.Timestamp("2020-01-02 03:04:05.5", tz="Europe/Amsterdam").as_unit("ms"),
                ("TIMESTAMP WITH TIME ZONE", "2020-01-02 02:04:05.5+00"),
            ),
            (pd.Timedelta(-1500, "ns"), ("INTERVAL", "-1 day 23:59:59.999999")),
            (pd.Timedelta(106_751_992 * 86_400, unit="s"), ("INTERVAL", "106751992 days")),
        ],
        ids=[
            "datetime",
            "aware datetime",
            "a zone giving no offset",
            "first date",
            "aware time",
            "timedelta",
            "numpy ms",
            "numpy months",
            "zero step months",
            "zero step seconds",
            "stepped past nanoseconds",
            "numpy duration",
            "pandas zoned ms",
            "pandas duration",
            "pandas wide duration",
        ],
    )
    def test_a_value_reads_alike_through_each(self, value: object, expected: tuple[str, ...]) -> None:
        # An object cell is typed from a sample that proves nothing about the rows it skipped, so a coarse unit
        # reads in microseconds there; everywhere else the value keeps its own unit.
        type_text, shown, *rest = expected
        readers = self.through_each(value, type_text)
        for read in readers[:-1]:
            assert read() == [(type_text, shown)]
        assert readers[-1]() == [(rest[0] if rest else type_text, shown)]

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            (datetime.time(12, tzinfo=zoneinfo.ZoneInfo("Europe/Amsterdam")), "offset depends on a date"),
            (np.timedelta64(1, "M"), "no fixed length"),
            (np.datetime64(2**62, "s"), "beyond the range of its engine type"),
        ],
        ids=["zoned time", "months", "past seconds"],
    )
    def test_a_value_is_refused_alike_by_each(self, value: object, message: str) -> None:
        for read in self.through_each(value, "VARCHAR"):
            with pytest.raises(exceptions.InvalidInputError, match=message):
                read()
