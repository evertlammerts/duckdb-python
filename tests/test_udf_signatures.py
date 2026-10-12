"""Python functions keep their parameters' names, kinds and defaults; types come from parameters or annotations."""

from __future__ import annotations

import contextvars
import datetime
import decimal
import enum
import functools
import sys
import threading
import typing
import uuid
from typing import TYPE_CHECKING

import numpy as np
import pytest

import duckdb
from duckdb import exceptions
from duckdb.frame import _signature, col, fn

from ._support import Failing, refusal_of_a_list_containing_itself

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def con() -> duckdb.frame.Connection:
    return duckdb.frame.connect()


def rows(con: duckdb.frame.Connection, sql: str) -> list[tuple[object, ...]]:
    with con._execute(sql) as result:
        return result.fetch_all()


def registered(con: duckdb.frame.Connection, name: str) -> tuple[list[str], list[str], str | None, str]:
    """What the catalog lists for a function: its fixed parameters' names and types, the `*args` type, the return."""
    query = "SELECT parameters, parameter_types, varargs, return_type FROM duckdb_functions() WHERE function_name = $1"
    with con._execute(query, [name]) as result:
        ((names, types, varargs, returns),) = result.fetch_all()
    return names, types, varargs, returns


def type_text(con: duckdb.frame.Connection, sql_type: str) -> str:
    """A type as the engine itself writes it."""
    ((text,),) = rows(con, f"SELECT typeof(NULL::{sql_type})")
    return str(text)


def annotated(annotation: object) -> Callable[..., int]:
    """A function of one parameter `x` carrying the annotation as Python keeps it, not as text."""

    def f(x):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN202
        return 0

    f.__annotations__ = {"x": annotation}
    return f


class Point:
    pass


class Flag(enum.IntEnum):
    ON = 1


def declares(x: datetime.date, y: datetime.date) -> datetime.date:
    return y


class Declares:
    def method(self, x: datetime.date, y: datetime.date) -> datetime.date:
        return y

    def __call__(self, x: datetime.date, y: datetime.date) -> datetime.date:
        return y


@functools.wraps(declares)
def wrapping(*args: typing.Any, **kwargs: typing.Any) -> datetime.date:  # noqa: ANN401
    return declares(*args, **kwargs)


# A constructor defined where `Day` is a name, inherited by a class defined here, where it is not.
ELSEWHERE: dict[str, object] = {}
exec(
    compile(
        "from datetime import date as Day\nclass Base:\n    def __new__(cls, x: 'Day', y: 'Day') -> 'Day':\n"
        "        return y\n",
        "<elsewhere>",
        "exec",
        dont_inherit=True,
    ),
    ELSEWHERE,
)
Inheriting = type("Inheriting", (typing.cast("type", ELSEWHERE["Base"]),), {})


def constructing(cls: type, x: datetime.date, y: datetime.date) -> datetime.date:
    return y


def taking(self: object, fixed: int, x: datetime.date, y: datetime.date) -> datetime.date:
    return y


Constructed = type("Meta", (type,), {"__call__": classmethod(constructing)})("Constructed", (), {})
Partly = type("Partly", (), {"__call__": functools.partialmethod(taking, 0)})

#: Callables whose annotations are those of a function they reach: Python evaluates them where that one is defined.
REACHING: dict[str, Callable[..., datetime.date]] = {
    "partial": functools.partial(declares),
    "bound method": Declares().method,
    "callable object": Declares(),
    "partial of a method": functools.partial(Declares().method),
    "wrapper": wrapping,
    "class with an inherited constructor": Inheriting,
    "class whose metaclass constructs it": Constructed,
    "object whose call is a partial method": Partly(),
}


class TestArgumentKinds:
    """Each Python parameter keeps its name, kind and default, so a SQL call reaches it as a Python call would."""

    def test_parameters_take_their_python_names(self, con: duckdb.frame.Connection) -> None:
        con.create_function("pair", lambda a, b: f"{a}/{b}", ["BIGINT", "VARCHAR"], "VARCHAR")
        assert rows(con, "SELECT pair(b := 'x', a => 1), pair(1, b := 'y')") == [("1/x", "1/y")]

    def test_a_positional_only_parameter_cannot_be_named(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda a, /: a, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT f(1)") == [(1,)]
        with pytest.raises(exceptions.ProgrammingError, match="No function matches"):
            rows(con, "SELECT f(a := 1)")

    def test_a_named_only_parameter_must_be_named(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda *, a: a, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT f(a := 2)") == [(2,)]
        with pytest.raises(exceptions.ProgrammingError, match="No function matches"):
            rows(con, "SELECT f(2)")

    def test_extra_positional_arguments_arrive_in_args(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda a, *rest: repr((a, rest)), returns="VARCHAR")
        assert rows(con, "SELECT f(1), f(1, 'x', 2.5)") == [("(1, ())", "(1, ('x', Decimal('2.5')))")]

    def test_extra_named_arguments_arrive_in_kwargs_as_the_caller_spelled_them(
        self, con: duckdb.frame.Connection
    ) -> None:
        con.create_function("f", lambda a, **extra: repr((a, extra)), returns="VARCHAR")
        assert rows(con, "SELECT f(1, Zed := 'z', b => 2)") == [("(1, {'Zed': 'z', 'b': 2})",)]

    def test_every_kind_together(self, con: duckdb.frame.Connection) -> None:
        def label(value, /, prefix="id", *parts, sep="-", **extra):  # type: ignore[no-untyped-def]  # noqa: ANN001, ANN002, ANN003, ANN202
            tags = "".join(f"[{k}={v}]" for k, v in extra.items())
            return sep.join([prefix, str(value), *map(str, parts)]) + tags

        con.create_function("label", label, returns="VARCHAR")
        got = rows(con, "SELECT label(7), label(7, 'x', 1, 2, sep := '/', loud := true)")
        assert got == [("id-7", "x/7/1/2[loud=True]")]

    def test_named_arguments_take_their_values_per_row(self, con: duckdb.frame.Connection) -> None:
        con.create_function(
            "f", lambda a, *, b, **extra: a + b + sum(extra.values()), ["BIGINT", "BIGINT", "BIGINT"], "BIGINT"
        )
        assert rows(con, "SELECT f(i, b := i * 10, c := i * 100) FROM range(3) t(i)") == [(0,), (111,), (222,)]

    def test_calls_differing_only_in_a_name_stay_apart(self, con: duckdb.frame.Connection) -> None:
        con.create_function("names", lambda **kw: ",".join(kw), returns="VARCHAR")
        assert rows(con, "SELECT names(k := i), names(j := i) FROM range(2) t(i)") == [("k", "j"), ("k", "j")]

    def test_named_arguments_match_ignoring_case(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda *, Value: Value, ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT f(value := 1), f(VALUE := 2)") == [(1, 2)]

    def test_a_positional_only_name_is_free_for_kwargs(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda a, /, **extra: repr((a, extra)), returns="VARCHAR")
        assert rows(con, "SELECT f(1, a := 2)") == [("(1, {'a': 2})",)]

    def test_names_equal_ignoring_case_are_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="Duplicate parameter name"):
            con.create_function("f", lambda a, A: 0, ["BIGINT", "BIGINT"], "BIGINT")

    @pytest.mark.parametrize("call", ["f()", "f(1, 2, 3)", "f(1, nope := 2)", "f(b := 1, 2)", "f(1, a := 2)"])
    def test_a_call_the_signature_does_not_fit_is_refused(self, con: duckdb.frame.Connection, call: str) -> None:
        con.create_function("f", lambda a, b=0: a + b, ["BIGINT", "BIGINT"], "BIGINT")
        with pytest.raises(exceptions.ProgrammingError, match="Binder Error"):
            rows(con, f"SELECT {call}")

    def test_a_partial_takes_the_parameters_it_left(self, con: duckdb.frame.Connection) -> None:
        con.create_function("minus", functools.partial(lambda a, b: a - b, 10), ["BIGINT"], "BIGINT")
        assert rows(con, "SELECT minus(3), minus(b := 4)") == [(7, 6)]

    def test_registering_with_other_types_adds_a_version(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda x: "one", ["BIGINT"], "VARCHAR")
        con.create_function("f", lambda x, y: "two", ["BIGINT", "BIGINT"], "VARCHAR")
        assert rows(con, "SELECT f(1), f(1, 2)") == [("one", "two")]

    def test_the_same_types_in_other_kinds_add_a_version(self, con: duckdb.frame.Connection) -> None:
        con.create_function("g", lambda x: "standard", ["BIGINT"], "VARCHAR")
        con.create_function("g", lambda *, x: "named-only", ["BIGINT"], "VARCHAR")
        assert rows(con, "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'g'") == [(2,)]
        assert rows(con, "SELECT g(1)") == [("standard",)]
        with pytest.raises(exceptions.ProgrammingError, match="Could not choose a best candidate"):
            rows(con, "SELECT g(x := 1)")

    def test_the_version_the_arguments_fit_most_cheaply_runs(self, con: duckdb.frame.Connection) -> None:
        con.create_function("h", lambda x: "bigint", ["BIGINT"], "VARCHAR")
        con.create_function("h", lambda x: "double", ["DOUBLE"], "VARCHAR")
        assert rows(con, "SELECT h(1), h(1.5), h(1::INTEGER)") == [("bigint", "double", "bigint")]

    def test_a_call_two_versions_fit_equally_well_is_refused(self, con: duckdb.frame.Connection) -> None:
        con.create_function("m", lambda x: "one", ["BIGINT"], "VARCHAR")
        con.create_function("m", lambda x, y=1: "two", ["BIGINT", "BIGINT"], "VARCHAR")
        assert rows(con, "SELECT m(1, 2)") == [("two",)]
        with pytest.raises(exceptions.ProgrammingError, match="Could not choose a best candidate"):
            rows(con, "SELECT m(1)")

    def test_a_non_callable_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(TypeError, match="callable"):
            con.create_function("f", 5, [], "BIGINT")  # type: ignore[arg-type]


class TestTypeOverrides:
    """`parameters` gives types: a list for every parameter in order, a dict for some by name."""

    def test_a_list_with_more_types_than_parameters_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=r"gives 2 types, but the function has 1 parameter: \*a"):
            con.create_function("f", lambda *a: 0, ["BIGINT", "BIGINT"], "BIGINT")

    def test_a_shorter_list_leaves_the_rest_to_python(self, con: duckdb.frame.Connection) -> None:
        con.create_function(
            "f",
            lambda x, shift=10, *rest, scale=2, **extra: (x + shift) * scale + len(rest) + len(extra),
            ["BIGINT"],
            "BIGINT",
        )
        assert registered(con, "f") == (["x"], ["BIGINT"], None, "BIGINT")
        assert rows(con, "SELECT f(1)") == [(22,)]
        for call in ("f(1, 2)", "f(1, shift := 2)", "f(1, scale := 3)"):
            with pytest.raises(exceptions.ProgrammingError, match="No function matches"):
                rows(con, f"SELECT {call}")

    def test_a_parameter_past_a_shorter_list_needs_a_default(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="parameter 'b' after them has no default"):
            con.create_function("f", lambda a, b: 0, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match="parameter 'c' after them has no default"):
            con.create_function("f", lambda a, *rest, c: 0, ["BIGINT"], "BIGINT")

    def test_a_default_left_to_python_is_the_functions_own_object(self, con: duckdb.frame.Connection) -> None:
        # Unlike a default SQL sees, which arrives as a fresh value, this one is shared by every call.
        box: list[int] = []
        captured = object()

        def f(x: int, seen: list[int] = box, held: object = captured) -> int:
            seen.append(x)
            return int(held is captured)

        con.create_function("f", f, ["BIGINT"], stability="volatile")
        assert rows(con, "SELECT f(i) FROM range(3) t(i)") == [(1,), (1,), (1,)]
        assert box == [0, 1, 2]

    def test_a_dict_overrides_only_the_parameters_it_names(self, con: duckdb.frame.Connection) -> None:
        def f(a: int, b: typing.Any) -> str:  # noqa: ANN401
            return repr((a, b))

        con.create_function("f", f, {"b": "VARCHAR"})
        assert registered(con, "f")[1] == ["BIGINT", "VARCHAR"]
        assert rows(con, "SELECT f(1, 'x')") == [("(1, 'x')",)]

    def test_a_dict_key_naming_no_parameter_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=r"'c'.*: a, b"):
            con.create_function("f", lambda a, b: 0, {"c": "BIGINT"}, "BIGINT")
        # A dict names a parameter as Python does, without the kind's mark.
        with pytest.raises(exceptions.InvalidInputError, match=r"'\*rest'.*: a, rest, extra$"):
            con.create_function("f", lambda a, *rest, **extra: 0, {"*rest": "BIGINT"}, "BIGINT")

    def test_a_dict_names_args_and_kwargs_by_their_python_names(self, con: duckdb.frame.Connection) -> None:
        con.create_function(
            "f", lambda *rest, **extra: repr((rest, extra)), {"rest": "BIGINT", "extra": "DOUBLE"}, "VARCHAR"
        )
        assert rows(con, "SELECT f('7', k := 8)") == [("((7,), {'k': 8.0})",)]

    @pytest.mark.parametrize("types", [["*BIGINT", "BIGINT"], ["**BIGINT", "*BIGINT"], ["BIGINT", "**BIGINT"]])
    def test_a_prefix_must_agree_with_the_parameters_kind(self, con: duckdb.frame.Connection, types: list[str]) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="prefix"):
            con.create_function("f", lambda a, *rest: 0, types, "BIGINT")

    def test_a_matching_prefix_is_accepted(self, con: duckdb.frame.Connection) -> None:
        con.create_function(
            "f", lambda a, *rest, **extra: len(rest) + len(extra), ["BIGINT", "*ANY", "**ANY"], "BIGINT"
        )
        assert rows(con, "SELECT f(1, 2, 3, k := 4)") == [(3,)]

    def test_typed_args_cast_each_extra(self, con: duckdb.frame.Connection) -> None:
        def f(*rest: int) -> str:
            return repr(rest)

        con.create_function("f", f)
        assert rows(con, "SELECT f('7', 8::TINYINT)") == [("(7, 8)",)]
        with pytest.raises(exceptions.ConversionError):
            rows(con, "SELECT f('x')")

    def test_typed_kwargs_cast_each_value(self, con: duckdb.frame.Connection) -> None:
        def f(**extra: float) -> str:
            return repr(extra)

        con.create_function("f", f)
        assert rows(con, "SELECT f(a := 1, b := '2.5')") == [("{'a': 1.0, 'b': 2.5}",)]

    def test_types_are_a_list_or_a_dict(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(TypeError, match="parameters"):
            con.create_function("f", lambda a: a, "BIGINT", "BIGINT")

    @pytest.mark.parametrize("parameters", [["BIGINT", None], {"b": None}, ["BIGINT", 5]])
    def test_a_type_is_text(self, con: duckdb.frame.Connection, parameters: object) -> None:
        with pytest.raises(TypeError, match="SQL type text, not"):
            con.create_function("f", lambda a, b: 0, parameters, "BIGINT")  # type: ignore[arg-type]


class TestNoSignature:
    """A callable Python has no signature for takes `parameters` as a list, with `*` and `**` marking the variadics."""

    def test_a_list_gives_positional_only_parameters(self, con: duckdb.frame.Connection) -> None:
        con.create_function("biggest", max, ["BIGINT", "BIGINT"], "BIGINT")
        assert rows(con, "SELECT biggest(1, 3)") == [(3,)]
        with pytest.raises(exceptions.ProgrammingError, match="No function matches"):
            rows(con, "SELECT biggest(arg0 := 1, arg1 := 3)")

    def test_a_star_prefix_gives_args(self, con: duckdb.frame.Connection) -> None:
        con.create_function("biggest", max, ["*BIGINT"], "BIGINT")
        assert rows(con, "SELECT biggest(1, 5, 3), biggest('2', 1)") == [(5, 2)]

    def test_a_double_star_prefix_gives_kwargs(self, con: duckdb.frame.Connection) -> None:
        con.create_function("pack", dict, ["**BIGINT"], "MAP(VARCHAR, BIGINT)")
        assert rows(con, "SELECT pack(a := 1, b := '2')") == [({"a": 1, "b": 2},)]

    @pytest.mark.parametrize(
        "types",
        [["*BIGINT", "BIGINT"], ["**ANY", "*ANY"], ["*BIGINT", "*BIGINT"], ["**ANY", "**ANY"], ["**ANY", "BIGINT"]],
    )
    def test_prefixes_out_of_order_are_refused(self, con: duckdb.frame.Connection, types: list[str]) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=r"\*"):
            con.create_function("f", max, types, "BIGINT")

    def test_without_a_list_it_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="no signature"):
            con.create_function("f", max, returns="BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match="no signature"):
            con.create_function("f", max, {"a": "BIGINT"}, "BIGINT")

    def test_without_returns_it_is_refused(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="returns"):
            con.create_function("f", max, ["BIGINT"])


#: Each annotation and the type it names, as SQL text.
ANNOTATIONS = [
    (bool, "BOOLEAN"),
    (int, "BIGINT"),
    (float, "DOUBLE"),
    (str, "VARCHAR"),
    (bytes, "BLOB"),
    (datetime.datetime, "TIMESTAMP"),
    (datetime.date, "DATE"),
    (datetime.time, "TIME"),
    (datetime.timedelta, "INTERVAL"),
    (uuid.UUID, "UUID"),
    (list[int], "BIGINT[]"),
    (typing.List[str], "VARCHAR[]"),  # noqa: UP006
    (tuple[float, ...], "DOUBLE[]"),
    (dict[str, int], "MAP(VARCHAR, BIGINT)"),
    (int | None, "BIGINT"),
    (typing.Optional[str], "VARCHAR"),  # noqa: UP045
    (typing.Union[None, bytes], "BLOB"),  # noqa: UP007
    (list[list[int]], "BIGINT[][]"),
    (list[int | None], "BIGINT[]"),
    (dict[str, list[float | None]], "MAP(VARCHAR, DOUBLE[])"),
]

#: Annotations no type is read from: the reason is in the refusal.
UNREADABLE = [
    decimal.Decimal,
    Point,
    Flag,
    list,
    tuple,
    dict,
    int | str,
    list[typing.Any],
    tuple[int, str],
    None,
    typing.Literal[1],
]

#: One value of each class of the annotation table, with nothing in it a type would depend on beyond the class.
VALUE_OF: dict[type, object] = {
    bool: True,
    int: 1,
    float: 1.5,
    str: "x",
    bytes: b"x",
    datetime.datetime: datetime.datetime(2026, 1, 2, 3, 4),
    datetime.date: datetime.date(2026, 1, 2),
    datetime.time: datetime.time(3, 4),
    datetime.timedelta: datetime.timedelta(days=1),
    uuid.UUID: uuid.UUID(int=1),
}


class TestAnnotations:
    """Types left out of `parameters` and `returns` are read from the function's annotations."""

    def test_types_come_from_the_annotations(self, con: duckdb.frame.Connection) -> None:
        def f(a: int, b: str, *rest: float, c: bool = False, **extra: bytes) -> str:
            return ""

        con.create_function("f", f)
        assert registered(con, "f") == (["a", "b", "c"], ["BIGINT", "VARCHAR", "BOOLEAN"], "DOUBLE", "VARCHAR")

    def test_an_unannotated_parameter_is_any(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda x: 0, returns="BIGINT")
        assert registered(con, "f")[1] == ["ANY"]

    def test_typing_any_is_any(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", annotated(typing.Any), returns="BIGINT")
        assert registered(con, "f")[1] == ["ANY"]

    @pytest.mark.parametrize(("annotation", "sql_type"), ANNOTATIONS, ids=[t for _, t in ANNOTATIONS])
    def test_an_annotation_names_its_type(
        self, con: duckdb.frame.Connection, annotation: object, sql_type: str
    ) -> None:
        con.create_function("f", annotated(annotation), returns="BIGINT")
        assert registered(con, "f")[1] == [type_text(con, sql_type)]

    @pytest.mark.parametrize("cls", list(_signature.CLASS_TYPES), ids=lambda c: c.__name__)
    def test_a_class_names_the_type_its_values_get(self, con: duckdb.frame.Connection, cls: type) -> None:
        # A missing example means a class joined the annotation table without one here.
        value = VALUE_OF[cls]
        with con._execute("SELECT typeof($1)", [value]) as result:
            ((own,),) = result.fetch_all()
        assert own == type_text(con, _signature.CLASS_TYPES[cls])

    @pytest.mark.parametrize("annotation", UNREADABLE, ids=repr)
    def test_an_unreadable_annotation_is_refused(self, con: duckdb.frame.Connection, annotation: object) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=r"parameter 'x'.*parameters"):
            con.create_function("f", annotated(annotation), returns="BIGINT")

    def test_decimal_asks_for_its_width_and_scale(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match=r"DECIMAL\(width, scale\)"):
            con.create_function("f", annotated(decimal.Decimal), returns="BIGINT")

    def test_an_override_skips_an_unreadable_annotation(self, con: duckdb.frame.Connection) -> None:
        def f(x: decimal.Decimal) -> decimal.Decimal:
            return x

        con.create_function("f", f, {"x": "DECIMAL(10, 2)"}, "DECIMAL(10, 2)")
        assert rows(con, "SELECT f(1.25)") == [(decimal.Decimal("1.25"),)]

    def test_string_annotations_are_evaluated(self, con: duckdb.frame.Connection) -> None:
        def f(x: int) -> list[int]:
            return [x]

        # This module defers its annotations, so Python keeps them as text.
        assert f.__annotations__ == {"x": "int", "return": "list[int]"}
        con.create_function("f", f)
        assert registered(con, "f")[1:] == (["BIGINT"], None, "BIGINT[]")

    @pytest.mark.parametrize("name", list(REACHING))
    def test_annotations_are_evaluated_where_the_reached_function_is_defined(
        self, con: duckdb.frame.Connection, name: str
    ) -> None:
        con.create_function("f", REACHING[name])
        assert registered(con, "f")[1:] == (["DATE", "DATE"], None, "DATE")
        assert rows(con, "SELECT f(DATE '2026-01-01', DATE '2026-01-02')") == [(datetime.date(2026, 1, 2),)]

    def test_an_annotation_python_cannot_evaluate_leaves_every_annotation_unread(
        self, con: duckdb.frame.Connection
    ) -> None:
        # Python evaluates a function's annotations together, so one that fails leaves the others unread too.
        def f(x: Missing, y: int) -> int:  # type: ignore[name-defined]  # noqa: F821
            return y

        cause = r"Python could not evaluate the function's annotations \(NameError: name 'Missing'"
        with pytest.raises(exceptions.InvalidInputError, match=rf"annotation 'Missing' of parameter 'x': {cause}"):
            con.create_function("f", f)
        with pytest.raises(exceptions.InvalidInputError, match=rf"annotation 'int' of parameter 'y': {cause}"):
            con.create_function("f", f, {"x": "VARCHAR"})
        with pytest.raises(exceptions.InvalidInputError, match=rf"annotation 'int' of the return: {cause}"):
            con.create_function("f", f, {"x": "VARCHAR", "y": "BIGINT"})
        con.create_function("f", f, {"x": "VARCHAR", "y": "BIGINT"}, "BIGINT")
        assert registered(con, "f")[1:] == (["VARCHAR", "BIGINT"], None, "BIGINT")
        con.create_function("g", lambda x, y: y, {"x": "VARCHAR"}, "BIGINT")
        assert registered(con, "g")[1] == ["VARCHAR", "ANY"]

    def test_an_annotation_given_a_type_is_never_formatted(self, con: duckdb.frame.Connection) -> None:
        # Formatting runs the annotation's own code; a refusal that never comes must not run it.
        class Unprintable:
            def __repr__(self) -> str:
                message = "repr ran"
                raise RuntimeError(message)

        def f(x: int, y: int) -> int:
            return y

        f.__annotations__ = {"x": "Missing", "y": Unprintable(), "return": "int"}
        con.create_function("f", f, ["VARCHAR", "BIGINT"], "BIGINT")
        assert registered(con, "f")[1] == ["VARCHAR", "BIGINT"]
        with pytest.raises(exceptions.InvalidInputError, match="annotation of type Unprintable of parameter 'y'"):
            con.create_function("g", f, {"x": "VARCHAR"}, "BIGINT")

    @pytest.mark.parametrize(
        "failing",
        [
            "Missing.Type",
            "Missing[int]",
            "list[Missing]",
            "typing.Missing",
            "int if Missing else str",
            "1 / 0",
            "list[",
        ],
    )
    def test_every_way_an_annotation_fails_leaves_all_unread(self, con: duckdb.frame.Connection, failing: str) -> None:
        def f(x: int, y: int) -> int:
            return y

        f.__annotations__ = {"x": failing, "y": "int", "return": "int"}
        with pytest.raises(exceptions.InvalidInputError, match="annotation 'int' of parameter 'y'"):
            con.create_function("f", f, {"x": "VARCHAR"})
        con.create_function("f", f, {"x": "VARCHAR", "y": "BIGINT"}, "BIGINT")
        assert registered(con, "f")[1:] == (["VARCHAR", "BIGINT"], None, "BIGINT")

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="annotations are evaluated lazily from Python 3.14")
    def test_a_lazy_annotation_python_cannot_evaluate_leaves_every_annotation_unread(
        self, con: duckdb.frame.Connection
    ) -> None:
        # Compiled without this module's deferred annotations, so Python keeps them as code, not as text.
        namespace: dict[str, object] = {}
        source = "def f(x: Missing, y: int) -> list[int]:\n    return [y]\n"
        exec(compile(source, "<lazy>", "exec", dont_inherit=True), namespace)
        f = typing.cast("Callable[..., list[int]]", namespace["f"])
        with pytest.raises(exceptions.InvalidInputError, match=r"parameter 'x'.*Missing"):
            con.create_function("f", f)
        with pytest.raises(exceptions.InvalidInputError, match="annotation 'int' of parameter 'y'"):
            con.create_function("f", f, {"x": "VARCHAR"})
        con.create_function("f", f, {"x": "VARCHAR", "y": "BIGINT"}, "BIGINT[]")
        assert registered(con, "f")[1:] == (["VARCHAR", "BIGINT"], None, "BIGINT[]")

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="annotations are evaluated lazily from Python 3.14")
    def test_a_lazy_annotation_that_raises_leaves_no_signature(self, con: duckdb.frame.Connection) -> None:
        # Python 3.14 runs the annotation's code even for its text, so inspect itself cannot give the signature.
        namespace: dict[str, object] = {}
        source = "def f(x: 1 / 0, y: int) -> int:\n    return y\n"
        exec(compile(source, "<lazy>", "exec", dont_inherit=True), namespace)
        f = typing.cast("Callable[..., int]", namespace["f"])
        for parameters in (None, {"x": "VARCHAR"}):
            with pytest.raises(
                exceptions.InvalidInputError, match=r"cannot give the function's signature.*ZeroDivision"
            ):
                con.create_function("f", f, parameters)

    @pytest.mark.skipif(sys.version_info < (3, 12), reason="type parameters arrive with Python 3.12")
    def test_an_annotation_naming_a_type_parameter_is_refused(self, con: duckdb.frame.Connection) -> None:
        # The module's own T, an int, is not what the function's type parameter T means.
        namespace: dict[str, object] = {}
        source = "T = int\ndef f[T](y: 'T') -> int:\n    return 0\n"
        exec(compile(source, "<generic>", "exec", dont_inherit=True), namespace)
        f = typing.cast("Callable[..., int]", namespace["f"])
        with pytest.raises(exceptions.InvalidInputError, match="annotation 'T' of parameter 'y'"):
            con.create_function("f", f)

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="annotations are evaluated lazily from Python 3.14")
    def test_a_lazy_annotation_sees_what_it_closes_over(self, con: duckdb.frame.Connection) -> None:
        namespace: dict[str, object] = {}
        source = (
            "Alias = str\ndef make():\n    Alias = int\n    def f(y: Alias) -> int:\n        return 0\n    return f\n"
        )
        exec(compile(source, "<closure>", "exec", dont_inherit=True), namespace)
        f = typing.cast("Callable[[], Callable[..., int]]", namespace["make"])()
        con.create_function("f", f)
        assert registered(con, "f")[1] == ["BIGINT"]

    @pytest.mark.skipif(sys.version_info < (3, 14), reason="annotations are evaluated lazily from Python 3.14")
    @pytest.mark.parametrize("name", ["Alias", "format"])
    def test_a_lazy_annotation_keeps_its_class_scope(self, con: duckdb.frame.Connection, name: str) -> None:
        # The module's own binding of the name, a str, must not replace the class's, an int.
        namespace: dict[str, object] = {}
        source = f"{name} = str\nclass C:\n    {name} = int\n    def f(self, y: {name}) -> int:\n        return 0\n"
        exec(compile(source, "<class>", "exec", dont_inherit=True), namespace)
        method = typing.cast("type", namespace["C"])().f
        con.create_function("f", method)
        assert registered(con, "f")[1] == ["BIGINT"]

    def test_the_return_type_comes_from_the_annotation(self, con: duckdb.frame.Connection) -> None:
        def f(x: typing.Any) -> list[str]:  # noqa: ANN401
            return [str(x)]

        con.create_function("f", f)
        assert registered(con, "f")[3] == "VARCHAR[]"
        assert rows(con, "SELECT f(1)") == [(["1"],)]

    def test_returns_wins_over_the_annotation(self, con: duckdb.frame.Connection) -> None:
        def f(x: int) -> int:
            return x

        con.create_function("f", f, returns="VARCHAR")
        assert registered(con, "f")[3] == "VARCHAR"

    def test_a_return_needs_returns_or_an_annotation(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(exceptions.InvalidInputError, match="returns"):
            con.create_function("f", lambda x: x)

    def test_an_any_return_annotation_is_refused(self, con: duckdb.frame.Connection) -> None:
        def f(x: int) -> typing.Any:  # noqa: ANN401
            return x

        with pytest.raises(exceptions.InvalidInputError, match="an ANY return type is not supported yet"):
            con.create_function("f", f)

    def test_an_unreadable_return_annotation_is_refused(self, con: duckdb.frame.Connection) -> None:
        def f(x: int) -> decimal.Decimal:
            return decimal.Decimal(x)

        with pytest.raises(exceptions.InvalidInputError, match=r"return.*returns"):
            con.create_function("f", f)

    def test_a_zoned_argument_does_not_fit_a_datetime_annotation(self, con: duckdb.frame.Connection) -> None:
        def f(x: datetime.datetime) -> str:
            return str(x)

        con.create_function("f", f)
        assert rows(con, "SELECT f(TIMESTAMP '2026-01-02 03:04:05')") == [("2026-01-02 03:04:05",)]
        with pytest.raises(exceptions.ProgrammingError, match="No function matches"):
            rows(con, "SELECT f(TIMESTAMPTZ '2026-01-02 03:04:05+00')")


class TestDefaults:
    """A Python default is converted once, at registration, and arrives wherever a call leaves its parameter out."""

    def test_an_omitted_parameter_takes_its_default(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda a, b="x", *, c=2.5: repr((a, b, c)), returns="VARCHAR")
        assert rows(con, "SELECT f(1), f(1, 'y', c := 1)") == [("(1, 'x', 2.5)", "(1, 'y', 1)")]

    def test_a_default_takes_the_declared_type(self, con: duckdb.frame.Connection) -> None:
        def f(x: float = 1, y: list[int] = (1, 2)) -> str:  # type: ignore[assignment]
            return repr((x, y))

        con.create_function("f", f)
        assert rows(con, "SELECT f()") == [("(1.0, [1, 2])",)]

    def test_an_any_default_keeps_its_own_type(self, con: duckdb.frame.Connection) -> None:
        con.create_function("kind", lambda x=datetime.date(2026, 1, 2): type(x).__name__, returns="VARCHAR")
        assert rows(con, "SELECT kind(), kind(1)") == [("date", "int")]

    def test_a_list_default_is_fresh_on_every_call(self, con: duckdb.frame.Connection) -> None:
        def f(acc: list[int] = []) -> int:  # noqa: B006
            acc.append(1)
            return len(acc)

        con.create_function("f", f, stability="volatile")
        assert rows(con, "SELECT f() FROM range(3)") == [(1,), (1,), (1,)]

    def test_a_nested_default_is_fresh_on_every_call(self, con: duckdb.frame.Connection) -> None:
        def f(acc: list[list[int]] = [[]], tags: dict[str, list[int]] = {"k": []}) -> int:  # noqa: B006
            acc[0].append(1)
            tags["k"].append(1)
            return len(acc[0]) + len(tags["k"])

        con.create_function("f", f, stability="volatile")
        assert rows(con, "SELECT f() FROM range(3)") == [(2,), (2,), (2,)]

    def test_an_any_default_is_converted_as_a_query_parameter(self, con: duckdb.frame.Connection) -> None:
        finer = np.datetime64("2026-01-02T03:04:05.123456789", "ns")
        with pytest.raises(exceptions.InvalidInputError, match=r"default of parameter 'x'.*microseconds at most"):
            con.create_function("f", lambda x=finer: 0, returns="BIGINT")

    def test_a_default_whose_conversion_raises_names_its_parameter(self, con: duckdb.frame.Connection) -> None:
        when = datetime.datetime(2026, 1, 2, tzinfo=Failing(ValueError, "no offset"))
        for default in (when, "\ud800"):
            with pytest.raises(
                exceptions.InvalidInputError, match=r"^Invalid Input Error: the default of parameter 'x'"
            ):
                con.create_function("f", lambda x=default: 0, returns="BIGINT")
        with pytest.raises(exceptions.InvalidInputError, match=r"default of parameter 'x'.*ValueError: no offset"):
            con.create_function("f", lambda x=when: 0, ["TIMESTAMPTZ"], "BIGINT")

    def test_a_default_that_contains_itself_is_refused(self) -> None:
        printed = refusal_of_a_list_containing_itself("con.create_function('f', lambda v=x: 0, ['BIGINT[]'], 'BIGINT')")
        assert "the default of parameter 'v' is a value that cannot be converted" in printed
        assert "a list or dict that contains itself has no SQL value" in printed

    def test_a_text_default_for_a_zoned_type_is_read_where_it_is_registered(self, con: duckdb.frame.Connection) -> None:
        con.run("SET TimeZone = 'America/New_York'")
        con.create_function("f", lambda x="2026-01-01 00:00:00": str(x), ["TIMESTAMPTZ"], "VARCHAR")
        other = con.duplicate()
        other.run("SET TimeZone = 'UTC'")
        with other._execute("SELECT f(), f('2026-01-01 00:00:00')") as result:
            assert result.fetch_all() == [("2026-01-01 05:00:00+00:00", "2026-01-01 00:00:00+00:00")]

    def test_a_refused_registration_aborts_an_open_transaction(self, con: duckdb.frame.Connection) -> None:
        con.run("BEGIN")
        con.run("CREATE TABLE t AS SELECT 1 AS a")
        with pytest.raises(exceptions.InvalidInputError, match="default of parameter 'x'"):
            con.create_function("f", lambda x="five": 0, ["BIGINT"], "BIGINT")
        with pytest.raises(exceptions.TransactionError, match="Current transaction is aborted"):
            rows(con, "SELECT count(*) FROM t")
        con.run("ROLLBACK")
        assert rows(con, "SELECT 1") == [(1,)]

    def test_a_default_the_declared_type_cannot_hold_is_refused(self, con: duckdb.frame.Connection) -> None:
        def f(x: int = "five") -> int:  # type: ignore[assignment]
            return 0

        with pytest.raises(exceptions.InvalidInputError, match="default of parameter 'x'"):
            con.create_function("f", f)

    def test_a_default_that_is_no_sql_value_is_refused(self, con: duckdb.frame.Connection) -> None:
        missing = object()
        with pytest.raises(exceptions.InvalidInputError, match="default of parameter 'x'"):
            con.create_function("f", lambda x=missing: 0, returns="BIGINT")

    def test_a_none_default_under_default_null_handling_is_null(self, con: duckdb.frame.Connection) -> None:
        calls: list[tuple[object, object]] = []

        def f(a: int, b: int | None = None) -> int:
            calls.append((a, b))
            return 0

        con.create_function("f", f)
        assert rows(con, "SELECT f(1), f(1, 2)") == [(None, 0)]
        assert calls == [(1, 2)]

    def test_a_none_default_under_special_null_handling_arrives(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda a, b=None: repr(b), ["BIGINT", "BIGINT"], "VARCHAR", null_handling="special")
        assert rows(con, "SELECT f(1)") == [("None",)]

    def test_a_null_named_argument_is_null(self, con: duckdb.frame.Connection) -> None:
        con.create_function("f", lambda a, **extra: 0, ["BIGINT", "ANY"], "BIGINT")
        assert rows(con, "SELECT f(1, k := NULL), f(1, k := 2)") == [(None, 0)]


class TestNamedArguments:
    """`fn` passes keyword arguments to SQL by name: `fn("round", x, precision=1)` is `round(x, precision := 1)`."""

    def test_a_built_in_takes_a_named_argument(self, con: duckdb.frame.Connection) -> None:
        plan = duckdb.frame.sql("SELECT 2.567 AS x").select(fn("round", col("x"), precision=1).alias("r"))
        assert plan.on(con).rows() == [(decimal.Decimal("2.6"),)]

    def test_a_python_function_takes_named_arguments(self, con: duckdb.frame.Connection) -> None:
        con.create_function(
            "label", lambda value, /, *, sep="-", **extra: f"{value}{sep}{sorted(extra)}", returns="VARCHAR"
        )
        plan = duckdb.frame.sql("SELECT 7 AS v").select(fn("label", col("v"), sep="/", loud=True, b=2).alias("l"))
        assert plan.on(con).rows() == [("7/['b', 'loud']",)]

    def test_a_macro_takes_a_named_argument(self, con: duckdb.frame.Connection) -> None:
        con.create_macro("add", ["a", ("b", 10)], col("a") + col("b"))
        plan = duckdb.frame.sql("SELECT 1 AS x").select(
            fn("add", col("x")).alias("p"), fn("add", col("x"), b=2).alias("q")
        )
        assert plan.on(con).rows() == [(11, 3)]

    def test_an_argument_called_name_passes_by_name(self, con: duckdb.frame.Connection) -> None:
        con.create_function("greet", lambda *, name: f"hi {name}", ["VARCHAR"], "VARCHAR")
        assert duckdb.frame.sql("SELECT 1").select(fn("greet", name="ann")).on(con).rows() == [("hi ann",)]

    def test_a_keyword_is_a_name_like_any(self, con: duckdb.frame.Connection) -> None:
        con.create_function("pick", lambda *, default, Select: f"{default}{Select}", ["VARCHAR", "VARCHAR"], "VARCHAR")
        plan = duckdb.frame.sql("SELECT 1").select(fn("pick", default="a", select="b"))
        assert plan.on(con).rows() == [("ab",)]

    def test_a_window_function_takes_a_named_argument(self, con: duckdb.frame.Connection) -> None:
        plan = duckdb.frame.sql("SELECT i FROM range(4) t(i)").select(
            col("i"), fn("lead", col("i"), offset=2).over(order_by=col("i")).alias("ahead")
        )
        assert plan.on(con).rows() == [(0, 2), (1, 3), (2, None), (3, None)]

    def test_named_arguments_follow_the_positional_ones_in_order(self) -> None:
        assert repr(fn("f", col("x"), b="t", a=1)) == """<Expr f("x", "b" := 't', "a" := 1)>"""

    def test_a_named_argument_keeps_ignore_nulls_inside_the_call(self, con: duckdb.frame.Connection) -> None:
        values = duckdb.frame.sql("SELECT * FROM (VALUES (0, 1), (1, NULL), (2, 3)) t(i, x)")
        lead = fn("lead", col("x"), offset=1).ignore_nulls().over(order_by=col("i"))
        assert values.select(col("i"), lead).on(con).rows() == [(0, 3), (1, 3), (2, None)]

    def test_a_named_value_can_be_a_parameter(self, con: duckdb.frame.Connection) -> None:
        con.create_function("scale", lambda x, *, by: x * by, ["BIGINT", "BIGINT"], "BIGINT")
        plan = duckdb.frame.sql("SELECT 3 AS x").select(fn("scale", col("x"), by=duckdb.frame.param("p")))
        assert plan.on(con).rows(parameters={"p": 5}) == [(15,)]

    @pytest.mark.parametrize(
        ("named", "error"),
        [
            ({"nope": 1}, "No function matches"),
            ({"rest": 1, "REST": 2}, "Duplicate named argument"),
            ({"": 1}, "zero-length"),
        ],
    )
    def test_a_call_the_function_does_not_fit_is_refused_at_run_time(
        self, con: duckdb.frame.Connection, named: dict[str, object], error: str
    ) -> None:
        con.create_function("take", lambda x, *, k=0: x, ["BIGINT", "BIGINT"], "BIGINT")
        con.create_function("gather", lambda x, **rest: x, ["BIGINT", "ANY"], "BIGINT")
        function = "gather" if "rest" in named else "take"
        plan = duckdb.frame.sql("SELECT 1 AS x").select(fn(function, col("x"), **named))
        with pytest.raises(exceptions.ProgrammingError, match=error):
            plan.on(con).rows()


class TestWorkerThreads:
    def test_workers_share_the_functions_objects_but_not_the_callers_context(
        self, con: duckdb.frame.Connection
    ) -> None:
        con.run("SET threads = 4")
        con.run("CREATE TABLE big AS SELECT i FROM range(1000000) t(i)")
        captured: list[int] = []
        context = contextvars.ContextVar("context", default="unset")
        context.set("set by the caller")
        local = threading.local()
        local.value = "set by the caller"
        seen: set[tuple[int, bool, str, str]] = set()

        def f(x: int, held: list[int] = captured) -> int:
            seen.add((threading.get_ident(), held is captured, context.get(), getattr(local, "value", "unset")))
            return x

        con.create_function("f", f, ["BIGINT"])
        rows(con, "SELECT sum(f(i)) FROM big")
        caller = threading.get_ident()
        workers = {(same, value, held) for ident, same, value, held in seen if ident != caller}
        assert workers == {(True, "unset", "unset")}
        callers = {(same, value, held) for ident, same, value, held in seen if ident == caller}
        assert callers <= {(True, "set by the caller", "set by the caller")}
        assert all(same for _, same, _, _ in seen)
