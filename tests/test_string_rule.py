"""Every public argument that takes a string reads it in one way, the one its annotation names."""

from __future__ import annotations

import ast
import ctypes
import datetime
import decimal
import importlib
import inspect
import itertools
import re
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ForwardRef

import numpy as np
import pytest

import duckdb
from duckdb._expressions.expr import (
    CaseBuilder,
    ThenBuilder,
    _as_expression,
    _as_expressions,
    _as_field_key,
    _as_name,
    _as_names,
    _as_sql,
    _as_text,
    _lift,
    name_parts,
)
from duckdb._expressions.func_namespaces import DtExpr, JsonExpr, ListExpr, StrExpr
from duckdb.frame import col, first_value, lag, last_value, lead, lit, row_number, sql, star, typing, values, when
from duckdb.frame.connection import _as_macro_parameters, _as_parameters
from duckdb.frame.plan import (
    Column,
    GroupedFrame,
    _argument,
    _as_expr,
    _as_exprs,
    _as_heading,
    _as_named_parameters,
    _as_rows,
    _as_sample_method,
    _option,
    _option_value,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def con() -> duckdb.frame.Connection:
    connection = duckdb.frame.connect()
    connection.run("CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a', 10), (2, 'b', 20), (3, 'a', 30)) AS v(id, g, x)")
    return connection


class TestANameArgumentTakesABareStringAsOneName:
    def test_star_excludes_the_one_column(self, con: duckdb.frame.Connection) -> None:
        assert duckdb.frame.table("t").select(star(exclude="id")).columns(con) == ["g", "x"]

    def test_values_has_the_one_column(self, con: duckdb.frame.Connection) -> None:
        assert values([(1,)], "ab").columns(con) == ["ab"]

    def test_a_macro_has_the_one_parameter(self, con: duckdb.frame.Connection) -> None:
        con.create_macro("plus_one", "ab", col("ab") + 1)
        assert sql("SELECT plus_one(41)").rows(con) == [(42,)]

    def test_a_list_of_names_stays_a_list(self, con: duckdb.frame.Connection) -> None:
        assert duckdb.frame.table("t").select(star(exclude=["id", "g"])).columns(con) == ["x"]
        assert values([(1, 2)], ["ab", "cd"]).columns(con) == ["ab", "cd"]
        con.create_macro("add", ["ab", "cd"], col("ab") + col("cd"))
        assert sql("SELECT add(40, 2)").rows(con) == [(42,)]


class TestAPairIsAnyIterableButText:
    def test_a_column_pair_can_be_a_numpy_row_or_an_iterator(self, con: duckdb.frame.Connection) -> None:
        assert values([("1",)], np.array([["a", "INTEGER"]])).schema(con) == [("a", "INTEGER")]
        assert values([("1",)], [iter(["a", "INTEGER"])]).schema(con) == [("a", "INTEGER")]

    def test_a_column_pair_without_a_type_leaves_the_column_untyped(self, con: duckdb.frame.Connection) -> None:
        assert values([("1", "2")], [("a", None), ("b",)]).schema(con) == [("a", "VARCHAR"), ("b", "VARCHAR")]

    @pytest.mark.parametrize("item", [b"ab", ("a", "INTEGER", "x"), ()], ids=["bytes", "three", "empty"])
    def test_a_column_is_never_text_bytes_or_the_wrong_length(self, item: object) -> None:
        with pytest.raises(TypeError, match="a name or a"):
            values([(1,)], [item])  # type: ignore[list-item]

    def test_a_macro_parameter_pair_can_be_a_numpy_row_or_a_list(self, con: duckdb.frame.Connection) -> None:
        con.create_macro("plus", ["x", np.array(["by", 2], dtype=object), ["also", 3]], col("x") + col("by"))
        assert sql("SELECT plus(40)").rows(con) == [(42,)]

    @pytest.mark.parametrize("item", [b"ab", iter(["by", 2]), 7], ids=["bytes", "iterator", "int"])
    def test_a_macro_parameter_is_read_by_position(self, con: duckdb.frame.Connection, item: object) -> None:
        with pytest.raises(TypeError, match="a name or a"):
            con.create_macro("plus", ["x", item], col("x"))  # type: ignore[list-item]


class TestRenamesAreAMappingOrPairs:
    def test_star_renames_from_a_dict_a_list_or_a_generator_of_pairs(self, con: duckdb.frame.Connection) -> None:
        plan = duckdb.frame.table("t")
        assert plan.select(star(rename={"g": "group"})).columns(con) == ["id", "group", "x"]
        assert plan.select(star(rename=[("g", "group"), ["x", "amount"]])).columns(con) == ["id", "group", "amount"]
        assert plan.select(star(rename=((old, old.upper()) for old in ["id"]))).columns(con) == ["ID", "g", "x"]

    @pytest.mark.parametrize("renames", ["gh", ["gh"], [b"gh"], [("g",)]], ids=["text", "text pair", "bytes", "single"])
    def test_a_string_is_never_a_pair_of_names(self, renames: object) -> None:
        with pytest.raises(TypeError, match="rename"):
            star(rename=renames)  # type: ignore[arg-type]


class TestConnectWritesEverySettingAsItsText:
    def test_a_numpy_integer_or_a_decimal_is_a_setting_value(self) -> None:
        for threads in (np.int64(2), decimal.Decimal(2)):
            with duckdb.frame.connect(threads=threads) as connection:
                assert sql("SELECT current_setting('threads')").rows(connection) == [(2,)]


class TestAValueSequenceRefusesABareString:
    def test_values_refuses_a_string_as_its_rows(self) -> None:
        with pytest.raises(TypeError, match="rows"):
            values("ab", ["x"])

    def test_values_refuses_a_string_as_a_row(self) -> None:
        with pytest.raises(TypeError, match="row"):
            values(["ab"], ["x", "y"])

    @pytest.mark.parametrize("parameters", ["ab", b"ab"], ids=["str", "bytes"])
    def test_run_refuses_a_string_as_its_parameters(self, con: duckdb.frame.Connection, parameters: object) -> None:
        with pytest.raises(TypeError, match="parameters"):
            con.run("SELECT ?, ?", parameters)  # type: ignore[arg-type]

    def test_a_plan_refuses_a_string_as_its_parameters(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(TypeError, match="parameters"):
            sql("SELECT 1").rows(con, parameters="ab")  # type: ignore[arg-type]

    def test_a_row_can_be_any_iterable_but_text(self, con: duckdb.frame.Connection) -> None:
        assert values([(x for x in [1, 2])], ["a", "b"]).rows(con) == [(1, 2)]
        assert values([[1, 2], range(3, 5)], ["a", "b"]).rows(con) == [(1, 2), (3, 4)]
        # A ctypes array iterates through __getitem__ alone, which no Iterable check sees.
        assert values([(ctypes.c_int * 2)(1, 2)], ["a", "b"]).rows(con) == [(1, 2)]

    def test_run_still_binds_a_list_a_generator_and_a_mapping(self, con: duckdb.frame.Connection) -> None:
        con.run("CREATE TABLE r (a VARCHAR, b VARCHAR)")
        con.run("INSERT INTO r VALUES (?, ?)", ["ab", "cd"])
        con.run("INSERT INTO r VALUES (?, ?)", (v for v in ["ef", "gh"]))
        con.run("INSERT INTO r VALUES ($a, $b)", {"a": "ij", "b": "kl"})
        assert sql("SELECT * FROM r").rows(con) == [("ab", "cd"), ("ef", "gh"), ("ij", "kl")]


class TestAnExpressionSlotRefusesAString:
    def test_when_refuses_a_string(self) -> None:
        with pytest.raises(TypeError, match=r"col\(\)"):
            when("x > 1")  # type: ignore[arg-type]

    def test_a_later_when_refuses_a_string(self) -> None:
        with pytest.raises(TypeError, match=r"col\(\)"):
            when(col("x") > 1).then(1).when("g")  # type: ignore[arg-type]

    def test_where_refuses_a_string(self) -> None:
        with pytest.raises(TypeError, match=r"col\(\)"):
            col("x").sum().where("x > 10")  # type: ignore[arg-type]

    def test_where_inside_a_namespace_refuses_a_string(self) -> None:
        with pytest.raises(TypeError, match=r"col\(\)"):
            col("g").string_agg(",").str().where("x > 10")  # type: ignore[arg-type]

    def test_a_condition_refuses_a_bare_python_value(self) -> None:
        with pytest.raises(TypeError, match=r"lit\(\)"):
            when(True)  # type: ignore[arg-type]

    @pytest.mark.parametrize("keys", ["g", ["g"], (col("id"), "g")], ids=["single", "list", "after an expression"])
    @pytest.mark.parametrize("slot", ["partition_by", "order_by"])
    def test_over_refuses_a_string_key(self, slot: str, keys: object) -> None:
        with pytest.raises(TypeError, match=r"col\(\)"):
            row_number().over(**{slot: keys})

    @pytest.mark.parametrize("function", [lag, lead, first_value, last_value])
    def test_a_window_subject_refuses_a_string(self, function: object) -> None:
        with pytest.raises(TypeError, match=r"col\(\)"):
            function("x")  # type: ignore[operator]

    def test_the_expression_forms_still_work(self, con: duckdb.frame.Connection) -> None:
        plan = (
            duckdb.frame.table("t")
            .select(
                "id",
                col("x").sum().over(partition_by=col("g")).alias("by_group"),
                lag(col("x")).over(order_by=[col("id")]).alias("before"),
                lag(lit("x")).over(order_by=col("id")).alias("text_before"),
                when(col("x") > 15).then("big").otherwise("small").alias("size"),
                col("x").sum().where(col("x") > 10).over().alias("big_total"),
            )
            .sort("id")
        )
        assert plan.rows(con) == [
            (1, 40, None, None, "small", 50),
            (2, 20, 10, "x", "big", 50),
            (3, 40, 20, "x", "big", 50),
        ]


class TestAValueSlotTakesAStringAsText:
    def test_operators_and_value_methods_read_text(self, con: duckdb.frame.Connection) -> None:
        plan = (
            duckdb.frame.table("t")
            .filter((col("g") == "a") & col("g").isin(["a", "z"]) & col("g").between("a", "a") & col("g").like("a%"))
            .select(col("g").concat("-", "z").alias("joined"), col("g").str().replace("a", "y").alias("replaced"))
        )
        assert plan.rows(con) == [("a-z", "y"), ("a-z", "y")]

    def test_a_function_parameter_reads_text(self, con: duckdb.frame.Connection) -> None:
        plan = sql("SELECT DATE '2026-10-13' AS d").select(col("d").dt().trunc("month").alias("m"))
        assert plan.rows(con) == [(datetime.datetime(2026, 10, 1),)]


class TestTheReadersRefuseWhatNoWayReads:
    def test_an_empty_alias_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            col("x").alias("")

    def test_an_unknown_sample_method_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="bernoulli, reservoir, system"):
            duckdb.frame.table("t").sample(1, method="zzz")  # type: ignore[arg-type]

    def test_sql_text_must_be_a_string(self) -> None:
        with pytest.raises(TypeError, match="SQL"):
            sql(1)  # type: ignore[arg-type]

    def test_drop_takes_a_name_not_an_expression(self) -> None:
        with pytest.raises(TypeError, match="column name"):
            duckdb.frame.table("t").drop(col("x"))  # type: ignore[arg-type]


# The classes a caller reaches without importing them, by being handed one; with the classes duckdb.frame exports,
# they are the classes whose constructors and methods are public. A returned class missing from both fails
# `test_every_returned_class_is_checked`.
RETURNED = [
    duckdb.frame.Frame,
    duckdb.frame.Bound,
    GroupedFrame,
    duckdb.frame.Connection,
    duckdb.frame.Expr,
    duckdb.frame.Side,
    CaseBuilder,
    ThenBuilder,
    StrExpr,
    DtExpr,
    ListExpr,
    JsonExpr,
]
# An Iterable, Sequence or Collection admits a bare string, so the slot must say through an alias how it reads one;
# Any admits everything, so it says nothing.
UNNAMED = {"str", "object", "Any", "Iterable", "Sequence", "Collection"}
# object.__eq__ and object.__ne__ take object, and an override may not narrow what they take.
EXEMPT_METHODS = {"__eq__", "__ne__"}


PACKAGE = Path(duckdb.__file__).parent


def is_protocol(name: str) -> bool:
    """A method Python or another library calls by its name: an operator, `__arrow_c_stream__`, `_repr_html_`."""
    dunder = name.startswith("__") and name.endswith("__") and name not in ("__init__", "__new__")
    return dunder or (name.startswith("_repr_") and name.endswith("_"))


def written_here(thing: object) -> bool:
    """Whether a method is this package's own code, not one Python or a dataclass generated for it."""
    code = getattr(thing, "__code__", None)
    return code is not None and Path(code.co_filename).is_relative_to(PACKAGE)


def surface() -> list[type]:
    exported = [thing for name in duckdb.frame.__all__ if isinstance(thing := getattr(duckdb.frame, name), type)]
    return list(dict.fromkeys([*exported, *RETURNED]))


def public_callables() -> list[tuple[str, str, Callable[..., object]]]:
    found: list[tuple[str, str, Callable[..., object]]] = []
    for name in duckdb.frame.__all__:
        thing = getattr(duckdb.frame, name)
        if callable(thing) and not isinstance(thing, type):
            found.append(("duckdb.frame", name, thing))
    seen: set[int] = set()
    for cls in surface():
        # An exception's constructor is Python's own.
        if not issubclass(cls, BaseException):
            found.append((cls.__name__, "constructor", cls))
        for name in dir(cls):
            thing = getattr(cls, name)
            if (name.startswith("_") and not is_protocol(name)) or not callable(thing) or id(thing) in seen:
                continue
            if not written_here(thing):
                continue
            seen.add(id(thing))
            found.append((thing.__qualname__.rsplit(".", 1)[0], name, thing))
    return found


def unnamed_types(annotation: str) -> set[str]:
    """The names in an annotation that do not say how a string is read; a callable's own types are its business."""
    found: set[str] = set()
    stack: list[ast.AST] = [ast.parse(annotation, mode="eval")]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "Callable":
            continue
        if isinstance(node, ast.Name) and node.id in UNNAMED:
            found.add(node.id)
        stack.extend(ast.iter_child_nodes(node))
    return found


def offenders() -> list[str]:
    """Every public parameter whose annotation does not say how it reads a string."""
    found = []
    for owner, name, function in public_callables():
        for parameter in inspect.signature(function).parameters.values():
            if parameter.name == "self" or (name in EXEMPT_METHODS and parameter.name == "other"):
                continue
            if parameter.annotation is inspect.Parameter.empty:
                found.append(f"{owner}.{name}({parameter.name}) is unannotated")
                continue
            # A NamedTuple holds its field annotations as forward references rather than text.
            held = parameter.annotation
            annotation = held.__forward_arg__ if isinstance(held, ForwardRef) else held
            assert isinstance(annotation, str), f"{owner}.{name}({parameter.name}) has an evaluated annotation"
            if unnamed := unnamed_types(annotation):
                found.append(f"{owner}.{name}({parameter.name}: {annotation}) names {sorted(unnamed)}")
    return found


class TestEverySignatureNamesItsReading:
    def test_the_surface_is_what_it_was(self) -> None:
        # A floor, so a walk that misses the methods cannot pass the checks below by checking nothing.
        assert len(public_callables()) > 350
        walked = {(owner, name) for owner, name, _ in public_callables()}
        assert {
            ("Column", "constructor"),
            ("Step", "constructor"),
            ("Step", "render"),
            ("Step", "__setstate__"),
            ("Frame", "constructor"),
            ("Bound", "__arrow_c_stream__"),
            ("Bound", "_repr_html_"),
            ("Side", "__getattr__"),
            ("Connection", "__exit__"),
        } <= walked

    def test_the_check_sees_a_string_slot_and_looks_past_a_callable(self) -> None:
        assert unnamed_types("Iterable[object] | None") == {"Iterable", "object"}
        assert unnamed_types("str | tuple[str, ...]") == {"str"}
        assert unnamed_types("Any") == {"Any"}
        assert unnamed_types("Expr | Callable[[Side, Side], object] | None") == set()
        assert unnamed_types("Callable[..., Any]") == set()
        assert unnamed_types("ColumnNames | None") == set()

    def test_no_parameter_admits_a_string_without_saying_how_it_reads_it(self) -> None:
        assert offenders() == []

    @pytest.mark.parametrize("annotation", ["Any", "str", "Iterable[ColumnName]"])
    def test_a_widened_argument_is_caught(self, monkeypatch: pytest.MonkeyPatch, annotation: str) -> None:
        monkeypatch.setitem(col.__annotations__, "name", annotation)
        assert [line for line in offenders() if line.startswith("duckdb.frame.col(")] == [
            f"duckdb.frame.col(name: {annotation}) names {sorted(unnamed_types(annotation))}"
        ]

    def test_every_returned_class_is_checked(self) -> None:
        checked = {cls.__name__ for cls in surface()}
        returned = set()
        for _, _, function in public_callables():
            annotation = inspect.signature(function).return_annotation
            if isinstance(annotation, str):
                returned |= {
                    node.id for node in ast.walk(ast.parse(annotation, mode="eval")) if isinstance(node, ast.Name)
                }
        ours = {name for name in returned if inspect.isclass(getattr(duckdb.frame, name, None) or globals().get(name))}
        assert ours - checked == set()

    def test_the_stable_aliases_are_the_public_ones(self) -> None:
        assert sorted(typing.__all__) == [
            "ColumnName",
            "ColumnNames",
            "ColumnOrExpr",
            "ColumnsOrExprs",
            "Exprs",
            "FilePath",
            "JoinHow",
            "NewName",
            "Parameters",
            "PythonValue",
            "QualifiedName",
            "SqlText",
            "ValueOrExpr",
            "ValueSequence",
        ]

    def test_every_alias_names_a_reader_that_exists(self) -> None:
        tree = ast.parse(Path(typing.__file__).read_text())
        aliases: dict[str, str] = {}
        for node, following in itertools.pairwise(tree.body):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                held = following.value if isinstance(following, ast.Expr) else None
                docstring = held.value if isinstance(held, ast.Constant) else None
                aliases[node.target.id] = docstring if isinstance(docstring, str) else ""
        assert set(typing.__all__) <= set(aliases)
        for alias, docstring in aliases.items():
            named = re.search(r"Reader: `([\w.]+)`", docstring)
            assert named is not None, f"{alias} names no reader"
            module, _, attribute = named.group(1).partition(".")
            found: object = importlib.import_module(module)
            for part in attribute.split("."):
                if not hasattr(found, part):
                    found = importlib.import_module(f"{module}.{part}") if isinstance(found, ModuleType) else None
                    module = f"{module}.{part}"
                else:
                    found = getattr(found, part)
                assert found is not None, f"{alias} names a reader that does not exist: {named.group(1)}"


class TestEachReader:
    @pytest.mark.parametrize(("value", "expected"), [("a.b", "a.b"), ("Ab", "Ab")], ids=["dotted", "cased"])
    def test_as_name_keeps_a_name_whole(self, value: str, expected: str) -> None:
        assert _as_name(value, "column name") == expected

    @pytest.mark.parametrize(("value", "error"), [(1, TypeError), (col("x"), TypeError), ("", ValueError)])
    def test_as_name_refuses_what_is_not_a_name(self, value: object, error: type[Exception]) -> None:
        with pytest.raises(error, match="column name"):
            _as_name(value, "column name")

    def test_as_names_reads_a_bare_string_as_one_name(self) -> None:
        assert _as_names("id", "column name") == ("id",)
        assert _as_names(["id", "a.b"], "column name") == ("id", "a.b")
        assert _as_names((n for n in ["x"]), "column name") == ("x",)

    @pytest.mark.parametrize("names", [1, b"id", col("id")], ids=["int", "bytes", "expression"])
    def test_as_names_refuses_what_is_neither_a_name_nor_a_list(self, names: object) -> None:
        with pytest.raises(TypeError, match="expected a column name or a list of them"):
            _as_names(names, "column name")

    def test_as_names_refuses_a_list_holding_something_else(self) -> None:
        with pytest.raises(TypeError, match="column name"):
            _as_names(["id", 1], "column name")

    def test_name_parts_qualifies_with_a_tuple_only(self) -> None:
        assert name_parts("a.b") == ("a.b",)
        assert name_parts(("s", "t")) == ("s", "t")
        with pytest.raises(TypeError):
            name_parts(["s", "t"])

    def test_as_field_key_takes_a_name_or_a_position(self) -> None:
        assert _as_field_key("f") == "f"
        assert _as_field_key(2) == 2
        with pytest.raises(TypeError, match="bool"):
            _as_field_key(True)

    def test_as_sql_and_as_text_take_strings(self) -> None:
        assert _as_sql("1 + 1", "sql_expr") == "1 + 1"
        assert _as_text("!", "escape") == "!"
        with pytest.raises(TypeError, match="sql_expr takes SQL"):
            _as_sql(b"1", "sql_expr")
        with pytest.raises(TypeError, match="escape is text"):
            _as_text(1, "escape")

    def test_lift_reads_a_string_as_text(self) -> None:
        assert repr(_lift("x")) == "<Expr 'x'>"
        assert repr(_lift(col("x"))) == '<Expr "x">'

    def test_as_expression_takes_an_expression_only(self) -> None:
        assert repr(_as_expression(col("x"), "when")) == '<Expr "x">'
        with pytest.raises(TypeError, match=r"when takes an expression, not the string 'x'.*col\(\).*sql_expr\(\)"):
            _as_expression("x", "when")
        with pytest.raises(TypeError, match=r"when takes an expression, not int; wrap a value in lit\(\)"):
            _as_expression(1, "when")

    def test_as_expressions_takes_one_or_a_list_or_tuple(self) -> None:
        assert len(_as_expressions(col("x"), "order_by")) == 1
        assert len(_as_expressions([col("x"), col("y")], "order_by")) == 2
        assert len(_as_expressions((col("x"), col("y")), "order_by")) == 2
        with pytest.raises(TypeError, match="order_by"):
            _as_expressions([col("x"), "y"], "order_by")

    @pytest.mark.parametrize("keys", [{col("x")}, (e for e in [col("x")])], ids=["set", "generator"])
    def test_as_expressions_refuses_a_collection_without_an_order(self, keys: object) -> None:
        # A window's order_by follows the order given, which a set does not have.
        with pytest.raises(TypeError, match="order_by takes an expression, not"):
            _as_expressions(keys, "order_by")

    def test_isin_refuses_a_bare_string_and_reads_list_items_as_values(self) -> None:
        assert repr(col("g").isin(["a", "b"])) == "<Expr (\"g\" IN ('a', 'b'))>"
        with pytest.raises(TypeError, match="isin takes a list"):
            col("g").isin("ab")

    def test_as_expr_and_as_exprs_read_a_string_as_a_column(self) -> None:
        assert repr(_as_expr("x")) == '<Expr "x">'
        assert [repr(e) for e in _as_exprs("x")] == ['<Expr "x">']
        assert [repr(e) for e in _as_exprs(["x", col("y")])] == ['<Expr "x">', '<Expr "y">']
        with pytest.raises(TypeError, match="lit"):
            _as_expr(1)

    def test_as_rows_and_as_heading(self) -> None:
        assert len(_as_rows([("a", 1)])[0]) == 2
        with pytest.raises(TypeError, match="list of rows"):
            _as_rows(b"ab")
        with pytest.raises(TypeError, match="list of rows"):
            _as_rows(42)
        with pytest.raises(TypeError, match="row of values"):
            _as_rows([1])
        assert _as_heading("ab") == (Column("ab", None),)
        assert _as_heading(["a", ("b", "INTEGER")]) == (Column("a", None), Column("b", "INTEGER"))
        with pytest.raises(TypeError, match="column type"):
            _as_heading([("b", 1)])
        with pytest.raises(TypeError, match="name or a"):
            _as_heading([("b", "INTEGER", 1)])
        for columns in (b"col", 42):
            with pytest.raises(TypeError, match="names or"):
                _as_heading(columns)

    def test_as_named_parameters_takes_a_mapping(self) -> None:
        assert _as_named_parameters(None) == {}
        assert _as_named_parameters({"p": "x"}) == {"p": "x"}
        with pytest.raises(TypeError, match="mapping"):
            _as_named_parameters([("p", "x")])

    def test_as_sample_method_takes_the_engines_words_in_any_case(self) -> None:
        assert _as_sample_method("Bernoulli") == "BERNOULLI"
        with pytest.raises(TypeError, match="sample method"):
            _as_sample_method(1)

    def test_option_reads_an_identifier(self) -> None:
        assert _option("partition_by") == "PARTITION_BY"
        with pytest.raises(ValueError, match="copy option"):
            _option("drop table")

    def test_option_value_and_argument_write_values(self) -> None:
        assert _option_value("format", "csv") == "'csv'"
        assert _option_value("partition_by", ["year"]) == "('year')"
        assert repr(_argument(Path("x.csv"))) == "<Expr 'x.csv'>"

    def test_as_parameters_refuses_only_a_bare_string(self) -> None:
        assert _as_parameters(["a"]) == ["a"]
        assert _as_parameters({"a": 1}) == {"a": 1}
        with pytest.raises(TypeError, match=r"for one value write \['ab'\]"):
            _as_parameters("ab")

    def test_as_macro_parameters_reads_a_bare_string_as_one_parameter(self) -> None:
        assert _as_macro_parameters("ab") == ["ab"]
        assert _as_macro_parameters(["a", ("b", 1)]) == ["a", ("b", 1)]
        with pytest.raises(TypeError, match="macro parameter"):
            _as_macro_parameters([1])
        for parameters in (b"ab", 42):
            with pytest.raises(TypeError, match="names or"):
                _as_macro_parameters(parameters)

    def test_join_how_folds_case_and_refuses_other_words(self, con: duckdb.frame.Connection) -> None:
        left = duckdb.frame.table("t").select("id")
        assert left.join(left, on="id", how="Left").count(con) == 3  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown join kind"):
            left.join(left, on="id", how="sideways")  # type: ignore[arg-type]

    def test_function_words_are_exact(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(duckdb.exceptions.InvalidInputError, match="null_handling"):
            con.create_function("f", lambda x: x, ["INTEGER"], "INTEGER", null_handling="SPECIAL")  # type: ignore[arg-type]
        with pytest.raises(duckdb.exceptions.InvalidInputError, match="stability"):
            con.create_function("f", lambda x: x, ["INTEGER"], "INTEGER", stability="stable")  # type: ignore[arg-type]

    def test_create_function_types_refuse_a_bare_string(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(TypeError, match="list or a dict of SQL type texts"):
            con.create_function("f", lambda x: x, "INTEGER", "INTEGER")

    def test_register_refuses_a_string(self, con: duckdb.frame.Connection) -> None:
        with pytest.raises(TypeError, match="str is none of these"):
            con.register("s", "abc")

    def test_connect_passes_setting_values_as_text(self, tmp_path: Path) -> None:
        with duckdb.frame.connect(threads=2, temp_directory=tmp_path) as connection:
            assert sql("SELECT current_setting('threads')").rows(connection) == [(2,)]
            assert sql("SELECT current_setting('temp_directory')").rows(connection) == [(str(tmp_path),)]
