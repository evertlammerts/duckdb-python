"""Build a piece of SQL as a Python value.

An expression such as `col("total") > 1000` describes a computation and holds no connection, schema or types, so the
same one can be used against different tables. Two rules differ from plain Python: a bare string is a value and `col`
names a column, and conditions combine with `&`, `|` and `~`, since Python's `and` and `or` cannot be overloaded.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import datetime
import decimal
import inspect
import re
import string
import sys
import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

from .. import _duckdb, exceptions
from .aggregates import AggregateMethods
from .keywords import KEYWORDS

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..frame.connection import Connection
    from ..frame.plan import Frame
    from ..frame.typing import (
        ColumnName,
        ColumnNames,
        Exprs,
        NewName,
        PythonValue,
        QualifiedName,
        SqlText,
        ValueOrExpr,
        ValueSequence,
        _AttributeName,
        _FieldKey,
        _Renames,
        _TextValue,
    )
    from .func_namespaces import DtExpr, JsonExpr, ListExpr, StrExpr

__all__ = [
    "Expr",
    "ParamSink",
    "Side",
    "coalesce",
    "col",
    "count_all",
    "fn",
    "lit",
    "param",
    "sql_expr",
    "star",
    "when",
]


def quote(name: str) -> str:
    """A SQL identifier, always quoted so no name can be read as syntax."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def fold_name(name: str) -> str:
    """A name as DuckDB compares names: ASCII letters case folded, every other character as written."""
    return name.translate(_ASCII_LOWER)


class PlanBase:
    """What an expression needs of a query; a base class, so nothing else with a `render` method can be spliced in."""

    def render(self, connection: Connection | None = None) -> str:  # pragma: no cover (abstract)
        raise NotImplementedError


def name_parts(name: object, what: str = "name") -> tuple[str, ...]:
    """A string as a one-part path, a tuple of strings as a qualified path. Nothing is ever split."""
    parts = (name,) if isinstance(name, str) else name
    if not isinstance(parts, tuple) or not all(isinstance(part, str) for part in parts):
        message = f"a {what} is a string, or a tuple of strings for a qualified one, not {name!r}"
        raise TypeError(message)
    if not parts or not all(parts):
        message = f"a {what} cannot be empty: {name!r}"
        raise ValueError(message)
    return parts


def _as_name(name: object, what: str) -> str:
    """One name, whole: a string, never split and never empty."""
    if not isinstance(name, str):
        message = f"a {what} is a string, not {name!r}"
        raise TypeError(message)
    if not name:
        message = f"a {what} cannot be empty"
        raise ValueError(message)
    return name


def _items(value: object) -> Iterator[object] | None:
    """The items of what Python can iterate, through `__iter__` or `__getitem__`, or None; text is the caller's."""
    try:
        return iter(cast("Iterable[object]", value))
    except TypeError:
        return None


def _as_names(names: object, what: str) -> tuple[str, ...]:
    """One name or several, where a bare string is one name and not its characters."""
    if isinstance(names, str):
        return (_as_name(names, what),)
    items = None if isinstance(names, bytes) else _items(names)
    if items is None:
        message = f"expected a {what} or a list of them, not {names!r}"
        raise TypeError(message)
    return tuple(_as_name(name, what) for name in items)


def _as_renames(renames: object) -> dict[str, str] | None:
    """Old names to new, as a mapping or as (old, new) pairs; a string is never a pair of names."""
    if renames is None or isinstance(renames, Mapping):
        pairs: Iterator[object] | None = iter(renames.items()) if renames is not None else iter(())
    else:
        pairs = None if isinstance(renames, (str, bytes)) else _items(renames)
    if pairs is None:
        message = f"rename takes a dict from old name to new, or (old, new) pairs, not {renames!r}"
        raise TypeError(message)
    out: dict[str, str] = {}
    for pair in pairs:
        both = None if isinstance(pair, (str, bytes)) else _items(pair)
        named = tuple(both) if both is not None else ()
        if len(named) != 2:
            message = f"a rename is an (old, new) pair of names, not {pair!r}"
            raise TypeError(message)
        out[_as_name(named[0], "column name")] = _as_name(named[1], "column name")
    return out or None


def _as_sql(text: object, what: str) -> str:
    """SQL text, taken as written."""
    if not isinstance(text, str):
        message = f"{what} takes SQL as a string, not {text!r}"
        raise TypeError(message)
    return text


def _as_text(value: object, what: str) -> str:
    """A value that must be text."""
    if not isinstance(value, str):
        message = f"{what} is text, not {value!r}"
        raise TypeError(message)
    return value


def _as_field_key(key: object) -> str | int:
    """A struct field's name, a map key, or a list position."""
    if isinstance(key, bool) or not isinstance(key, (str, int)):
        message = f"an expression is indexed by a field name or a position, not by {type(key).__name__}"
        raise TypeError(message)
    return key


def identifier(parts: tuple[str, ...]) -> str:
    """The SQL for a name's parts, each quoted, joined by dots. The parts were checked where they were made."""
    return ".".join(quote(part) for part in parts)


plain_identifier = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def function_name(name: str | tuple[str, ...]) -> str:
    """A function's name as DuckDB writes it, quoted only when it must be; syntax forms like COALESCE bypass it."""
    return ".".join(
        part if plain_identifier.match(part) and fold_name(part) not in KEYWORDS else quote(part)
        for part in name_parts(name, "function name")
    )


# What an operand may be without an explicit lit(); dates, decimals and UUIDs are here because they convert exactly.
LITERAL_TYPES = (
    bool,
    int,
    float,
    str,
    bytes,
    list,
    tuple,
    datetime.date,
    datetime.time,
    datetime.timedelta,
    decimal.Decimal,
    uuid.UUID,
    dict,
)

# Widening order for numeric literals, so a list of mixed numbers gets a type none overflows; only numbers widen.
_NUMERIC_RANK = {"INTEGER": 1, "BIGINT": 2, "HUGEINT": 3, "DOUBLE": 4}

_INT32 = 2**31
_INT64 = 2**63


def _widen(types: list[str]) -> str | None:
    """A common element type for a list, or None when there is no safe one."""
    unique = list(dict.fromkeys(types))
    if len(unique) == 1:
        return unique[0]
    if all(t in _NUMERIC_RANK for t in unique):
        return "DOUBLE" if "DOUBLE" in unique else max(unique, key=lambda t: _NUMERIC_RANK[t])
    return None


def _is_nat(value: object) -> bool:
    """Whether `value` is pandas' or numpy's NaT, which stands for a missing value as None does."""
    cls = type(value)
    if cls.__name__ == "NaTType" and cls.__module__.startswith("pandas"):
        return True
    # numpy's NaT is the least count, in any unit and step.
    return _is_numpy_temporal(value) and int(cast("Any", value).view("i8")) == -(2**63)


def _is_numpy_temporal(value: object) -> bool:
    """Whether `value` is a numpy datetime64 or timedelta64 scalar, recognised without importing numpy."""
    cls = type(value)
    return cls.__module__ == "numpy" and cls.__name__ in ("datetime64", "timedelta64")


#: Exact types no numpy scalar can have, so a value of one needs no numpy check; a subclass such as np.float64 does.
_NATIVE = frozenset(
    {
        type(None),
        bool,
        int,
        float,
        str,
        bytes,
        decimal.Decimal,
        uuid.UUID,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
    }
)

_CONTAINERS = (list, tuple, dict)


class _NumpyTypes(NamedTuple):
    scalars: tuple[type, ...]
    duration: type


def _numpy_types() -> _NumpyTypes | None:
    """The scalar types of numpy, or None when no value can be a numpy scalar."""
    # Looked up rather than imported, so that importing duckdb does not import numpy; without it there are no scalars.
    numpy = sys.modules.get("numpy")
    # A numpy still importing on another thread can lack these, and then nothing has made one of its scalars yet.
    found = [getattr(numpy, name, None) for name in ("bool_", "integer", "floating", "timedelta64")]
    if numpy is None or not all(isinstance(kind, type) for kind in found):
        return None
    *scalars, duration = cast("list[type]", found)
    return _NumpyTypes(tuple(scalars), duration)


def _converted(value: object, numpy: _NumpyTypes) -> object:
    """One value that is no container, a numpy bool, integer or float scalar becoming its Python value."""
    if not isinstance(value, numpy.scalars):
        return value
    # A timedelta64 is a numpy integer too, and its unit would be lost.
    return value if isinstance(value, numpy.duration) else cast("Any", value).item()


def _as_python(value: object) -> object:
    """A numpy bool, integer or float scalar as the Python value it stands for, and anything else as it is."""
    if type(value) in _NATIVE:
        return value
    numpy = _numpy_types()
    return value if numpy is None else _converted(value, numpy)


def _all_native(container: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> bool:
    """Whether a list, tuple or dict holds only values of the native types, so nothing in it needs converting."""
    if isinstance(container, dict):
        return _NATIVE.issuperset(map(type, container)) and _NATIVE.issuperset(map(type, container.values()))
    return _NATIVE.issuperset(map(type, container))


def _container_values(container: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> Sequence[object]:
    """A container's values in order, a dict's keys and values alternating."""
    if isinstance(container, dict):
        return [part for entry in container.items() for part in entry]
    return container


def _rebuilt(container: object, children: list[object]) -> object:
    """A container of the same kind holding `children`, in the order `_container_values` gives them."""
    if isinstance(container, dict):
        return dict(zip(children[::2], children[1::2], strict=True))
    return children if isinstance(container, list) else tuple(children)


def plain_value(value: object) -> object:
    """`value` with every numpy bool, integer and float scalar in it, however deeply nested, as its Python value.

    A container holding none comes back as the same object, so a caller still sees whatever happens to it later.
    """
    if type(value) in _NATIVE:
        return value
    numpy = _numpy_types()
    if numpy is None:
        return value
    if not isinstance(value, _CONTAINERS):
        return _converted(value, numpy)
    if _all_native(value):
        return value
    # A stack, not recursion, so a value nested deeper than Python's recursion limit still binds as it did before.
    # Each entry holds a container, its values, where the walk over them stands, and the values converted so far.
    first = _container_values(value)
    stack: list[tuple[object, Sequence[object], Iterator[object], list[object]]] = [(value, first, iter(first), [])]
    open_ids = {id(value)}
    while True:
        container, values, pending, done = stack[-1]
        for child in pending:
            if type(child) in _NATIVE:
                done.append(child)
            elif not isinstance(child, _CONTAINERS):
                done.append(_converted(child, numpy))
            elif id(child) in open_ids or _all_native(child):
                # A container met again inside itself stays as it is, so a cycle cannot keep the walk going.
                done.append(child)
            else:
                inner = _container_values(child)
                stack.append((child, inner, iter(inner), []))
                open_ids.add(id(child))
                break
        else:
            stack.pop()
            open_ids.discard(id(container))
            unchanged = all(new is old for new, old in zip(done, values, strict=True))
            rebuilt = container if unchanged else _rebuilt(container, done)
            if not stack:
                return rebuilt
            stack[-1][3].append(rebuilt)


def _is_temporal(value: object) -> bool:
    """Whether `value` is a date, time or duration of Python's, pandas' or numpy's, which the engine module converts."""
    return isinstance(value, (datetime.date, datetime.time, datetime.timedelta)) or _is_numpy_temporal(value)


def _conversion_type_of(value: object) -> str | None:
    """The type a parameter binds as, by the engine module's own conversion.

    Refused, the value is written in instead, where the same conversion refuses it again.
    """
    try:
        return _duckdb.literal_type(value)
    except exceptions.Error:
        return None


def _holds_temporal(value: object) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_holds_temporal(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_temporal(key) or _holds_temporal(item) for key, item in value.items())
    return _is_temporal(value)


def sql_type_of(value: object) -> str | None:
    """The SQL type to bind a Python value as, or None when it is ambiguous."""
    # Integer widths mirror DuckDB's own, so a bound value lands on the type an inline literal would have.
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        if -_INT32 <= value < _INT32:
            return "INTEGER"
        if -_INT64 <= value < _INT64:
            return "BIGINT"
        return "HUGEINT"
    if isinstance(value, float):
        return "DOUBLE"
    if isinstance(value, str):
        return "VARCHAR"
    if isinstance(value, bytes):
        return "BLOB"
    if _is_nat(value):
        return None
    if isinstance(value, datetime.datetime):
        # A pandas Timestamp or other subclass carries its own unit, which only the engine's conversion types.
        if type(value) is datetime.datetime:
            return "TIMESTAMP WITH TIME ZONE" if value.utcoffset() is not None else "TIMESTAMP"
        return _conversion_type_of(value)
    if isinstance(value, datetime.date):
        return "DATE"
    if isinstance(value, datetime.time):
        # An aware time's offset may depend on a date, which only the conversion can refuse.
        return "TIME" if value.tzinfo is None else _conversion_type_of(value)
    if isinstance(value, datetime.timedelta):
        return "INTERVAL"
    if _is_numpy_temporal(value):
        return _conversion_type_of(value)
    if isinstance(value, decimal.Decimal):
        return "DECIMAL"
    if isinstance(value, uuid.UUID):
        return "UUID"
    if isinstance(value, (list, tuple)):
        element_types = [sql_type_of(item) for item in value]
        if not element_types or any(t is None for t in element_types):
            return None
        element = _widen([t for t in element_types if t is not None])
        return f"{element}[]" if element else None
    if isinstance(value, dict):
        # Text keys make a STRUCT and any others a MAP, the same rule `render_literal` and the C++ converter use.
        if not value:
            # An empty dict is an empty STRUCT, which has no type to bind as, so it is written into the SQL.
            return None
        if all(isinstance(k, str) for k in value):
            fields = [(k, sql_type_of(v)) for k, v in value.items()]
            if any(t is None for _, t in fields):
                return None
            return "STRUCT(" + ", ".join(f"{quote(k)} {t}" for k, t in fields) + ")"
        key_types = [sql_type_of(k) for k in value]
        value_types = [sql_type_of(v) for v in value.values()]
        if any(t is None for t in key_types) or any(t is None for t in value_types):
            return None
        key = _widen([t for t in key_types if t is not None])
        val = _widen([t for t in value_types if t is not None])
        return f"MAP({key}, {val})" if key and val else None
    return None


def _needs_param(value: object) -> bool:
    """Whether a value is bound as a parameter rather than written into the SQL."""
    # Text and composites stay out of the SQL, decimals and dates bind exactly, and numbers inline as DuckDB types them.
    return _is_numpy_temporal(value) or isinstance(
        value,
        (str, bytes, list, tuple, dict, datetime.date, datetime.time, datetime.timedelta, decimal.Decimal, uuid.UUID),
    )


def render_literal(value: object) -> str:
    """A value written into the SQL text, only where that is safe.

    A date, time or duration, alone or anywhere in a list or dict, is converted by the engine module as it would be
    bound as a parameter, so it means the same written in, and is refused where a parameter would be.
    """
    value = plain_value(value)
    if isinstance(value, (list, tuple, dict)) and _holds_temporal(value):
        _duckdb.literal_type(value)
    return _rendered(value)


def _rendered(value: object) -> str:
    if value is None or _is_nat(value):
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "'NaN'::DOUBLE"
        if value == float("inf"):
            return "'Infinity'::DOUBLE"
        if value == float("-inf"):
            return "'-Infinity'::DOUBLE"
        # Written bare, DuckDB would read 1.5 as DECIMAL, where a Python float is a double.
        return f"{value!r}::DOUBLE"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bytes):
        # One escape per byte: DuckDB reads exactly two hex digits after each, so one escape would cover only the first.
        escaped = "".join(f"\\x{byte:02x}" for byte in value)
        return f"'{escaped}'::BLOB"
    if _is_temporal(value):
        return _duckdb.temporal_literal(value)
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            message = f"cannot render a non-finite Decimal: {value!r}"
            raise TypeError(message)
        # Width and scale come from the value: a bare DECIMAL is DECIMAL(18,3) and silently rounds anything finer.
        _, digits, exponent = value.as_tuple()
        scale = max(0, -int(exponent))
        width = max(len(digits) + max(int(exponent), 0), scale + 1)
        if width > 38:
            message = f"cannot render a Decimal wider than 38 digits: {value!r}"
            raise TypeError(message)
        return f"CAST('{value}' AS DECIMAL({width}, {scale}))"
    if isinstance(value, uuid.UUID):
        return f"UUID '{value}'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_rendered(item) for item in value) + "]"
    if isinstance(value, dict):
        # Reached only when nothing is collecting parameters; text keys make a struct, anything else a map.
        if all(isinstance(k, str) for k in value):
            entries = ", ".join(f"{_rendered(k)}: {_rendered(v)}" for k, v in value.items())
            return "{" + entries + "}"
        entries = ", ".join(f"{_rendered(k)}: {_rendered(v)}" for k, v in value.items())
        return "MAP {" + entries + "}"
    message = f"cannot render a literal of type {type(value).__name__}: {value!r}"
    raise TypeError(message)


# --- collecting the values a query binds -----------------------------------

_sink_stack: contextvars.ContextVar[tuple[ParamSink, ...]] = contextvars.ContextVar("duckdb_param_sink", default=())

#: The error a `param()` raises when nothing is collecting values, or None where standing in NULL is right.
_param_refusal: contextvars.ContextVar[str | None] = contextvars.ContextVar("duckdb_param_refusal", default=None)


class ParamSink:
    """Collects the values a query binds, numbered `$1`, `$2` and so on; `param()` shares the numbering."""

    __slots__ = ("_token", "entries")

    def __init__(self) -> None:
        #: ("literal", value, type) or ("reference", name)
        self.entries: list[tuple[str, Any, str | None]] = []

    def add_literal(self, value: object, type_text: str | None) -> int:
        """Record a value pulled out of the SQL, returning its 1-based position."""
        self.entries.append(("literal", value, type_text))
        return len(self.entries)

    def add_reference(self, name: str) -> int:
        """Record a named placeholder, returning its 1-based position."""
        self.entries.append(("reference", name, None))
        return len(self.entries)

    def __enter__(self) -> ParamSink:
        self._token = _sink_stack.set((*_sink_stack.get(), self))
        return self

    def __exit__(self, *exc: object) -> None:
        _sink_stack.reset(self._token)


#: While a query is being built, each step's name by identity, so a subquery used twice becomes one reference.
_step_names: contextvars.ContextVar[dict[int, str] | None] = contextvars.ContextVar("duckdb_step_names", default=None)


@contextlib.contextmanager
def rendering_steps(names: dict[int, str]) -> Iterator[None]:
    """Make the step names visible to subqueries while the SQL is built."""
    token = _step_names.set(names)
    try:
        yield
    finally:
        _step_names.reset(token)


def _children(value: object) -> Iterable[object]:
    """What a walk over an expression descends into: a node's fields, and the items of a list or tuple."""
    if isinstance(value, Expr):
        return vars(value).values()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def parameters_in(value: object) -> list[str]:
    """The names of every `param()` an expression, or a container of them, holds, in tree order."""
    found: list[str] = []
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Param):
            found.append(item.name)
            continue
        stack.extend(reversed(list(_children(item))))
    return found


def subqueries(value: object) -> list[PlanBase]:
    """Every query an expression, or a container of them, refers to, in tree order, each once."""
    found: list[PlanBase] = []
    seen: set[int] = set()
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, SubQuery):
            if id(item.query) not in seen:
                seen.add(id(item.query))
                found.append(item.query)
            continue
        stack.extend(reversed(list(_children(item))))
    return found


def active_sink() -> ParamSink | None:
    """The innermost collector of parameter values, if one is active."""
    stack = _sink_stack.get()
    return stack[-1] if stack else None


@contextlib.contextmanager
def suspended_sinks() -> Iterator[None]:
    """Build SQL with values written in, since `$1` tells DuckDB nothing when it is asked for column types."""
    token = _sink_stack.set(())
    try:
        yield
    finally:
        _sink_stack.reset(token)


@contextlib.contextmanager
def refusing_parameters(message: str) -> Iterator[None]:
    """Make a `param()` raise `TypeError(message)` rather than stand in NULL, which a macro body would bake in."""
    token = _param_refusal.set(message)
    try:
        yield
    finally:
        _param_refusal.reset(token)


# --- the tree --------------------------------------------------------------


def _coerce(other: object) -> Expr | Any:
    """An Expr passes through, a literal wraps, a callable becomes a lambda, anything else defers to Python."""
    other = _as_python(other)
    if isinstance(other, Expr):
        return other
    if other is None or isinstance(other, LITERAL_TYPES) or _is_numpy_temporal(other):
        return Lit(other)
    if callable(other):
        return _as_lambda(other)
    return NotImplemented


def _as_lambda(function: Callable[..., object]) -> Expr:
    """A SQL lambda from a Python one, called once when built; a default would pick a different DuckDB overload."""
    label = getattr(function, "__name__", None) or repr(function)
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError):
        message = f"cannot read the signature of {label} to build a SQL lambda from it"
        raise TypeError(message) from None
    if not parameters:
        message = f"a SQL lambda takes at least one parameter and {label} takes none; did you mean to call it?"
        raise TypeError(message)
    plain = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    for parameter in parameters:
        if parameter.kind not in plain or parameter.default is not inspect.Parameter.empty:
            message = (
                f"cannot build a SQL lambda from {label}: every parameter becomes a lambda "
                f"variable, so the signature must be plain positional names without defaults, "
                f"and `{parameter}` ({parameter.kind.description}) is not one"
            )
            raise TypeError(message)
    names = [p.name for p in parameters]
    try:
        body = function(*(Variable(name) for name in names))
    except TypeError as reason:
        message = f"while building a SQL lambda from {label}: {reason}"
        raise TypeError(message) from reason
    body = _as_python(body)
    if not isinstance(body, Expr) and not (body is None or isinstance(body, LITERAL_TYPES) or _is_numpy_temporal(body)):
        message = (
            f"the lambda returned {body!r}, which is not an expression; it runs once, at build "
            f"time, on expressions rather than values, so its body must be built from them"
        )
        raise TypeError(message)
    return Lambda(names, _lift(body))


def _lift(value: object) -> Expr:
    coerced = _coerce(value)
    if coerced is NotImplemented:
        message = f"cannot use {value!r} as an expression operand"
        raise TypeError(message)
    return coerced


def _as_expression(value: object, what: str) -> Expr:
    """An argument only an expression can fill, such as a condition, where a string would silently be a constant."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        message = (
            f"{what} takes an expression, not the string {value!r}; "
            f"name a column with col() or write SQL with sql_expr()"
        )
        raise TypeError(message)
    message = f"{what} takes an expression, not {type(value).__name__}; wrap a value in lit()"
    raise TypeError(message)


def _as_expressions(values: object, what: str) -> list[Expr]:
    """One expression, or a list or tuple of them; a set has no order to give a window, so it is not several."""
    items = list(values) if isinstance(values, (list, tuple)) else [values]
    return [_as_expression(item, what) for item in items]


class FuncNamespaces:
    """The function namespace entries, on their own class because `str` and `list` shadow builtins."""

    def str(self) -> StrExpr:
        """This expression with the string functions in scope, `col("s").str().upper()`; nothing is checked or cast."""
        from .func_namespaces import StrExpr

        return StrExpr(cast("Expr", self))

    def dt(self) -> DtExpr:
        """This expression as a date, time or timestamp: `col("placed").dt().year()`."""
        from .func_namespaces import DtExpr

        return DtExpr(cast("Expr", self))

    def list(self) -> ListExpr:
        """This expression as a list: `col("tags").list().contains("vip")`."""
        from .func_namespaces import ListExpr

        return ListExpr(cast("Expr", self))

    def json(self) -> JsonExpr:
        """This expression as JSON, with the string functions in scope too: `col("p").json().extract("$.id")`."""
        from .func_namespaces import JsonExpr

        return JsonExpr(cast("Expr", self))


class Expr(AggregateMethods, FuncNamespaces):
    """One node of an expression tree."""

    # numpy then hands an operator with a numpy scalar on its left back to the expression, which keeps the scalar,
    # where numpy would convert it to a Python value first and lose its unit.
    __array_ufunc__ = None

    def __init__(self) -> None:
        self._alias: str | None = None
        self._order: str | None = None

    # -- turning into SQL

    def fragment(self) -> str:
        """The SQL for this node, parenthesised so precedence cannot change what it means."""
        raise NotImplementedError

    def as_select(self) -> str:
        """The SQL for a select list, carrying the alias if there is one."""
        rendered = self.fragment()
        return f"{rendered} AS {quote(self._alias)}" if self._alias else rendered

    def as_order(self) -> str:
        """The SQL for an ORDER BY item, carrying the direction if there is one."""
        rendered = self.fragment()
        return f"{rendered} {self._order}" if self._order else rendered

    def _with(self, **changes: object) -> Expr:
        """A copy with different presentation, its lists and dicts copied so a change cannot reach the original."""
        clone = object.__new__(type(self))
        clone.__dict__.update(
            (name, list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
            for name, value in self.__dict__.items()
        )
        clone.__dict__.update(changes)
        return clone

    # -- presentation

    def alias(self, name: NewName) -> Expr:
        """Name the column in the output."""
        return self._with(_alias=_as_name(name, "column name"))

    def asc(self) -> Expr:
        """Sort ascending."""
        return self._with(_order="ASC")

    def desc(self) -> Expr:
        """Sort descending."""
        return self._with(_order="DESC")

    def nulls_first(self) -> Expr:
        """Where NULLs sort. Keeps any direction already set."""
        return self._with(_order=f"{self._order or 'ASC'} NULLS FIRST")

    def nulls_last(self) -> Expr:
        """Where NULLs sort. Keeps any direction already set."""
        return self._with(_order=f"{self._order or 'ASC'} NULLS LAST")

    # -- operators, all pure Python

    def _binary(self, op: str, other: object, *, reverse: bool = False) -> Expr | Any:
        coerced = _coerce(other)
        if coerced is NotImplemented:
            return NotImplemented
        left, right = (coerced, self) if reverse else (self, coerced)
        return Binary(op, left, right)

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        # `col("x") == None` is a SQL NULL comparison and never true, which is SQL's answer rather than an invented one.
        return self._equality("=", other)

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return self._equality("!=", other)

    def _equality(self, op: str, other: object) -> Expr:
        # Raised rather than deferred: Python answers an unknown operand by identity, a False no condition can tell.
        return Binary(op, self, _lift(other))

    def __lt__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("<", other)

    def __le__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("<=", other)

    def __gt__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary(">", other)

    def __ge__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary(">=", other)

    def __add__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("+", other)

    def __radd__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("+", other, reverse=True)

    def __sub__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("-", other)

    def __rsub__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("-", other, reverse=True)

    def __mul__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("*", other)

    def __rmul__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("*", other, reverse=True)

    def __truediv__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("/", other)

    def __rtruediv__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("/", other, reverse=True)

    def __floordiv__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("//", other)

    def __mod__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("%", other)

    def __pow__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("**", other)

    def __and__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("AND", other)

    def __or__(self, other: ValueOrExpr) -> Expr | Any:
        return self._binary("OR", other)

    def __invert__(self) -> Expr:
        return Unary("NOT", self)

    def __neg__(self) -> Expr:
        return Unary("-", self)

    def __getitem__(self, key: _FieldKey) -> Expr:
        """A struct field by name, a list element by position, or a map entry by key."""
        return Index(self, _as_field_key(_as_python(key)))

    def __iter__(self) -> Iterator[Any]:
        # Without this, `for x in e` would fall back to e[0], e[1] and never stop.
        message = "an expression is not iterable; index it with a field name or a position"
        raise TypeError(message)

    def __bool__(self) -> bool:
        """Refuse to be a condition: `==` builds an expression, so `if col("x") == 1:` would always be taken."""
        message = "an expression has no truth value; combine with & | ~, and test with .is_null()"
        raise TypeError(message)

    def __hash__(self) -> int:
        # Defined because __eq__ is, and an expression is not a dict key.
        return id(self)

    # -- predicates and casts

    def n_unique(self) -> Expr:
        """How many distinct values there are, ignoring NULL."""
        # count(DISTINCT x) is syntax, not a function name, so it cannot go in the table with the others.
        return Distinct("count", self)

    def concat(self, *others: ValueOrExpr) -> Expr:
        """Join text values."""
        # SQL concatenates with ||, and an expression carries no types, so + cannot tell which was meant.
        return Concat([self, *(_lift(o) for o in others)])

    def is_null(self) -> Expr:
        """Whether this is NULL. Not the same as `== None`, which gives NULL."""
        return Postfix("IS NULL", self)

    def is_not_null(self) -> Expr:
        """Whether this is not NULL."""
        return Postfix("IS NOT NULL", self)

    def isin(self, values: ValueSequence | Frame | Expr) -> Expr:
        """Membership of a list of values, a one-column query, or a list-typed column; an empty list never matches."""
        if isinstance(values, PlanBase):
            return Binary("IN", self, SubQuery(values))
        if isinstance(values, Expr):
            return Binary("IN", self, values)
        if isinstance(values, (str, bytes)):
            # Iterating text would test each character, which nobody means, and one value is what `==` is for.
            message = (
                f"isin takes a list of values, a query or a list-typed expression; for one value use == {values!r}"
            )
            raise TypeError(message)
        return In(self, [_lift(v) for v in values])

    def like(self, pattern: ValueOrExpr, *, escape: _TextValue | None = None) -> Expr:
        """Text match where `%` is any run of characters and `_` is any one; negate with `~`."""
        return Like("LIKE", self, _lift(pattern), _as_text(escape, "escape") if escape is not None else None)

    def ilike(self, pattern: ValueOrExpr, *, escape: _TextValue | None = None) -> Expr:
        """`like`, ignoring case."""
        return Like("ILIKE", self, _lift(pattern), _as_text(escape, "escape") if escape is not None else None)

    def between(self, low: ValueOrExpr, high: ValueOrExpr) -> Expr:
        """Inclusive range."""
        return Between(self, _lift(low), _lift(high))

    def cast(self, type_text: SqlText) -> Expr:
        """Cast to a SQL type, written as text and spliced in as SQL, so never taken from untrusted input."""
        return Cast(self, _as_sql(type_text, "cast"))

    def try_cast(self, type_text: SqlText) -> Expr:
        """Cast to a SQL type, giving NULL where the cast would fail; the type is SQL text, as for `cast`."""
        return Cast(self, _as_sql(type_text, "try_cast"), safe=True)

    def where(self, predicate: Expr) -> Expr:
        """Aggregate only rows where the predicate holds; goes before `.over()`, and a scalar call fails at run time."""
        if not isinstance(self, (Func, Distinct)):
            message = "where() applies to an aggregate call, as col('x').sum() or fn('sum', ...)"
            raise TypeError(message)
        return self._with(_filter=_as_expression(predicate, "where"))

    def over(
        self,
        partition_by: Exprs | None = None,
        order_by: Exprs | None = None,
        *,
        rows: tuple[int | None, int | None] | None = None,
        range: tuple[int | None, int | None] | None = None,
    ) -> Expr:
        """Turn an aggregate into a window function.

        `rows` and `range` bound it as (start, end), counted from the current row, with None meaning unbounded.
        So `rows=(-2, 0)` is the current row and the two before it.
        """
        partitions = _as_expressions(partition_by, "partition_by") if partition_by is not None else []
        orders = _as_expressions(order_by, "order_by") if order_by is not None else []
        if rows is not None and range is not None:
            message = "a window is bounded by rows or by range, not both"
            raise TypeError(message)
        frame = ("ROWS", rows) if rows is not None else ("RANGE", range) if range is not None else None
        return Over(self, partitions, orders, frame)

    def ignore_nulls(self) -> Expr:
        """Skip NULLs, for `first_value`, `last_value`, `lag`, `lead` and `nth_value`."""
        if not isinstance(self, Func):
            message = "ignore_nulls applies to a function call"
            raise TypeError(message)
        return self._with(_ignore_nulls=True)

    def _call(self, function: str, *args: object) -> Expr:
        """A function call with this expression as the first argument."""
        return Func(function, [self, *(_lift(a) for a in args)])

    def _call_at(self, function: str, position: int, *args: object) -> Expr:
        """A function call with this expression at `position`, for functions like `date_trunc` that take it later."""
        lifted = [_lift(a) for a in args]
        return Func(function, [*lifted[:position], self, *lifted[position:]])

    def __repr__(self) -> str:
        # Values are collected and put back, so a parameter shows its name where the plain SQL would show NULL.
        with ParamSink() as sink:
            text = self.fragment()
        for position, (kind, value, _) in reversed(list(enumerate(sink.entries, 1))):
            shown = f"${value}" if kind == "reference" else render_literal(value)
            text = text.replace(f"${position}", shown)
        return f"<Expr {text}>"


class Col(Expr):
    """A column reference: one name, or a join side's column as two parts."""

    def __init__(self, parts: tuple[str, ...]) -> None:
        super().__init__()
        self.parts = parts

    def fragment(self) -> str:
        return identifier(self.parts)


class Side:
    """One side of a join, inside its condition, where `l["id"]` is that side's column."""

    def __init__(self, alias: NewName) -> None:
        self.alias = alias

    def __getitem__(self, name: ColumnName) -> Expr:
        if not isinstance(name, str):
            message = f"a join side is indexed by column name, not by {type(name).__name__}"  # type: ignore[unreachable]
            raise TypeError(message)
        return Col((self.alias, *name_parts(name, "column name")))

    def __getattr__(self, name: _AttributeName) -> Any:
        if name.startswith("__") or name == "alias":
            raise AttributeError(name)
        message = f"a join side has no attribute {name!r}; its columns are reached as {self.alias}[{name!r}]"
        raise AttributeError(message)

    def __repr__(self) -> str:
        return f"<Side {self.alias}>"


class Index(Expr):
    """A struct field, list element or map entry; the key is written into the SQL, since a field name must be."""

    def __init__(self, base: Expr, key: str | int) -> None:
        super().__init__()
        self.base = base
        self.key = key

    def fragment(self) -> str:
        return f"{self.base.fragment()}[{render_literal(self.key)}]"


class FamilyExpr(Expr):
    """The same expression with one group of functions in scope; the SQL, name and sort direction are unchanged."""

    #: Method name to (function, where the expression goes, parameter types), filled in by each generated class.
    SPEC: ClassVar[dict[str, tuple[str, int, list[str]]]] = {}

    def __init__(self, inner: Expr) -> None:
        super().__init__()
        self.inner = inner
        # A name or sort direction set before entering must survive it.
        self._alias = inner._alias
        self._order = inner._order

    def fragment(self) -> str:
        return self.inner.fragment()

    # These check what the call really is, which only the wrapped expression can answer, then wrap the result again.

    def where(self, predicate: Expr) -> Expr:
        """`Expr.where`, applied to the wrapped aggregate call."""
        return self._rewrapped(self.inner.where(predicate))

    def ignore_nulls(self) -> Expr:
        """`Expr.ignore_nulls`, applied to the wrapped function call."""
        return self._rewrapped(self.inner.ignore_nulls())

    def _rewrapped(self, inner: Expr) -> Expr:
        wrapped = type(self)(inner)
        wrapped._alias = self._alias
        wrapped._order = self._order
        return wrapped


class Star(Expr):
    """`*`, optionally excluding or renaming columns."""

    def __init__(self, exclude: Iterable[str] = (), rename: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.exclude = list(exclude)
        self.rename = dict(rename or {})

    def fragment(self) -> str:
        rendered = "*"
        if self.exclude:
            rendered += " EXCLUDE (" + ", ".join(quote(c) for c in self.exclude) + ")"
        if self.rename:
            pairs = ", ".join(f"{quote(k)} AS {quote(v)}" for k, v in self.rename.items())
            rendered += f" RENAME ({pairs})"
        return rendered


class Lit(Expr):
    """A Python value, written into the SQL where that is safe and otherwise bound as `$n`."""

    def __init__(self, value: object) -> None:
        super().__init__()
        # A snapshot, so a list or dict the caller goes on changing cannot change the query with it.
        try:
            self.value = plain_value(copy.deepcopy(value) if isinstance(value, (list, tuple, dict)) else value)
        except (TypeError, copy.Error) as reason:
            message = f"a literal holds a value that is not plain data: {reason}"
            raise TypeError(message) from None

    def fragment(self) -> str:
        sink = active_sink()
        if sink is not None and _needs_param(self.value):
            type_text = sql_type_of(self.value)
            if type_text is not None:
                position = sink.add_literal(self.value, type_text)
                return f"${position}"
        return render_literal(self.value)


class Param(Expr):
    """A named placeholder whose value is supplied when the query runs."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def fragment(self) -> str:
        sink = active_sink()
        if sink is None:
            refusal = _param_refusal.get()
            if refusal is not None:
                raise TypeError(refusal)
            # Only reached when DuckDB is being asked for column types, where NULL says as little as a placeholder.
            return "NULL"
        return f"${sink.add_reference(self.name)}"


class Raw(Expr):
    """A piece of SQL supplied by the caller, spliced in unchanged."""

    def __init__(self, sql: str) -> None:
        super().__init__()
        self.sql = sql

    def fragment(self) -> str:
        return f"({self.sql})"


class Binary(Expr):
    """A binary operator."""

    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def fragment(self) -> str:
        if self.op == "//":
            return f"({self.left.fragment()} // {self.right.fragment()})"
        if self.op == "**":
            return f"pow({self.left.fragment()}, {self.right.fragment()})"
        return f"({self.left.fragment()} {self.op} {self.right.fragment()})"


class Unary(Expr):
    """A prefix operator."""

    def __init__(self, op: str, operand: Expr) -> None:
        super().__init__()
        self.op = op
        self.operand = operand

    def fragment(self) -> str:
        return f"({self.op} {self.operand.fragment()})"


class Postfix(Expr):
    """A postfix operator, such as IS NULL."""

    def __init__(self, op: str, operand: Expr) -> None:
        super().__init__()
        self.op = op
        self.operand = operand

    def fragment(self) -> str:
        return f"({self.operand.fragment()} {self.op})"


class Cast(Expr):
    """An explicit cast, or a TRY_CAST that gives NULL instead of failing."""

    def __init__(self, operand: Expr, type_text: str, *, safe: bool = False) -> None:
        super().__init__()
        self.operand = operand
        self.type_text = type_text
        self.safe = safe

    def fragment(self) -> str:
        keyword = "TRY_CAST" if self.safe else "CAST"
        return f"{keyword}({self.operand.fragment()} AS {self.type_text})"


class In(Expr):
    """Membership of a value list."""

    def __init__(self, operand: Expr, values: list[Expr]) -> None:
        super().__init__()
        self.operand = operand
        self.values = values

    def fragment(self) -> str:
        if not self.values:
            return "FALSE"  # nothing is a member of an empty set
        rendered = ", ".join(v.fragment() for v in self.values)
        return f"({self.operand.fragment()} IN ({rendered}))"


class Like(Expr):
    """A LIKE or ILIKE match."""

    def __init__(self, keyword: str, operand: Expr, pattern: Expr, escape: str | None) -> None:
        super().__init__()
        self.keyword = keyword
        self.operand = operand
        self.pattern = pattern
        self.escape = escape

    def fragment(self) -> str:
        rendered = f"({self.operand.fragment()} {self.keyword} {self.pattern.fragment()}"
        if self.escape is not None:
            rendered += f" ESCAPE {render_literal(self.escape)}"
        return rendered + ")"


class Variable(Expr):
    """A lambda's parameter, which takes the name even where a column has it too, so the Python name decides."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def fragment(self) -> str:
        return quote(self.name)


class Lambda(Expr):
    """A SQL lambda: `lambda x: body`, as `list_filter` and its siblings take."""

    def __init__(self, names: list[str], body: Expr) -> None:
        super().__init__()
        self.names = names
        self.body = body

    def fragment(self) -> str:
        return f"lambda {', '.join(quote(name) for name in self.names)}: {self.body.fragment()}"


class SubQuery(Expr):
    """A whole query standing in for a value."""

    def __init__(self, query: PlanBase) -> None:
        super().__init__()
        self.query = query

    def fragment(self) -> str:
        names = _step_names.get()
        if names is not None and id(self.query) in names:
            # Already a step of the query being built, so a reference to it, computed once however often it is used.
            return f"(SELECT * FROM {names[id(self.query)]})"
        # Built on its own, with its own steps inside these parentheses; its values still reach the collector.
        return f"({self.query.render()})"


class Untranslatable(ValueError):
    """Raised by a translator for a node it does not model, or would not answer as the engine does."""


class Between(Expr):
    """An inclusive range test."""

    def __init__(self, operand: Expr, low: Expr, high: Expr) -> None:
        super().__init__()
        self.operand = operand
        self.low = low
        self.high = high

    def fragment(self) -> str:
        return f"({self.operand.fragment()} BETWEEN {self.low.fragment()} AND {self.high.fragment()})"


class Func(Expr):
    """A function call, its named arguments after the positional ones."""

    def __init__(
        self, name: str | tuple[str, ...], args: list[Expr], named: list[tuple[str, Expr]] | None = None
    ) -> None:
        super().__init__()
        self.name = name_parts(name, "function name")
        self.args = args
        self.named = named or []
        self._ignore_nulls = False
        self._filter: Expr | None = None

    def fragment(self) -> str:
        rendered = ", ".join(
            [a.fragment() for a in self.args] + [f"{quote(k)} := {v.fragment()}" for k, v in self.named]
        )
        # IGNORE NULLS goes inside the call, where DuckDB reads it; after the closing parenthesis it is a syntax error.
        tail = " IGNORE NULLS" if self._ignore_nulls else ""
        return f"{function_name(self.name)}({rendered}{tail})" + _filter_clause(self._filter)


class Coalesce(Expr):
    """COALESCE, written as the parser's own form, since it is syntax rather than a function."""

    def __init__(self, args: list[Expr]) -> None:
        super().__init__()
        self.args = args

    def fragment(self) -> str:
        return "COALESCE(" + ", ".join(a.fragment() for a in self.args) + ")"


class Distinct(Expr):
    """An aggregate over the distinct values, such as `count(DISTINCT x)`."""

    def __init__(self, name: str | tuple[str, ...], operand: Expr) -> None:
        super().__init__()
        self.name = name_parts(name, "function name")
        self.operand = operand
        self._filter: Expr | None = None

    def fragment(self) -> str:
        return f"{function_name(self.name)}(DISTINCT {self.operand.fragment()})" + _filter_clause(self._filter)


def _filter_clause(predicate: Expr | None) -> str:
    """The FILTER clause of an aggregate, if it has one."""
    return f" FILTER (WHERE {predicate.fragment()})" if predicate is not None else ""


class Concat(Expr):
    """Text concatenation, which SQL spells `||`."""

    def __init__(self, parts: list[Expr]) -> None:
        super().__init__()
        self.parts = parts

    def fragment(self) -> str:
        return "(" + " || ".join(p.fragment() for p in self.parts) + ")"


def _bound(offset: int | None, *, start: bool) -> str:
    """One end of a window frame, from an offset relative to the current row."""
    if offset is None:
        return "UNBOUNDED PRECEDING" if start else "UNBOUNDED FOLLOWING"
    if offset == 0:
        return "CURRENT ROW"
    return f"{abs(int(offset))} {'PRECEDING' if offset < 0 else 'FOLLOWING'}"


class Over(Expr):
    """A window function."""

    def __init__(
        self,
        operand: Expr,
        partitions: list[Expr],
        orders: list[Expr],
        frame: tuple[str, tuple[int | None, int | None]] | None = None,
    ) -> None:
        super().__init__()
        self.operand = operand
        self.partitions = partitions
        self.orders = orders
        self.frame = frame

    def fragment(self) -> str:
        parts = []
        if self.partitions:
            parts.append("PARTITION BY " + ", ".join(p.fragment() for p in self.partitions))
        if self.orders:
            parts.append("ORDER BY " + ", ".join(o.as_order() for o in self.orders))
        if self.frame is not None:
            unit, (start, end) = self.frame
            parts.append(f"{unit} BETWEEN {_bound(start, start=True)} AND {_bound(end, start=False)}")
        return f"{self.operand.fragment()} OVER ({' '.join(parts)})"


class Case(Expr):
    """A CASE expression, built by `when(...).then(...)`."""

    def __init__(self, branches: list[tuple[Expr, Expr]], default: Expr | None = None) -> None:
        super().__init__()
        self.branches = branches
        self.default = default

    def fragment(self) -> str:
        parts = ["CASE"]
        parts += [f"WHEN {c.fragment()} THEN {r.fragment()}" for c, r in self.branches]
        if self.default is not None:
            parts.append(f"ELSE {self.default.fragment()}")
        parts.append("END")
        return "(" + " ".join(parts) + ")"


class CaseBuilder:
    """The half-built CASE that `when()` returns."""

    def __init__(self, branches: list[tuple[Expr, Expr]], condition: Expr) -> None:
        self._branches = branches
        self._condition = condition

    def then(self, result: ValueOrExpr) -> ThenBuilder:
        """The value for the condition just given."""
        return ThenBuilder([*self._branches, (self._condition, _lift(result))])


class ThenBuilder:
    """A CASE with at least one complete branch."""

    def __init__(self, branches: list[tuple[Expr, Expr]]) -> None:
        self._branches = branches

    def when(self, condition: Expr) -> CaseBuilder:
        """Another condition."""
        return CaseBuilder(self._branches, _as_expression(condition, "when"))

    def otherwise(self, result: ValueOrExpr) -> Expr:
        """The fallback, completing the expression."""
        return Case(self._branches, _lift(result))

    def end(self) -> Expr:
        """Complete the expression with no fallback, so unmatched rows are NULL."""
        return Case(self._branches)


# --- constructors ----------------------------------------------------------


def col(name: ColumnName) -> Expr:
    """A column by name, quoted whole so any name works and none is split; a struct field is `col("st")["field"]`."""
    if not isinstance(name, str):
        message = f"col() takes one column name, not {name!r}; a join side's column is l[name] inside the join's on"  # type: ignore[unreachable]
        raise TypeError(message)
    return Col(name_parts(name, "column name"))


def lit(value: PythonValue) -> Expr:
    """A Python value. Rarely needed, since operators accept values directly."""
    return Lit(value)


def param(name: NewName) -> Expr:
    """A placeholder whose value is supplied when the query runs."""
    return Param(_as_name(name, "parameter name"))


def star(exclude: ColumnNames = (), rename: _Renames | None = None) -> Expr:
    """`*`, optionally dropping or renaming columns; `exclude="id"` drops the one column `id`."""
    return Star(_as_names(exclude, "column name"), _as_renames(rename))


def fn(name: QualifiedName, /, *args: ValueOrExpr, **named: ValueOrExpr) -> Expr:
    """Any SQL function by name, a tuple for a schema-qualified one, with the arguments treated as anywhere else.

    A keyword argument passes by name: `fn("round", col("x"), precision=1)` is `round("x", "precision" := 1)`.
    """
    return Func(name, [_lift(a) for a in args], [(key, _lift(value)) for key, value in named.items()])


def sql_expr(sql: SqlText) -> Expr:
    """A raw SQL fragment, spliced in unchanged and never escaped, so never build one from untrusted input."""
    return Raw(_as_sql(sql, "sql_expr"))


def when(condition: Expr) -> CaseBuilder:
    """Begins a CASE. Continue with `.then(...)`."""
    return CaseBuilder([], _as_expression(condition, "when"))


def coalesce(*values: ValueOrExpr) -> Expr:
    """The first argument that is not NULL."""
    return Coalesce([_lift(v) for v in values])


def count_all() -> Expr:
    """How many rows there are, as `count(*)`; unlike `col("x").count()`, NULLs are not skipped."""
    return Func("count", [Star()])


def _window(name: str) -> Any:
    def make(*args: ValueOrExpr) -> Expr:
        return Func(name, [_lift(a) for a in args])

    make.__name__ = name
    make.__doc__ = f"The `{name}` window function."
    return make


row_number = _window("row_number")
rank = _window("rank")
dense_rank = _window("dense_rank")
ntile = _window("ntile")


def _over_subject(name: str, subject: Expr, args: tuple[ValueOrExpr, ...]) -> Expr:
    return Func(name, [_as_expression(subject, name), *(_lift(a) for a in args)])


def lag(subject: Expr, /, *args: ValueOrExpr) -> Expr:
    """The `lag` window function: `subject` some rows back, `lag(col("x"))` or `lag(col("x"), 2, 0)`."""
    return _over_subject("lag", subject, args)


def lead(subject: Expr, /, *args: ValueOrExpr) -> Expr:
    """The `lead` window function: `subject` some rows ahead, `lead(col("x"))` or `lead(col("x"), 2, 0)`."""
    return _over_subject("lead", subject, args)


def first_value(subject: Expr, /, *args: ValueOrExpr) -> Expr:
    """The `first_value` window function: `subject` at the first row of the window frame."""
    return _over_subject("first_value", subject, args)


def last_value(subject: Expr, /, *args: ValueOrExpr) -> Expr:
    """The `last_value` window function: `subject` at the last row of the window frame."""
    return _over_subject("last_value", subject, args)


__all__ += [
    "Untranslatable",
    "dense_rank",
    "first_value",
    "fold_name",
    "function_name",
    "identifier",
    "lag",
    "last_value",
    "lead",
    "name_parts",
    "ntile",
    "plain_identifier",
    "quote",
    "rank",
    "row_number",
    "sql_type_of",
]
