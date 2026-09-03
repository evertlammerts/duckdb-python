"""Build a piece of SQL as a Python value.

An expression describes a computation. It is not evaluated until you run it,
and it holds no connection, schema or types, so one can be reused against
different tables.

    col("total") > 1000     ->  ("total" > 1000)

Two rules differ from plain Python:

- A bare string is a value, never a column name. Use `col` for a column.
- Combine with `&`, `|` and `~`. Python's `and`/`or` cannot be overloaded.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import datetime
import decimal
import inspect
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, cast

from ._aggregates import AggregateMethods

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from ._func_namespaces import DtExpr, JsonExpr, ListExpr, StrExpr

__all__ = [
    "Expr",
    "ParamSink",
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


class PlanBase:
    """What an expression knows about a plan: that it renders to a query.

    `Frame` subclasses this. A base class rather than a protocol so that only
    a plan is accepted where a plan is meant: a structural check would take
    anything with a `render` method, a template engine included, and splice
    it into the SQL.
    """

    def render(self, connection: Any = None) -> str:  # pragma: no cover (abstract)
        raise NotImplementedError


def qualified(name: str) -> str:
    """A dotted name, quoted part by part so `main.orders` stays two identifiers."""
    return ".".join(quote(part) for part in name.split("."))


# What an operand may be without an explicit lit(). The temporal, decimal and
# UUID types are here because the binding layer converts them exactly; leaving
# them out made the expression layer narrower than the values it can bind.
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

# Widening order for the numeric literals, so a list of mixed numbers gets an
# element type that none of them overflows. Numbers are the only values that
# widen: the engine refuses a list or a map that mixes text and numbers, and
# a claimed VARCHAR would only move that refusal from the SQL to the binding.
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


def sql_type_of(value: object) -> str | None:
    """The SQL type to bind a Python value as, or None when it is ambiguous."""
    # Integer widths mirror DuckDB's literal typing, so a bound value lands on
    # the same type an inline literal would have.
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
    # datetime before date: datetime subclasses date, so order decides.
    if isinstance(value, datetime.datetime):
        return "TIMESTAMP WITH TIME ZONE" if value.tzinfo else "TIMESTAMP"
    if isinstance(value, datetime.date):
        return "DATE"
    if isinstance(value, datetime.time):
        return "TIME WITH TIME ZONE" if value.tzinfo else "TIME"
    if isinstance(value, datetime.timedelta):
        return "INTERVAL"
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
        # One rule for a dict, shared with `render_literal` and with the
        # converter in pyconv.cpp: text keys make a STRUCT, any other keys a
        # MAP. An empty dict is an empty STRUCT, which has no type to bind as
        # and is written into the SQL instead.
        if not value:
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
    # Text, bytes and composites: the injection surface, so never inlined.
    # Temporal, decimal and UUID: bound because the converter keeps them exact,
    # a Decimal's scale and a datetime's offset included.
    # Numbers, booleans and NULL are absent here: nothing to escape, and
    # inlining keeps DuckDB's own literal typing.
    return isinstance(
        value,
        (str, bytes, list, tuple, dict, datetime.date, datetime.time, datetime.timedelta, decimal.Decimal, uuid.UUID),
    )


def render_literal(value: object) -> str:
    """A literal rendered into the SQL text. Only used where it is safe to."""
    if value is None:
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
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    if isinstance(value, bytes):
        # One escape per byte. DuckDB reads exactly two hex digits after
        # each escape, so a single one over the whole string would take
        # everything past the first byte as literal text.
        escaped = "".join(f"\\x{byte:02x}" for byte in value)
        return f"'{escaped}'::BLOB"
    # These render for the schema oracle only, where no sink is active. They
    # are built from the object's own fields, never from caller text, so there
    # is nothing here to escape.
    if isinstance(value, datetime.datetime):
        keyword = "TIMESTAMPTZ" if value.tzinfo else "TIMESTAMP"
        return f"{keyword} '{value.isoformat(sep=' ')}'"
    if isinstance(value, datetime.date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, datetime.time):
        return f"TIME '{value.isoformat()}'"
    if isinstance(value, datetime.timedelta):
        return f"INTERVAL '{value.total_seconds()} seconds'"
    if isinstance(value, decimal.Decimal):
        if not value.is_finite():
            message = f"cannot render a non-finite Decimal: {value!r}"
            raise TypeError(message)
        # Width and scale come from the value itself: a bare DECIMAL is
        # DECIMAL(18,3), which silently rounds anything finer.
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
        return "[" + ", ".join(render_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        # The schema oracle's form only; a dict binds as a parameter when a
        # sink is active. Text keys are a struct, anything else a map.
        if all(isinstance(k, str) for k in value):
            entries = ", ".join(f"{render_literal(k)}: {render_literal(v)}" for k, v in value.items())
            return "{" + entries + "}"
        entries = ", ".join(f"{render_literal(k)}: {render_literal(v)}" for k, v in value.items())
        return "MAP {" + entries + "}"
    message = f"cannot render a literal of type {type(value).__name__}: {value!r}"
    raise TypeError(message)


# --- parameter sink --------------------------------------------------------

_sink_stack: contextvars.ContextVar[tuple[ParamSink, ...]] = contextvars.ContextVar("duckdb_param_sink", default=())

#: The refusal a `param()` meets when it renders with no sink active, or None
#: where the NULL stand-in is right. Separate from the sink stack, so that a
#: nested `suspended_sinks()` cannot lift it.
_param_refusal: contextvars.ContextVar[str | None] = contextvars.ContextVar("duckdb_param_refusal", default=None)


class ParamSink:
    """Collects the values a plan binds, in render order, numbering them `$n`.

    Lifted literals and `param()` placeholders share one numbering space.
    Rendering with no sink active leaves real literals in place, which is what
    the schema oracle needs: an untyped `$n` tells the binder nothing.
    """

    __slots__ = ("_token", "entries")

    def __init__(self) -> None:
        #: ("literal", value, type) or ("reference", name)
        self.entries: list[tuple[str, Any, str | None]] = []

    def add_literal(self, value: object, type_text: str | None) -> int:
        """Record a lifted literal, returning its 1-based position."""
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


#: While a plan renders, the step name of every plan in its graph, keyed by
#: identity. A subquery whose plan is in here renders as a reference to that
#: step instead of inlining its text, so a plan used twice is computed once.
_step_names: contextvars.ContextVar[dict[int, str] | None] = contextvars.ContextVar("duckdb_step_names", default=None)


@contextlib.contextmanager
def rendering_steps(names: dict[int, str]) -> Iterator[None]:
    """Make step names visible to subqueries for the duration of a render."""
    token = _step_names.set(names)
    try:
        yield
    finally:
        _step_names.reset(token)


def _children(value: object) -> Iterable[object]:
    """What an expression-tree walk descends into: the fields of a node, the items of a list or tuple.

    The one definition of a child, so that a field holding its expressions
    in a list is walked like any other.
    """
    if isinstance(value, Expr):
        return vars(value).values()
    if isinstance(value, (list, tuple)):
        return value
    return ()


def parameters_in(value: object) -> list[str]:
    """The names of every `param()` an expression (or a container of them) holds, in tree order."""
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
    """Every plan an expression (or a container of them) refers to, in tree order, each once."""
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
    """The innermost sink, if one is active."""
    stack = _sink_stack.get()
    return stack[-1] if stack else None


@contextlib.contextmanager
def suspended_sinks() -> Iterator[None]:
    """Render with no sink, so literals stay in the text.

    The schema oracle needs this. An untyped `$1` tells the binder nothing,
    while the literal it stands for tells it everything.
    """
    token = _sink_stack.set(())
    try:
        yield
    finally:
        _sink_stack.reset(token)


@contextlib.contextmanager
def refusing_parameters(message: str) -> Iterator[None]:
    """Make a `param()` that renders with no sink raise `TypeError(message)` instead of standing in NULL.

    For a rendering that has nothing to bind a parameter to, a macro
    definition: NULL would be written into it and every call would answer
    NULL.
    """
    token = _param_refusal.set(message)
    try:
        yield
    finally:
        _param_refusal.reset(token)


# --- the tree --------------------------------------------------------------


def _coerce(other: object) -> Expr | Any:
    """An Expr passes through, a literal wraps, a callable becomes a lambda, anything else defers to Python."""
    if isinstance(other, Expr):
        return other
    if other is None or isinstance(other, LITERAL_TYPES):
        return Lit(other)
    if callable(other):
        return _as_lambda(other)
    return NotImplemented


def _as_lambda(function: Callable[..., object]) -> Expr:
    """A SQL lambda from a Python one, by running it once with a variable per parameter.

    The function is called at build time with expressions standing for its
    parameters, so whatever it returns is the body, already a tree; the
    names in the SQL are the Python parameters' own. It must build an
    expression: a plain value is taken as a constant body, and anything
    else is refused here, where the mistake is.

    Every parameter becomes a lambda variable, so the signature must be
    plain positional names: no defaults, which would silently pick a
    different engine overload, and no keyword-only, `*args` or `**kwargs`
    parameters. A callable with no parameters is refused too: a SQL lambda
    has no zero-variable form, and a bare function where a value was meant
    is usually a call that was never made.
    """
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
    if not isinstance(body, Expr) and not (body is None or isinstance(body, LITERAL_TYPES)):
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


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


class FuncNamespaces:
    """The function namespace entries, on their own class because `str` and `list` shadow builtins."""

    def str(self) -> StrExpr:
        """This expression as text: the string functions, `col("s").str().upper()`.

        A function namespace is a method scope, not a cast: nothing is
        checked here, and the engine still judges every call. The families
        exist for two reasons: without them, every engine function would be
        a method on every expression, several hundred names after each dot;
        and the family prefixes could not be dropped, since `.list().min()`
        can mean `list_min` only while `.min()` still means the aggregate.
        """
        from ._func_namespaces import StrExpr

        return StrExpr(cast("Expr", self))

    def dt(self) -> DtExpr:
        """This expression as a date, time or timestamp: `col("placed").dt().year()`."""
        from ._func_namespaces import DtExpr

        return DtExpr(cast("Expr", self))

    def list(self) -> ListExpr:
        """This expression as a list: `col("tags").list().contains("vip")`."""
        from ._func_namespaces import ListExpr

        return ListExpr(cast("Expr", self))

    def json(self) -> JsonExpr:
        """This expression as a JSON document: `col("payload").json().extract("$.id")`.

        JSON is text underneath, so the string functions are in scope too.
        """
        from ._func_namespaces import JsonExpr

        return JsonExpr(cast("Expr", self))


class Expr(AggregateMethods, FuncNamespaces):
    """One node of an expression tree."""

    def __init__(self) -> None:
        self._alias: str | None = None
        self._order: str | None = None

    # -- rendering

    def fragment(self) -> str:
        """The SQL for this node, parenthesised so precedence never bites."""
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
        """A copy carrying different presentation. Nodes stay immutable.

        Mutable fields are copied, not shared. Subclasses keep their operands
        in lists and dicts, and a clone that aliased them would let a change
        through one name show up under the other.
        """
        clone = object.__new__(type(self))
        clone.__dict__.update(
            (name, list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict) else value)
            for name, value in self.__dict__.items()
        )
        clone.__dict__.update(changes)
        return clone

    # -- presentation

    def alias(self, name: str) -> Expr:
        """Name the column in the output."""
        return self._with(_alias=name)

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

    def __eq__(self, other: object) -> Expr | Any:  # type: ignore[override]
        # `col("x") == None` is a SQL NULL comparison, which is never true. That
        # is SQL's answer, and surfacing it beats inventing one.
        return self._binary("=", other)

    def __ne__(self, other: object) -> Expr | Any:  # type: ignore[override]
        return self._binary("!=", other)

    def __lt__(self, other: object) -> Expr | Any:
        return self._binary("<", other)

    def __le__(self, other: object) -> Expr | Any:
        return self._binary("<=", other)

    def __gt__(self, other: object) -> Expr | Any:
        return self._binary(">", other)

    def __ge__(self, other: object) -> Expr | Any:
        return self._binary(">=", other)

    def __add__(self, other: object) -> Expr | Any:
        return self._binary("+", other)

    def __radd__(self, other: object) -> Expr | Any:
        return self._binary("+", other, reverse=True)

    def __sub__(self, other: object) -> Expr | Any:
        return self._binary("-", other)

    def __rsub__(self, other: object) -> Expr | Any:
        return self._binary("-", other, reverse=True)

    def __mul__(self, other: object) -> Expr | Any:
        return self._binary("*", other)

    def __rmul__(self, other: object) -> Expr | Any:
        return self._binary("*", other, reverse=True)

    def __truediv__(self, other: object) -> Expr | Any:
        return self._binary("/", other)

    def __rtruediv__(self, other: object) -> Expr | Any:
        return self._binary("/", other, reverse=True)

    def __floordiv__(self, other: object) -> Expr | Any:
        return self._binary("//", other)

    def __mod__(self, other: object) -> Expr | Any:
        return self._binary("%", other)

    def __pow__(self, other: object) -> Expr | Any:
        return self._binary("**", other)

    def __and__(self, other: object) -> Expr | Any:
        return self._binary("AND", other)

    def __or__(self, other: object) -> Expr | Any:
        return self._binary("OR", other)

    def __invert__(self) -> Expr:
        return Unary("NOT", self)

    def __neg__(self) -> Expr:
        return Unary("-", self)

    def __bool__(self) -> bool:
        """Refuse to be treated as a condition.

        `==` builds a node rather than comparing, so an expression is always
        truthy. That makes `col("a") in [col("b")]` true, and `if col("x") ==
        1:` always taken. Raising turns both into an error at the line that
        wrote them.
        """
        message = "an expression has no truth value; combine with & | ~, and test with .is_null()"
        raise TypeError(message)

    def __hash__(self) -> int:
        # Defined because __eq__ is, and an expression is not a dict key.
        return id(self)

    # -- predicates and casts

    def n_unique(self) -> Expr:
        """How many distinct values there are, ignoring NULL."""
        # count(DISTINCT x) is syntax, not a function name, so it cannot go in
        # the shortcut table with the others.
        return Distinct("count", self)

    def concat(self, *others: object) -> Expr:
        """Join text values."""
        # SQL concatenates with ||; + on two text values is an error. An
        # expression carries no types, so + cannot work out which was meant.
        return Concat([self, *(_lift(o) for o in others)])

    def is_null(self) -> Expr:
        """Whether this is NULL. Not the same as `== None`, which gives NULL."""
        return Postfix("IS NULL", self)

    def is_not_null(self) -> Expr:
        """Whether this is not NULL."""
        return Postfix("IS NOT NULL", self)

    def isin(self, values: Iterable[object] | PlanBase | Expr) -> Expr:
        """Membership: in a list of values, in a one-column plan, or in a list-typed expression.

        An empty list is never a match. Given an expression, the engine reads
        it as a list and tests membership of it row by row: `x IN xs`.
        """
        if isinstance(values, PlanBase):
            return Binary("IN", self, SubQuery(values))
        if isinstance(values, Expr):
            return Binary("IN", self, values)
        if isinstance(values, (str, bytes)):
            # Iterating text would test each character. Nobody means that, and
            # one value is what `==` is for.
            message = (
                f"isin takes a list of values, a query or a list-typed expression; for one value use == {values!r}"
            )
            raise TypeError(message)
        return In(self, [_lift(v) for v in values])

    def like(self, pattern: object, *, escape: str | None = None) -> Expr:
        """Text match, where `%` is any run of characters and `_` is any one.

        LIKE is an operator rather than a function, so `fn("like", ...)` cannot
        reach it. Negate with `~`.
        """
        return Like("LIKE", self, _lift(pattern), escape)

    def ilike(self, pattern: object, *, escape: str | None = None) -> Expr:
        """`like`, ignoring case."""
        return Like("ILIKE", self, _lift(pattern), escape)

    def between(self, low: object, high: object) -> Expr:
        """Inclusive range."""
        return Between(self, _lift(low), _lift(high))

    def cast(self, type_text: str) -> Expr:
        """Cast to a SQL type, written as text."""
        return Cast(self, type_text)

    def try_cast(self, type_text: str) -> Expr:
        """Cast to a SQL type, giving NULL where the cast would fail."""
        return Cast(self, type_text, safe=True)

    def where(self, predicate: object) -> Expr:
        """Aggregate only the rows where the predicate holds: `sum(x) FILTER (WHERE ...)`.

        Applies to an aggregate call, `count_all()` included. Put it before
        `.over()` when the aggregate is a window. Which calls aggregate is the
        engine's to say: extensions add aggregates, so there is no closed list
        here, and a scalar call is refused when the plan is bound or run.
        """
        if not isinstance(self, (Func, Distinct)):
            message = "where() applies to an aggregate call, as col('x').sum() or fn('sum', ...)"
            raise TypeError(message)
        return self._with(_filter=_lift(predicate))

    def over(
        self,
        partition_by: Iterable[object] | object | None = None,
        order_by: Iterable[object] | object | None = None,
        *,
        rows: tuple[int | None, int | None] | None = None,
        range: tuple[int | None, int | None] | None = None,
    ) -> Expr:
        """Turn an aggregate into a window function.

        `rows` or `range` bounds the window as (start, end), each counted from
        the current row: a negative number is that many before, a positive
        number that many after, 0 is the current row and None is unbounded.
        So `rows=(-2, 0)` is the current row and the two before it.
        """
        partitions = [_lift(e) for e in _as_list(partition_by)] if partition_by is not None else []
        orders = [_lift(e) for e in _as_list(order_by)] if order_by is not None else []
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
        """A function call with this expression at `position` among the arguments.

        For the functions that take their subject other than first, as
        `date_trunc('month', ts)` and `list_prepend(e, list)` do.
        """
        lifted = [_lift(a) for a in args]
        return Func(function, [*lifted[:position], self, *lifted[position:]])

    def __repr__(self) -> str:
        # Rendered into a sink and read back, so a parameter shows its name
        # and a literal its value, where a blind render shows NULL and text.
        with ParamSink() as sink:
            text = self.fragment()
        for position, (kind, value, _) in reversed(list(enumerate(sink.entries, 1))):
            shown = f"${value}" if kind == "reference" else render_literal(value)
            text = text.replace(f"${position}", shown)
        return f"<Expr {text}>"


class Col(Expr):
    """A column reference, optionally qualified."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def fragment(self) -> str:
        # A dotted name is a qualified reference: each part is quoted on its
        # own, so `l.id` becomes "l"."id" rather than one odd identifier.
        # Needed to disambiguate a join, where both sides can carry the name.
        return qualified(self.name)


class FamilyExpr(Expr):
    """A function namespace over an expression: the same expression, with one family's methods in scope.

    Entering one (`.str()`, `.dt()`, `.list()`, `.json()`) asserts how the
    expression is meant; it changes nothing about the expression, which is
    why this renders as what it wraps and keeps its alias and order. A
    family class is a method scope, not a type: the engine still owns the
    types and judges every call.
    """

    #: Method name to (function, position of the expression, parameter types),
    #: filled in by each generated family class; the tests read it.
    SPEC: ClassVar[dict[str, tuple[str, int, list[str]]]] = {}

    def __init__(self, inner: Expr) -> None:
        super().__init__()
        self.inner = inner
        # A name or direction set before entering the family must survive it.
        self._alias = inner._alias
        self._order = inner._order

    def fragment(self) -> str:
        return self.inner.fragment()

    # The aggregate builders gate on what the call really is, which only the
    # wrapped expression can answer; they apply there and keep the family in
    # scope, so entering one before or after them reads the same.

    def where(self, predicate: object) -> Expr:
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

    def __init__(self, exclude: Iterable[str] = (), rename: dict[str, str] | None = None) -> None:
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
    """A Python value: inlined where that is safe, otherwise bound as `$n`."""

    def __init__(self, value: object) -> None:
        super().__init__()
        # A snapshot: a plan is a value, so a list or dict the caller goes on
        # changing must not change the plan with it.
        try:
            self.value = copy.deepcopy(value) if isinstance(value, (list, tuple, dict)) else value
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
    """A named placeholder whose value is supplied at execution."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def fragment(self) -> str:
        sink = active_sink()
        if sink is None:
            refusal = _param_refusal.get()
            if refusal is not None:
                raise TypeError(refusal)
            # Rendering without a sink happens for the schema oracle, where an
            # untyped placeholder tells the binder nothing. NULL binds and
            # carries no type either, which is the honest stand-in.
            return "NULL"
        return f"${sink.add_reference(self.name)}"


class Raw(Expr):
    """A SQL fragment supplied by the caller, spliced in unchanged."""

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
    """A lambda's parameter: a name bound inside the lambda, not a column.

    Inside the body the engine gives the name to the variable even when a
    column has it too, so the Python parameter's name decides what it
    shadows.
    """

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
            # The plan is a step of the query being rendered, so this is a
            # reference to it. Rendered once as a CTE, however often it is
            # used, which is what "computed once" means.
            return f"(SELECT * FROM {names[id(self.query)]})"
        # Rendered on its own, with its own steps local to these parentheses.
        # Its literals still reach whichever sink is active.
        return f"({self.query.render()})"


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
    """A function call."""

    def __init__(self, name: str, args: list[Expr]) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self._ignore_nulls = False
        self._filter: Expr | None = None

    def fragment(self) -> str:
        rendered = ", ".join(a.fragment() for a in self.args)
        # IGNORE NULLS goes inside the call, where DuckDB reads it; after the
        # closing parenthesis it is a syntax error.
        tail = " IGNORE NULLS" if self._ignore_nulls else ""
        return f"{self.name}({rendered}{tail})" + _filter_clause(self._filter)


class Distinct(Expr):
    """An aggregate over distinct values, such as `count(DISTINCT x)`."""

    def __init__(self, name: str, operand: Expr) -> None:
        super().__init__()
        self.name = name
        self.operand = operand
        self._filter: Expr | None = None

    def fragment(self) -> str:
        return f"{self.name}(DISTINCT {self.operand.fragment()})" + _filter_clause(self._filter)


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

    def then(self, result: object) -> ThenBuilder:
        """The value for the condition just given."""
        return ThenBuilder([*self._branches, (self._condition, _lift(result))])


class ThenBuilder:
    """A CASE with at least one complete branch."""

    def __init__(self, branches: list[tuple[Expr, Expr]]) -> None:
        self._branches = branches

    def when(self, condition: object) -> CaseBuilder:
        """Another condition."""
        return CaseBuilder(self._branches, _lift(condition))

    def otherwise(self, result: object) -> Expr:
        """The fallback, completing the expression."""
        return Case(self._branches, _lift(result))

    def end(self) -> Expr:
        """Complete the expression with no fallback, so unmatched rows are NULL."""
        return Case(self._branches)


# --- constructors ----------------------------------------------------------


def col(name: str) -> Expr:
    """A column, by name. Always quoted, so reserved words and spaces are fine."""
    return Col(name)


def lit(value: object) -> Expr:
    """A Python value. Rarely needed, since operators accept values directly."""
    return Lit(value)


def param(name: str) -> Expr:
    """A placeholder whose value is supplied when the query runs."""
    return Param(name)


def star(exclude: Iterable[str] = (), rename: dict[str, str] | None = None) -> Expr:
    """`*`, optionally dropping or renaming columns."""
    return Star(exclude, rename)


def fn(name: str, *args: object) -> Expr:
    """Any SQL function by name. Arguments follow the usual binding rules."""
    return Func(name, [_lift(a) for a in args])


def sql_expr(sql: str) -> Expr:
    """A raw fragment, spliced in unchanged.

    Nothing in it is quoted, escaped or bound. Never build one from untrusted
    input.
    """
    return Raw(sql)


def when(condition: object) -> CaseBuilder:
    """Begins a CASE. Continue with `.then(...)`."""
    return CaseBuilder([], _lift(condition))


def coalesce(*values: object) -> Expr:
    """The first argument that is not NULL."""
    return Func("coalesce", [_lift(v) for v in values])


def count_all() -> Expr:
    """How many rows there are: `count(*)`.

    Not a method, because it counts rows rather than a column's values.
    `col("x").count()` skips NULLs; this does not.
    """
    return Func("count", [Star()])


def _window(name: str) -> Any:
    def make(*args: object) -> Expr:
        return Func(name, [_lift(a) for a in args])

    make.__name__ = name
    make.__doc__ = f"The `{name}` window function."
    return make


row_number = _window("row_number")
rank = _window("rank")
dense_rank = _window("dense_rank")
lag = _window("lag")
lead = _window("lead")
ntile = _window("ntile")
first_value = _window("first_value")
last_value = _window("last_value")

__all__ += [
    "dense_rank",
    "first_value",
    "lag",
    "last_value",
    "lead",
    "ntile",
    "qualified",
    "quote",
    "rank",
    "row_number",
    "sql_type_of",
]
