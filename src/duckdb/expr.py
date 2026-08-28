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

import contextvars
import datetime
import decimal
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = [
    "Expr",
    "ParamSink",
    "coalesce",
    "col",
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


@runtime_checkable
class Renderable(Protocol):
    """Anything that can render itself as a complete query.

    A `Frame` satisfies this. Declared as a shape rather than imported so
    expressions stay independent of the frame layer, which imports this one.
    """

    def render(self) -> str: ...


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
)

# Widening order for the numeric literals, so a list of mixed numbers gets an
# element type that none of them overflows.
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
    if "VARCHAR" in unique:
        return "VARCHAR"
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
        (str, bytes, list, tuple, datetime.date, datetime.time, datetime.timedelta, decimal.Decimal, uuid.UUID),
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
        return f"'\\x{value.hex()}'::BLOB"
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
        return f"CAST('{value}' AS DECIMAL)"
    if isinstance(value, uuid.UUID):
        return f"UUID '{value}'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_literal(item) for item in value) + "]"
    message = f"cannot render a literal of type {type(value).__name__}: {value!r}"
    raise TypeError(message)


# --- parameter sink --------------------------------------------------------

_sink_stack: contextvars.ContextVar[tuple[ParamSink, ...]] = contextvars.ContextVar("duckdb_param_sink", default=())


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


def active_sink() -> ParamSink | None:
    """The innermost sink, if one is active."""
    stack = _sink_stack.get()
    return stack[-1] if stack else None


# --- the tree --------------------------------------------------------------


def _coerce(other: object) -> Expr | Any:
    """An Expr passes through, a literal wraps, anything else defers to Python."""
    if isinstance(other, Expr):
        return other
    if other is None or isinstance(other, LITERAL_TYPES):
        return Lit(other)
    return NotImplemented


def _lift(value: object) -> Expr:
    coerced = _coerce(value)
    if coerced is NotImplemented:
        message = f"cannot use {value!r} as an expression operand"
        raise TypeError(message)
    return coerced


def _as_list(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else [value]


class Expr:
    """One node of an expression tree."""

    #: Aggregate and window functions reachable as methods, name to SQL function.
    _CALLABLES: ClassVar[dict[str, str]] = {
        "sum": "sum",
        "mean": "avg",
        "avg": "avg",
        "min": "min",
        "max": "max",
        "count": "count",
        "median": "median",
        "std": "stddev_samp",
        "var": "var_samp",
        "first": "first",
        "last": "last",
        "any_value": "any_value",
        "bit_and": "bit_and",
        "bit_or": "bit_or",
        "bool_and": "bool_and",
        "bool_or": "bool_or",
        "product": "product",
        "string_agg": "string_agg",
        "skewness": "skewness",
        "kurtosis": "kurtosis",
        "entropy": "entropy",
    }

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
        """A copy carrying different presentation. Nodes stay immutable."""
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
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

    def isin(self, values: Iterable[object] | Renderable) -> Expr:
        """Membership, in a list of values or in a one-column query.

        An empty list is never a match.
        """
        if isinstance(values, Renderable):
            return Binary("IN", self, SubQuery(values))
        return In(self, [_lift(v) for v in _as_list(list(values))])

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

    def over(
        self,
        partition_by: Iterable[object] | object | None = None,
        order_by: Iterable[object] | object | None = None,
    ) -> Expr:
        """Turn an aggregate into a window function."""
        partitions = [_lift(e) for e in _as_list(partition_by)] if partition_by is not None else []
        orders = [_lift(e) for e in _as_list(order_by)] if order_by is not None else []
        return Over(self, partitions, orders)

    def __getattr__(self, name: str) -> Any:
        # Aggregate shortcuts, so col("v").sum() reads the way SQL does. Only
        # consulted for names the class does not define.
        function = type(self)._CALLABLES.get(name)
        if function is None:
            message = f"{type(self).__name__!r} object has no attribute {name!r}"
            raise AttributeError(message)

        def call(*args: object) -> Expr:
            return Func(function, [self, *(_lift(a) for a in args)])

        return call

    def __repr__(self) -> str:
        return f"<Expr {self.fragment()}>"


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
        self.value = value

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
    """An explicit cast."""

    def __init__(self, operand: Expr, type_text: str) -> None:
        super().__init__()
        self.operand = operand
        self.type_text = type_text

    def fragment(self) -> str:
        return f"CAST({self.operand.fragment()} AS {self.type_text})"


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


class SubQuery(Expr):
    """A whole query standing in for a value."""

    def __init__(self, query: Renderable) -> None:
        super().__init__()
        self.query = query

    def fragment(self) -> str:
        # Rendered here rather than when this node was built, so the query's
        # own literals reach whichever sink is active for the statement it
        # lands in. Its step names are local to these parentheses; an outer
        # step of the same name is shadowed, not confused with it.
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

    def fragment(self) -> str:
        rendered = ", ".join(a.fragment() for a in self.args)
        return f"{self.name}({rendered})"


class Distinct(Expr):
    """An aggregate over distinct values, such as `count(DISTINCT x)`."""

    def __init__(self, name: str, operand: Expr) -> None:
        super().__init__()
        self.name = name
        self.operand = operand

    def fragment(self) -> str:
        return f"{self.name}(DISTINCT {self.operand.fragment()})"


class Concat(Expr):
    """Text concatenation, which SQL spells `||`."""

    def __init__(self, parts: list[Expr]) -> None:
        super().__init__()
        self.parts = parts

    def fragment(self) -> str:
        return "(" + " || ".join(p.fragment() for p in self.parts) + ")"


class Over(Expr):
    """A window function."""

    def __init__(self, operand: Expr, partitions: list[Expr], orders: list[Expr]) -> None:
        super().__init__()
        self.operand = operand
        self.partitions = partitions
        self.orders = orders

    def fragment(self) -> str:
        parts = []
        if self.partitions:
            parts.append("PARTITION BY " + ", ".join(p.fragment() for p in self.partitions))
        if self.orders:
            parts.append("ORDER BY " + ", ".join(o.as_order() for o in self.orders))
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


def render_with_params(expression: Expr) -> tuple[str, list[tuple[str, Any, str | None]]]:
    """Render an expression with a fresh sink, returning the SQL and its bindings."""
    with ParamSink() as sink:
        return expression.fragment(), list(sink.entries)


def iter_entries(sink: ParamSink) -> Iterator[tuple[str, Any, str | None]]:
    """The sink's bindings in render order."""
    yield from sink.entries
