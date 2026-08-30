"""A query you build up a step at a time.

A frame is a plan, and only a plan. It holds no connection and no database, so
it renders to SQL that will run wherever that SQL is valid, and the same plan
can be a macro body, a query on one connection, and a query on another.

    plan = table("people").filter(col("age") > 30).select(col("name"))
    plan.render()                 # SQL, no engine involved
    plan.rows(con)                # rows, from a connection you pass
    plan.on(con).rows()           # the same, with the connection filled in

A connection is an argument to the things that need one: resolving a schema,
running the query, writing the rows somewhere. Nothing else.

The plan is a graph rather than a tree, so a frame used in two places is
computed once. Column names are worked out here, because this library decided
them. Types, and any name the engine chose, are asked for.

Nothing a connection said is ever stored on a plan. A schema is one catalog's
answer at one moment, and a plan that remembered it would render one way here
and the same way somewhere the answer differs. Shapes are worked out afresh
against whichever connection is running. The one schema a caller states is
a `values()` plan's, which is a statement by the caller rather than a memory.
"""

from __future__ import annotations

import dataclasses
import html
import os
import re
from collections.abc import Iterable, Sized
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .connection import Connection, LiveResult
from .exceptions import Error
from .expr import (
    Col,
    Expr,
    Lit,
    ParamSink,
    PlanBase,
    Star,
    SubQuery,
    col,
    count_all,
    parameters_in,
    qualified,
    quote,
    render_literal,
    rendering_steps,
    subqueries,
    suspended_sinks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = ["Bound", "Column", "Frame", "NeedsConnection", "Step", "sql", "table", "values"]


class Column(NamedTuple):
    """One output column.

    `type` is None when the client knows the name but not the type: the name
    was decided here, the type was decided by an expression only the engine can
    resolve. Nothing guesses; an unknown type is asked for when it is needed.
    """

    name: str
    type: str | None = None


#: What a step produces, in order.
Shape = tuple[Column, ...]

#: How many stub answers one connection keeps. The keys are query text, so a
#: long-running process could otherwise accumulate them without bound.
_STUB_LIMIT = 512


def _stub_shape(connection: Connection, sql: str) -> Shape:
    """Bind a stub question, remembering the answer on the connection.

    A stub carries its whole input inline and names no table, so its answer
    does not depend on the catalog. It does depend on the engine's settings:
    `SELECT 7 / 2` binds as DOUBLE by default and as INTEGER under
    `integer_division`. Those belong to a connection, so the memory does too.
    """
    with connection._stub_lock:
        _forget_if_moved(connection)
        remembered = connection._stub_answers.get(sql)
    if remembered is not None:
        return cast("Shape", remembered)
    output, _ = connection._engine().bind(sql)
    shape = tuple(Column(name, type_text) for name, type_text in output)
    with connection._stub_lock:
        _forget_if_moved(connection)
        # The oldest answer goes, one at a time. Clearing everything at the
        # limit would throw away a working set at the busiest moment.
        while len(connection._stub_answers) >= _STUB_LIMIT:
            connection._stub_answers.pop(next(iter(connection._stub_answers)))
        connection._stub_answers[sql] = shape
    return shape


def _forget_if_moved(connection: Connection) -> None:
    """Drop the answers if any connection to this database has run a changing statement since."""
    current = connection._catalog.generation
    if connection._stub_generation != current:
        connection._stub_answers.clear()
        connection._stub_generation = current


def _completer(known: dict[int, Shape], connection: Connection | None) -> Callable[[Frame], Shape]:
    """A function giving a node's shape with every type known.

    Bottom-up: a step whose input carries an unknown type has that input
    completed first, so every question is one stub deep and nothing ever
    binds a whole chain. Completed shapes are written back into `known`,
    which lives for one resolution only.
    """

    def typed(node: Frame) -> Shape:
        shape = known[id(node)]
        if all(column.type is not None for column in shape):
            return shape
        if connection is None:
            message = "types are the engine's to say; pass a connection"
            raise NeedsConnection(message)
        answered = node._ask(connection, typed)
        # Names are ours, types are the engine's. Positions line up because
        # the engine answers the same select list this library rendered.
        if len(answered) == len(shape):
            answered = tuple(Column(ours.name, theirs.type) for ours, theirs in zip(shape, answered, strict=True))
        known[id(node)] = answered
        return answered

    return typed


def _uses_of(expressions: Iterable[Expr]) -> tuple[Frame, ...]:
    """The plans the expressions of a step refer to, each once, in order."""
    found: dict[int, Frame] = {}
    for expression in expressions:
        for plan in subqueries(expression):
            if isinstance(plan, Frame):
                found.setdefault(id(plan), plan)
    return tuple(found.values())


def _as_stub(shape: Shape) -> str:
    """A select list producing an empty relation of this shape."""
    return ", ".join(f"NULL::{column.type} AS {quote(column.name)}" for column in shape)


def _duplicates(names: list[str]) -> list[str]:
    """The names appearing more than once, in order of first appearance."""
    seen: dict[str, int] = {}
    for name in names:
        seen[name] = seen.get(name, 0) + 1
    return [name for name, count in seen.items() if count > 1]


def _require_unique(shape: Shape, verb: str) -> Shape:
    """Refuse a step that would produce one name twice.

    A frame's column names are its whole addressing scheme: `col(name)` in the
    next step has to mean exactly one thing. DuckDB will not catch this once
    the step is behind a WITH; it binds the later reference to whichever came
    first and answers.
    """
    repeated = _duplicates([column.name for column in shape])
    if repeated:
        listed = ", ".join(repr(name) for name in repeated)
        message = f"{verb} would produce {listed} more than once, and `col` could not tell them apart"
        raise ValueError(message)
    return shape


def _type_of(name: str, shape: Shape) -> str | None:
    """The type of a column in a shape, if it is known."""
    for column in shape:
        if column.name == name:
            return column.type
    return None


def _contributed(expression: Expr, source: Shape) -> list[Column] | None:
    """What one select-list expression adds, or None if the engine names it.

    A name the caller wrote is known here. A name DuckDB invents, such as the
    `(x * 2.5)` it gives an unaliased computation, is not, and that is what
    sends the step to the binder.
    """
    alias = expression._alias
    if isinstance(expression, Star):
        if alias:
            return None
        kept = [c for c in source if c.name not in set(expression.exclude)]
        return [Column(expression.rename.get(c.name, c.name), c.type) for c in kept]
    if isinstance(expression, Col):
        # A dotted reference names a side of a join; the column is the last part.
        bare = expression.name.rsplit(".", 1)[-1]
        return [Column(alias or bare, _type_of(bare, source))]
    if alias:
        # The caller named it, but an expression computed it, so the type is
        # the engine's to say.
        return [Column(alias, None)]
    return None


class JoinKind(NamedTuple):
    """What one kind of join renders as, and what it needs and keeps."""

    keyword: str
    needs_on: bool
    keeps_right: bool


#: The join kinds. Closed, so `how` can never carry text into the statement,
#: and one row per kind, so nothing about a kind is decided in two places.
_JOIN_KINDS = {
    "inner": JoinKind("INNER", needs_on=True, keeps_right=True),
    "left": JoinKind("LEFT", needs_on=True, keeps_right=True),
    "right": JoinKind("RIGHT", needs_on=True, keeps_right=True),
    "outer": JoinKind("FULL OUTER", needs_on=True, keeps_right=True),
    "full": JoinKind("FULL OUTER", needs_on=True, keeps_right=True),
    "semi": JoinKind("SEMI", needs_on=True, keeps_right=False),
    "anti": JoinKind("ANTI", needs_on=True, keeps_right=False),
    "cross": JoinKind("CROSS", needs_on=False, keeps_right=True),
    "positional": JoinKind("POSITIONAL", needs_on=False, keeps_right=True),
    "natural": JoinKind("NATURAL", needs_on=False, keeps_right=True),
    "asof": JoinKind("ASOF", needs_on=True, keeps_right=True),
}

_OPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _option(name: str) -> str:
    """A COPY option name. Checked, because it cannot be quoted."""
    if not _OPTION_NAME.match(name):
        message = f"not a copy option name: {name!r}"
        raise ValueError(message)
    return name.upper()


def _option_value(name: str, value: object) -> str:
    """A COPY option value, spelled as SQL spells it.

    An expression renders as itself, so `star()` is `*` and `col("x")` a
    name. A list or tuple is the parenthesised list COPY reads a column
    list from, `('grp', 'id')`. A dict is a struct. Anything else is a
    literal. Written into the text, because COPY takes no parameters; a
    `param()` in here has nothing to bind it, so it is refused.
    """
    if isinstance(value, Expr):
        held = parameters_in(value)
        if held:
            message = f"COPY takes no parameters; option {name!r} holds param({held[0]!r})"
            raise TypeError(message)
        return value.fragment()
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_option_value(name, v) for v in value) + ")"
    return render_literal(value)


def _options_clause(options: dict[str, object]) -> str:
    """The `(NAME value, ...)` of a COPY, or nothing when there are no options.

    Rendered with no sink active, so a literal is written into the text
    whatever is going on around it: COPY cannot bind one.
    """
    if not options:
        return ""
    with suspended_sinks():
        return " (" + ", ".join(f"{_option(k)} {_option_value(k, v)}" for k, v in options.items()) + ")"


def _as_expr(value: object) -> Expr:
    """One expression, for the verbs that take a single column at a time.

    Unlike `_as_exprs`, a list is not a sequence of arguments here: there is
    only one slot to fill, so a list can only have been meant as a value, and
    `lit` is how you say that.
    """
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return col(value)
    message = f"expected a column name or expression, got {value!r}; wrap a value in lit()"
    raise TypeError(message)


def _as_exprs(values: Iterable[object] | object) -> list[Expr]:
    """Accept an expression, a column name, or a sequence of either."""
    # Any iterable but text is a sequence of columns. A generator or a set is
    # as good as a list here, and a string is one name, not its characters.
    items = list(values) if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Expr)) else [values]
    out: list[Expr] = []
    for item in items:
        if isinstance(item, Expr):
            out.append(item)
        elif isinstance(item, str):
            # A bare string here is a column name, unlike inside an expression
            # where it is a value. The position decides, and there is nothing
            # else a string could usefully mean in a select list.
            out.append(col(item))
        else:
            message = f"expected a column name or expression, got {item!r}"
            raise TypeError(message)
    return out


class NeedsConnection(ValueError):
    """Working this out means asking the engine, and no connection was given."""


# --- steps: one record per verb, rendering as a function of it ---------------


@dataclasses.dataclass(frozen=True, eq=False)
class Step:
    """One verb of a plan and its arguments, as plain values.

    A step is data, not a closure: what it renders to, what it produces, and
    which expressions it holds are all functions of the record. That is what
    lets a plan be pickled, compared, and read.
    """

    def __post_init__(self) -> None:
        """Checks on the arguments. Steps with arguments to check override this."""

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore a pickled step and run the checks construction runs."""
        self.__dict__.update(state)
        self.__post_init__()

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        """The SQL of this step over its inputs' step names, given their shapes if known."""
        raise NotImplementedError

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        """What this step produces, from its inputs' shapes; None if only the engine knows."""
        return None

    def needs_shapes(self) -> bool:
        """Whether `render()` cannot do without its inputs' shapes."""
        return False

    def expressions(self) -> tuple[Expr, ...]:
        """Every expression this step holds."""
        return ()

    def __eq__(self, other: object) -> bool:
        """Same verb, same arguments. Expressions compare by what they render to.

        Written out because a generated dataclass `__eq__` would ask each
        expression `==`, which builds a comparison node instead of answering.
        """
        if type(other) is not type(self):
            return NotImplemented
        return _comparable(self) == _comparable(other)

    def __hash__(self) -> int:
        return hash(_comparable(self))


def _comparable(value: object) -> object:
    """A step's fields as plain values, expressions as their rendered text.

    Rendered into a sink, so the text carries `$n` for each literal and each
    `param()`, and the sink says which value or which name each one is.
    Rendered without one, every parameter would read as NULL and two plans
    bound to different parameters would compare equal.
    """
    if isinstance(value, Step):
        return (type(value).__name__, tuple(_comparable(getattr(value, f.name)) for f in dataclasses.fields(value)))
    if isinstance(value, Expr):
        with ParamSink() as sink:
            text = value.as_select()
        bound = tuple((kind, name if kind == "reference" else repr(name)) for kind, name, _ in sink.entries)
        return (text, bound)
    if isinstance(value, tuple):
        return tuple(_comparable(v) for v in value)
    return value


def _at_least_one(items: Sized, verb: str, what: str = "column") -> None:
    """Refuse a step given nothing to work on, which would render a dangling clause."""
    if not items:
        message = f"{verb} needs at least one {what}"
        raise TypeError(message)


@dataclasses.dataclass(frozen=True, eq=False)
class Table(Step):
    """A table or view, by name."""

    name: str

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        """Read the table; its name is quoted, so it cannot turn into more SQL."""
        return f"SELECT * FROM {qualified(self.name)}"


@dataclasses.dataclass(frozen=True, eq=False)
class Sql(Step):
    """A query, as SQL text."""

    text: str

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        """The text, as written."""
        return self.text


@dataclasses.dataclass(frozen=True, eq=False)
class Values(Step):
    """Rows given in memory."""

    rows: tuple[tuple[Expr, ...], ...]
    heading: Shape

    def __post_init__(self) -> None:
        _at_least_one(self.heading, "values")
        _require_unique(self.heading, "values")
        for row in self.rows:
            if len(row) != len(self.heading):
                message = f"a row has {len(row)} values for {len(self.heading)} columns"
                raise ValueError(message)
        if not self.rows and any(c.type is None for c in self.heading):
            message = "values with no rows needs a type for every column"
            raise ValueError(message)

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        if not self.rows:
            return f"SELECT {_as_stub(self.heading)} WHERE FALSE"
        rendered = ", ".join("(" + ", ".join(v.fragment() for v in row) + ")" for row in self.rows)
        columns = ", ".join(quote(c.name) for c in self.heading)
        # A type given is a cast, so the shape this step reports is what
        # the engine produces. A column given without one takes the type
        # the engine infers from the values.
        selected = ", ".join(
            f"{quote(c.name)}::{c.type} AS {quote(c.name)}" if c.type is not None else quote(c.name)
            for c in self.heading
        )
        return f'SELECT {selected} FROM (VALUES {rendered}) AS "values"({columns})'

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return self.heading

    def expressions(self) -> tuple[Expr, ...]:
        return tuple(v for row in self.rows for v in row)


@dataclasses.dataclass(frozen=True, eq=False)
class Filter(Step):
    predicate: Expr

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT * FROM {names[0]} WHERE {self.predicate.fragment()}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return shapes[0]

    def expressions(self) -> tuple[Expr, ...]:
        return (self.predicate,)


@dataclasses.dataclass(frozen=True, eq=False)
class Select(Step):
    columns: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _at_least_one(self.columns, "select")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT {', '.join(e.as_select() for e in self.columns)} FROM {names[0]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return _projected(self.columns, shapes[0], "select")

    def expressions(self) -> tuple[Expr, ...]:
        return self.columns


@dataclasses.dataclass(frozen=True, eq=False)
class WithColumns(Step):
    columns: tuple[tuple[str, Expr], ...]

    def __post_init__(self) -> None:
        _at_least_one(self.columns, "with_columns")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        source = names[0]
        if shapes[0] is None:
            # Nothing is known about the input, so the SQL cannot say which
            # names it replaces. This form needs no such knowledge: keep every
            # input column but the ones being set, then add those. The price
            # is order: a replaced column moves to the end.
            excluded = ", ".join(render_literal(name) for name, _ in self.columns)
            added = ", ".join(e.alias(n).as_select() for n, e in self.columns)
            return f"SELECT COLUMNS(lambda c: c NOT IN ({excluded})), {added} FROM {source}"
        existing = {column.name for column in shapes[0]}
        # A name already present is replaced in place, keeping column order; a
        # new one is appended. EXCLUDE cannot serve both, because excluding a
        # name that is not there is an error.
        replaced = [f"{e.fragment()} AS {quote(n)}" for n, e in self.columns if n in existing]
        appended = [e.alias(n).as_select() for n, e in self.columns if n not in existing]
        star = f"* REPLACE ({', '.join(replaced)})" if replaced else "*"
        return f"SELECT {', '.join([star, *appended])} FROM {source}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        setting = {name for name, _ in self.columns}
        existing = {column.name for column in shapes[0]}
        # A replaced column keeps its position, a new one is appended, and both
        # take their type from the engine because an expression made it.
        kept = [Column(c.name, None) if c.name in setting else c for c in shapes[0]]
        new = (Column(name, None) for name, _ in self.columns if name not in existing)
        return _require_unique((*kept, *new), "with_columns")

    def expressions(self) -> tuple[Expr, ...]:
        return tuple(e for _, e in self.columns)


@dataclasses.dataclass(frozen=True, eq=False)
class Drop(Step):
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        _at_least_one(self.names, "drop")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT * EXCLUDE ({', '.join(quote(c) for c in self.names)}) FROM {names[0]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        dropped = set(self.names)
        return tuple(c for c in shapes[0] if c.name not in dropped)


@dataclasses.dataclass(frozen=True, eq=False)
class Rename(Step):
    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _at_least_one(self.pairs, "rename")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        rendered = ", ".join(f"{quote(old)} AS {quote(new)}" for old, new in self.pairs)
        return f"SELECT * RENAME ({rendered}) FROM {names[0]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        renamed = dict(self.pairs)
        return _require_unique(tuple(Column(renamed.get(c.name, c.name), c.type) for c in shapes[0]), "rename")


@dataclasses.dataclass(frozen=True, eq=False)
class Sort(Step):
    keys: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _at_least_one(self.keys, "sort")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT * FROM {names[0]} ORDER BY {', '.join(e.as_order() for e in self.keys)}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return shapes[0]

    def expressions(self) -> tuple[Expr, ...]:
        return self.keys


@dataclasses.dataclass(frozen=True, eq=False)
class Limit(Step):
    count: int | None
    offset: int

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        clause = f" LIMIT {self.count}" if self.count is not None else ""
        clause += f" OFFSET {self.offset}" if self.offset else ""
        return f"SELECT * FROM {names[0]}{clause}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return shapes[0]


@dataclasses.dataclass(frozen=True, eq=False)
class Distinct(Step):
    keys: tuple[Expr, ...] | None

    def __post_init__(self) -> None:
        if self.keys is not None:
            _at_least_one(self.keys, "distinct(on=...)")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        if self.keys is None:
            return f"SELECT DISTINCT * FROM {names[0]}"
        return f"SELECT DISTINCT ON ({', '.join(e.fragment() for e in self.keys)}) * FROM {names[0]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return shapes[0]

    def expressions(self) -> tuple[Expr, ...]:
        return self.keys or ()


@dataclasses.dataclass(frozen=True, eq=False)
class Sample(Step):
    size: str
    arguments: str

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT * FROM {names[0]} USING SAMPLE {self.size} ({self.arguments})"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return shapes[0]


@dataclasses.dataclass(frozen=True, eq=False)
class Unnest(Step):
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        _at_least_one(self.columns, "unnest")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        expanded = ", ".join(f"unnest({quote(c)}) AS {quote(c)}" for c in self.columns)
        return f"SELECT * REPLACE ({expanded}) FROM {names[0]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        # Names and order are untouched; an opened column becomes its element
        # type, which only the engine knows.
        opened = set(self.columns)
        return tuple(Column(c.name, None) if c.name in opened else c for c in shapes[0])


@dataclasses.dataclass(frozen=True, eq=False)
class Unpivot(Step):
    columns: tuple[str, ...]
    name: str
    value: str

    def __post_init__(self) -> None:
        _at_least_one(self.columns, "unpivot")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        folded = ", ".join(quote(c) for c in self.columns)
        return f"UNPIVOT (SELECT * FROM {names[0]}) ON {folded} INTO NAME {quote(self.name)} VALUE {quote(self.value)}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        folded = set(self.columns)
        kept = (c for c in shapes[0] if c.name not in folded)
        return _require_unique((*kept, Column(self.name, "VARCHAR"), Column(self.value, None)), "unpivot")


@dataclasses.dataclass(frozen=True, eq=False)
class Aggregate(Step):
    keys: tuple[Expr, ...]
    aggregates: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _at_least_one((*self.keys, *self.aggregates), "aggregate", "aggregate or group key")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        selected = [k.as_select() for k in self.keys] + [e.as_select() for e in self.aggregates]
        clause = " GROUP BY " + ", ".join(k.fragment() for k in self.keys) if self.keys else ""
        return f"SELECT {', '.join(selected)} FROM {names[0]}{clause}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return _projected((*self.keys, *self.aggregates), shapes[0], "aggregate")

    def expressions(self) -> tuple[Expr, ...]:
        return (*self.keys, *self.aggregates)


@dataclasses.dataclass(frozen=True, eq=False)
class Describe(Step):
    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        # Wrapped in a SELECT because a bare SUMMARIZE cannot follow a WITH,
        # and every step but the first is reached through one.
        return f"SELECT * FROM (SUMMARIZE SELECT * FROM {names[0]})"


@dataclasses.dataclass(frozen=True, eq=False)
class Join(Step):
    """A join of two inputs. The sides are named `l` and `r` in the SQL."""

    how: str
    on: Expr | None
    using: tuple[str, ...]
    suffix: str | None

    def __post_init__(self) -> None:
        if self.how not in _JOIN_KINDS:
            message = f"unknown join kind {self.how!r}; one of {', '.join(sorted(_JOIN_KINDS))}"
            raise ValueError(message)
        if self._kind().needs_on and self.on is None and not self.using:
            message = f"a {self.how} join needs `on`"
            raise TypeError(message)

    def _kind(self) -> JoinKind:
        return _JOIN_KINDS[self.how]

    def needs_shapes(self) -> bool:
        return self.suffix is not None and self._kind().keeps_right

    def _clashing(self, left: Shape, right: Shape) -> list[str]:
        """Right-hand names the left already carries; USING keys fold and do not clash."""
        left_names = {column.name for column in left}
        folded = set(self.using)
        return [c.name for c in right if c.name in left_names and c.name not in folded]

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        kind = self._kind()
        keys = [quote(c) for c in self.using]
        if not kind.needs_on:
            clause = ""
        elif self.on is not None:
            clause = f" ON {self.on.fragment()}"
        else:
            clause = " USING (" + ", ".join(keys) + ")"
        left_shape, right_shape = shapes
        if kind.keeps_right and self.suffix is not None and (left_shape is None or right_shape is None):
            message = (
                "a join with a suffix renames the right side's clashing columns, and which ones "
                "clash depends on both sides' columns; this needs a connection to render"
            )
            raise NeedsConnection(message)
        shared = (
            self._clashing(left_shape, right_shape)
            if kind.keeps_right and left_shape is not None and right_shape is not None
            else []
        )
        projection = "*"
        if shared:
            # Only when renaming: the plain star keeps the SQL closest to what
            # the reader wrote, and USING's own folding intact.
            parts = [f"{quote(n)} AS {quote(n + str(self.suffix))}" for n in shared]
            excluded = f" EXCLUDE ({', '.join(keys)})" if keys else ""
            projection = f"l.*, r.*{excluded} RENAME ({', '.join(parts)})"
        # Both sides are aliased so joining a frame to itself works, and so an
        # ON expression can tell the sides apart, as col("l.id").
        return f"SELECT {projection} FROM {names[0]} AS l {kind.keyword} JOIN {names[1]} AS r{clause}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        left_shape, right_shape = shapes
        if not self._kind().keeps_right:
            return left_shape
        shared = self._clashing(left_shape, right_shape)
        if shared and self.suffix is None:
            listed = ", ".join(repr(name) for name in shared)
            message = (
                f"both sides of this join have {listed}; the result would carry each name "
                f"twice and `col` could not tell them apart. Pass suffix=... to rename the "
                f"right side, or rename before joining."
            )
            raise ValueError(message)
        renamed = {name: name + str(self.suffix) for name in shared}
        folded = set(self.using)
        carried = [Column(renamed.get(c.name, c.name), c.type) for c in right_shape if c.name not in folded]
        # The renamed copies have to be free too: suffixing onto a name the
        # left already holds only moves the clash.
        return _require_unique((*left_shape, *carried), "join")

    def expressions(self) -> tuple[Expr, ...]:
        return (self.on,) if self.on is not None else ()


@dataclasses.dataclass(frozen=True, eq=False)
class SetOp(Step):
    keyword: str
    by_name: bool = False

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT * FROM {names[0]} {self.keyword} SELECT * FROM {names[1]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        # By name, a column only on the right is added, so the engine decides.
        return None if self.by_name else shapes[0]


def _projected(chosen: tuple[Expr, ...], source: Shape, verb: str) -> Shape | None:
    """Shape of a select list, or None where the engine names a column."""
    out: list[Column] = []
    for expression in chosen:
        columns = _contributed(expression, source)
        if columns is None:
            return None
        out.extend(columns)
    return _require_unique(tuple(out), verb)


class Frame(PlanBase):
    """One step of a query. Immutable: every verb returns a new frame."""

    def __init__(self, step: Step, inputs: tuple[Frame, ...] = ()) -> None:
        #: What this step is: its verb and arguments, as data. Rendering,
        #: shape, and the expressions held are all functions of it, and its
        #: expressions render only when the plan does, inside the parameter
        #: sink, so their literals are bound rather than written into the SQL.
        self._step = step
        self._inputs = inputs
        #: Plans this step refers to through subqueries in its expressions.
        #: Part of the graph, so they render once as steps of their own, but
        #: not named to the step: an expression finds its plan by identity.
        self._uses = _uses_of(step.expressions())

    @property
    def step(self) -> Step:
        """This step's verb and arguments."""
        return self._step

    @property
    def inputs(self) -> tuple[Frame, ...]:
        """The plans this step reads."""
        return self._inputs

    # -- graph and rendering

    def _order(self) -> list[Frame]:
        """Every node this frame depends on, inputs before the nodes using them."""
        # An explicit stack rather than recursion: a chain built in a loop can
        # be thousands of steps deep, and every render and schema lookup comes
        # through here.
        seen: set[int] = set()
        order: list[Frame] = []
        stack: list[tuple[Frame, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            # Reversed, so the inputs are visited left to right as they would
            # be by recursion. Step numbering and parameter order follow this.
            stack.extend((parent, False) for parent in reversed(node._inputs + node._uses))
        return order

    def render(self, connection: Connection | None = None) -> str:
        """The whole query as SQL, with one CTE per step.

        With a connection, this is the text execution would use on it. Without
        one, it is a pure function of the plan, and two things follow. A step
        whose SQL depends on its input's columns renders a form that does not
        need them, and that form can order columns differently: `with_columns`
        moves a replaced column to the end, where the executed form keeps it in
        place. And nothing is checked: two columns of one name in a join, or a
        table that does not exist, are found when the plan runs, not here.

        A frame used twice appears once: the graph is walked by identity, so
        the engine sees the reuse rather than a duplicated subtree.
        """
        order = self._order()
        shapes = self._shapes(connection, order) if connection is not None else None
        return self._render(shapes, order)

    def _render(self, shapes: dict[int, Shape] | None = None, order: list[Frame] | None = None) -> str:
        """The SQL, given whatever shapes are known."""
        order = self._order() if order is None else order
        names = {id(node): quote(f"_s{i}") for i, node in enumerate(order)}

        def body_of(node: Frame) -> str:
            # Shapes are handed down, never read off the node. A step that
            # needs to know its input's columns says so by using them.
            given = tuple((shapes or {}).get(id(parent)) for parent in node._inputs)
            return node._step.render(tuple(names[id(p)] for p in node._inputs), given)

        with rendering_steps(names):
            if len(order) == 1:
                return body_of(self)
            ctes = ",\n".join(f"{names[id(n)]} AS (\n{body_of(n)}\n)" for n in order[:-1])
            return f"WITH {ctes}\n{body_of(order[-1])}"

    def _derive(self, step: Step, *inputs: Frame) -> Frame:
        return Frame(step, inputs or (self,))

    def _definition(self, connection: Connection) -> str:
        """The SQL for a macro body, resolved only as far as rendering needs.

        A body may name a macro parameter, which the engine binds only once
        the macro exists, so a step referring to one cannot be asked about.
        The steps that must know their inputs' columns, a suffixed join, get
        those inputs resolved; everything else renders as it would blind.
        """
        order = self._order()
        needed: set[int] = set()
        for node in order:
            if node._step.needs_shapes():
                stack = list(node._inputs)
                while stack:
                    parent = stack.pop()
                    if id(parent) not in needed:
                        needed.add(id(parent))
                        stack.extend(parent._inputs + parent._uses)
        shapes = self._shapes(connection, order, needed) if needed else None
        return self._render(shapes, order)

    # -- shape: names derived here, types from the binder

    def _shapes(
        self,
        connection: Connection | None = None,
        order: list[Frame] | None = None,
        only: set[int] | None = None,
    ) -> dict[int, Shape]:
        """The shape of every step, worked out fresh, keyed by identity.

        Returned rather than stored. Which columns a source has is a fact
        about one catalog at one moment: it changes with the connection and it
        changes under DDL, so a plan that kept it would render by whatever it
        was told first.

        `only` restricts the work to those steps, which must be closed under
        inputs and uses: a macro body resolves just what its rendering needs.
        """
        known: dict[int, Shape] = {}
        typed = _completer(known, connection)
        for node in self._order() if order is None else order:
            if only is not None and id(node) not in only:
                continue
            given = tuple(known[id(parent)] for parent in node._inputs)
            shape = node._step.shape(given)
            if shape is None:
                if connection is None:
                    message = (
                        "which columns this step produces is the engine's to say; pass a connection "
                        "(only a plan built from values() can answer without one)"
                    )
                    raise NeedsConnection(message)
                shape = node._ask(connection, typed)
            known[id(node)] = shape
        return known

    def _ask(self, connection: Connection, typed: Callable[[Frame], Shape]) -> Shape:
        """Ask the engine about this step alone.

        A step with inputs is asked over empty tables of the right shape, so
        the question stays small however long the chain is, and names no
        catalog object. That makes its answer a function of the text, so it is
        remembered on the connection. A source does name the catalog, and its
        answer is only true of the connection that gave it, so it is not.
        """
        stubs: list[str] = []
        names: dict[int, str] = {}
        given: list[Shape] = []
        for position, parent in enumerate(self._inputs):
            shape = typed(parent)
            given.append(shape)
            stub = quote(f"_in{position}")
            stubs.append(f"{stub} AS (SELECT {_as_stub(shape)} WHERE FALSE)")
            names[id(parent)] = stub
        for position, used in enumerate(self._uses):
            # A subquery's plan is stubbed too, so the question stays free of
            # the catalog even when an expression refers to another plan.
            stub = quote(f"_use{position}")
            stubs.append(f"{stub} AS (SELECT {_as_stub(typed(used))} WHERE FALSE)")
            names[id(used)] = stub
        with suspended_sinks(), rendering_steps(names):
            body = self._step.render(tuple(names[id(p)] for p in self._inputs), tuple(given))
        if not stubs:
            # A source. Its answer belongs to this catalog, so it is asked
            # every time rather than remembered.
            output, _ = connection._engine().bind(body)
            return tuple(Column(name, type_text) for name, type_text in output)
        return _stub_shape(connection, f"WITH {', '.join(stubs)}\n{body}")

    def resolve(self, connection: Connection | None = None) -> Shape:
        """What this step produces, in order.

        Names are derived wherever this library decided them, which is most
        verbs. Where DuckDB decides a name, such as the `(x * 2.5)` it invents
        for an unaliased computation, the engine is asked. Types are left
        unknown until something asks, because nothing about rendering needs
        them.
        """
        return self._shapes(connection)[id(self)]

    def schema(self, connection: Connection | None = None) -> list[tuple[str, str]]:
        """The columns this frame produces, as (name, type).

        Types are the engine's to say, so this asks it if it has not already.
        `.columns` alone usually costs nothing.
        """
        shapes = self._shapes(connection)
        # Complete this step the way an input is completed for a step above
        # it: one stub, over typed inputs.
        shape = _completer(shapes, connection)(self)
        unresolved = [column.name for column in shape if column.type is None]
        if unresolved:  # pragma: no cover - the binder types everything it returns
            message = f"no type reported for {unresolved}"
            raise Error(message)
        return [(column.name, type_text) for column in shape if (type_text := column.type) is not None]

    def columns(self, connection: Connection | None = None) -> list[str]:
        """The column names.

        Worked out here wherever this library decided them, so a connection is
        only needed when the engine named something, or to read a source.
        """
        return [column.name for column in self.resolve(connection)]

    def types(self, connection: Connection | None = None) -> list[str]:
        """The column types. The engine's to say, so it gets asked."""
        return [type_text for _, type_text in self.schema(connection)]

    def describe(self) -> Frame:
        """Per-column statistics, as a frame: count, min, max, average, quartiles.

        This reads the rows, unlike `.schema`, which only asks the binder.
        """
        return self._derive(Describe())

    # -- verbs

    def filter(self, predicate: Expr) -> Frame:
        """Keep the rows where `predicate` holds.

        Takes an expression. For a condition written as SQL, wrap it:
        `filter(sql_expr("..."))`, so that every place raw text enters a
        query says so at the call.
        """
        if not isinstance(predicate, Expr):  # the annotation says Expr; callers do not always listen
            message = (  # type: ignore[unreachable]
                f"filter takes an expression, not {type(predicate).__name__}; "
                f"for a condition written as SQL, use filter(sql_expr(...))"
            )
            raise TypeError(message)
        return self._derive(Filter(predicate))

    def select(self, *columns: object) -> Frame:
        """Keep only these columns or expressions.

        Pure projection: it does not group, unlike the older client's `select`.
        """
        return self._derive(Select(tuple(_as_exprs(list(columns)))))

    def with_columns(self, **columns: object) -> Frame:
        """Add columns, replacing any of the same name."""
        return self._derive(WithColumns(tuple((name, _as_expr(value)) for name, value in columns.items())))

    def drop(self, *columns: str) -> Frame:
        """Remove columns."""
        return self._derive(Drop(tuple(columns)))

    def rename(self, **columns: str) -> Frame:
        """Rename columns, given as old=new."""
        return self._derive(Rename(tuple(columns.items())))

    def sort(self, *columns: object) -> Frame:
        """Order the rows. Use `.desc()` and `.nulls_last()` on the columns."""
        return self._derive(Sort(tuple(_as_exprs(list(columns)))))

    def limit(self, count: int, offset: int = 0) -> Frame:
        """Keep at most `count` rows, optionally skipping `offset` first."""
        return self._derive(Limit(int(count), int(offset)))

    def head(self, count: int = 5) -> Frame:
        """The first `count` rows."""
        return self.limit(count)

    def offset(self, count: int) -> Frame:
        """Skip `count` rows."""
        return self._derive(Limit(None, int(count)))

    def distinct(self, on: Iterable[object] | object | None = None) -> Frame:
        """Remove duplicate rows, or duplicates of `on` only."""
        return self._derive(Distinct(None if on is None else tuple(_as_exprs(on))))

    def sample(
        self,
        n: int | None = None,
        *,
        percent: float | None = None,
        seed: int | None = None,
        method: str | None = None,
    ) -> Frame:
        """A random subset: `n` rows, or `percent` of them.

        Give a `seed` to get the same rows every run. `method` is reservoir,
        bernoulli or system; the default suits whichever size you asked for.
        """
        if (n is None) == (percent is None):
            message = "sample takes either n or percent"
            raise TypeError(message)
        # Bernoulli and system sample each row independently, so they cannot
        # hit an exact count. Only reservoir can, so that is the default for n.
        size = f"{int(n)} ROWS" if n is not None else f"{float(percent or 0)} PERCENT"
        chosen = _option(method) if method else ("RESERVOIR" if n is not None else "BERNOULLI")
        arguments = f"{chosen}, {int(seed)}" if seed is not None else chosen
        return self._derive(Sample(size, arguments))

    def unnest(self, *columns: str) -> Frame:
        """Expand list columns to one row per element, repeating the rest.

        Unnesting two columns together walks them in step rather than making
        every pairing; the shorter one runs out as NULL.
        """
        return self._derive(Unnest(tuple(columns)))

    def unpivot(self, *columns: str, name: str = "name", value: str = "value") -> Frame:
        """Turn columns into rows, one row per column named.

        The columns not named are kept and repeat down the rows. `name` and
        `value` name the two columns that replace the ones folded away.
        """
        return self._derive(Unpivot(tuple(columns), name, value))

    def aggregate(self, *aggregates: object, group_by: Iterable[object] | object | None = None) -> Frame:
        """Group and aggregate. Group keys come first in the output."""
        # `is None`, not truthiness: an expression refuses to be a condition.
        keys = () if group_by is None else tuple(_as_exprs(group_by))
        return self._derive(Aggregate(keys, tuple(_as_exprs(list(aggregates)))))

    def group_by(self, *columns: object) -> GroupedFrame:
        """Begin a grouped aggregation. Continue with `.agg(...)`."""
        return GroupedFrame(self, _as_exprs(list(columns)))

    def join(
        self,
        other: Frame,
        on: Expr | str | Iterable[str] | None = None,
        how: str = "inner",
        suffix: str | None = None,
    ) -> Frame:
        """Join to another frame.

        `on` is a column name, a list of them for a USING join, or an
        expression for an ON join. Inside that expression the two sides are
        `l` and `r`, so `col("l.id") == col("r.order_id")` is unambiguous even
        when both carry the same column name.

        `how` is inner, left, right, outer, semi, anti, cross, positional,
        natural or asof.

        A join carries every column of both sides through, so two sides sharing
        a name would leave the result with that name twice. A later `col(name)`
        would then silently mean whichever came first. That is refused when the
        plan is resolved or run; `render()` with no connection cannot know the
        sides' columns and renders the join unchecked. Pass `suffix` to rename
        the right side's copies instead.
        """
        using: tuple[str, ...] = ()
        if isinstance(on, str):
            using = (on,)
        elif on is not None and not isinstance(on, Expr):
            using = tuple(str(c) for c in on)
        return Frame(Join(how.lower(), on if isinstance(on, Expr) else None, using, suffix), (self, other))

    def cross(self, other: Frame) -> Frame:
        """Every combination of rows from both frames."""
        return self.join(other, how="cross")

    def union(self, other: Frame, *, all: bool = True) -> Frame:
        """Rows from both frames, keeping duplicates unless `all` is false."""
        return Frame(SetOp("UNION ALL" if all else "UNION"), (self, other))

    def union_by_name(self, other: Frame, *, all: bool = True) -> Frame:
        """Union, matching columns by name rather than position."""
        return Frame(SetOp("UNION ALL BY NAME" if all else "UNION BY NAME", by_name=True), (self, other))

    def intersect(self, other: Frame) -> Frame:
        """Rows present in both frames."""
        return Frame(SetOp("INTERSECT"), (self, other))

    def except_(self, other: Frame) -> Frame:
        """Rows in this frame and not the other."""
        return Frame(SetOp("EXCEPT"), (self, other))

    def __getitem__(self, name: object) -> Frame:
        """A single column, as a plan.

        Takes `object` rather than `str` because Python will pass integers
        here on its own if it decides a plan is a sequence.
        """
        if not isinstance(name, str):
            # Without this, Python's old sequence protocol would take
            # `for row in plan` to mean plan[0], plan[1], ... and build plans
            # forever, because __getitem__ never runs out.
            message = f"a plan is indexed by column name, not by {type(name).__name__}"
            raise TypeError(message)
        return self.select(col(name))

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Refuse to iterate without a connection.

        Defined only to say so: leaving it out would let the old sequence
        protocol fall back to __getitem__ and loop forever.
        """
        message = "iterating a plan needs a connection; use rows(connection)"
        raise TypeError(message)

    # -- as a value inside another query

    def scalar(self) -> Expr:
        """This query used where a single value is expected.

        It must return one row and one column. It is not correlated: it cannot
        see the columns of the query it lands in. It becomes a step of the plan
        it lands in, one CTE however many times it is used; that it is then
        computed once is the engine's treatment of a CTE used more than once,
        which a test holds it to.

            budget = totals.aggregate(col("amount").mean().alias("m"))
            orders.filter(col("amount") > budget.scalar())
        """
        return SubQuery(self)

    # -- execution

    def _sql_and_values(
        self,
        wrap: Callable[[str], str] | None = None,
        connection: Connection | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> tuple[str, list[Any] | None]:
        """The SQL for this frame and the values it binds.

        With a connection, the text execution uses on it. Rendered inside a
        sink, so lifted literals are bound as `$n` instead of being written
        into the text. A `param(name)` takes its value from `parameters`;
        every name must be supplied, and every name supplied must be used, so
        a typo cannot pass silently either way.
        """
        order = self._order()
        shapes = self._resolution(connection, order) if connection is not None else None
        with ParamSink() as sink:
            sql = self._render(shapes, order)
            if wrap is not None:
                sql = wrap(sql)
        supplied = dict(parameters or {})
        values: list[Any] = []
        used: set[str] = set()
        missing: list[str] = []
        for kind, value, _ in sink.entries:
            if kind == "literal":
                values.append(value)
            elif value in supplied:
                values.append(supplied[value])
                used.add(value)
            else:
                missing.append(value)
        if missing:
            listed = ", ".join(repr(name) for name in dict.fromkeys(missing))
            message = f"no value for parameter {listed}; pass parameters={{name: value}}"
            raise ValueError(message)
        unused = sorted(set(supplied) - used)
        if unused:
            message = f"parameters {', '.join(repr(n) for n in unused)} are not used by this plan"
            raise ValueError(message)
        return sql, values or None

    @staticmethod
    def _on(connection: object) -> Connection:
        """The connection a plan runs on, refusing anything else.

        A `dbapi.Connection` looks the part: it has an engine handle too. But
        its cursors account for the one live result per connection, and a plan
        run around them would leave that accounting wrong.
        """
        if not isinstance(connection, Connection):
            message = (
                f"a plan runs on a duckdb.Connection, not {type(connection).__name__}; "
                f"to run it through the DB-API, execute plan.render() on a cursor"
            )
            raise TypeError(message)
        return connection

    def _resolution(self, connection: Connection, order: list[Frame]) -> dict[int, Shape] | None:
        """The shapes execution renders with, worked out on this connection.

        Every time, because it is what tells a join which columns clash and
        what refuses a step that would produce one name twice, and both
        answers belong to the catalog being read. A lone source is the one
        plan that needs none: its text is the query, and some statements the
        engine will run it cannot describe in advance (PIVOT is one).
        """
        return None if not self._inputs and not self._uses else self._shapes(connection, order)

    def _execute(self, connection: Connection, parameters: Mapping[str, object] | None) -> LiveResult:
        connection = self._on(connection)
        sql, values = self._sql_and_values(connection=connection, parameters=parameters)
        return connection._execute(sql, values)

    def _run(self, connection: Connection, wrap: Callable[[str], str], parameters: Mapping[str, object] | None) -> int:
        """Run a statement built around this frame, reporting rows changed."""
        connection = self._on(connection)
        sql, values = self._sql_and_values(wrap, connection, parameters)
        with connection._execute(sql, values) as result:
            return result.drain()

    def rows(self, connection: Connection, *, parameters: Mapping[str, object] | None = None) -> list[tuple[Any, ...]]:
        """Every row, as tuples."""
        with self._execute(connection, parameters) as result:
            return result.fetch_all()

    def first(
        self, connection: Connection, *, parameters: Mapping[str, object] | None = None
    ) -> tuple[Any, ...] | None:
        """The first row, or None if there are none."""
        with self._execute(connection, parameters) as result:
            rows = result.fetch_rows(1)
        return rows[0] if rows else None

    def iter_rows(
        self, connection: Connection, *, parameters: Mapping[str, object] | None = None
    ) -> Iterator[tuple[Any, ...]]:
        """Every row, a batch at a time."""
        with self._execute(connection, parameters) as result:
            while batch := result.fetch_rows(1024):
                yield from batch

    def to_dicts(
        self, connection: Connection, *, parameters: Mapping[str, object] | None = None
    ) -> list[dict[str, Any]]:
        """Every row, as a dict keyed by column name."""
        names = self.columns(connection)
        return [dict(zip(names, row, strict=True)) for row in self.rows(connection, parameters=parameters)]

    def on(self, connection: Connection) -> Bound:
        """This plan on a connection, so the terminals take no argument.

        `plan.on(con).rows()` reads the way a dataframe does; the plan itself
        still holds no connection, and the same plan can be put on another.
        """
        return Bound(self, self._on(connection))

    def count(self, connection: Connection, *, parameters: Mapping[str, object] | None = None) -> int:
        """How many rows this plan produces. Runs a count."""
        counted = self._derive(Aggregate((), (count_all().alias("count"),)))
        row = counted.first(connection, parameters=parameters)
        return int(row[0]) if row else 0

    # -- writing the rows somewhere

    def create(
        self,
        connection: Connection,
        name: str,
        *,
        replace: bool = False,
        temporary: bool = False,
        parameters: Mapping[str, object] | None = None,
    ) -> int:
        """Store the rows in a new table. Returns how many were written."""
        prefix = "CREATE OR REPLACE" if replace else "CREATE"
        kind = "TEMPORARY TABLE" if temporary else "TABLE"
        return self._run(connection, lambda q: f"{prefix} {kind} {qualified(name)} AS {q}", parameters)

    def insert_into(self, connection: Connection, name: str, *, parameters: Mapping[str, object] | None = None) -> int:
        """Append the rows to a table that exists. Returns how many were added."""
        return self._run(connection, lambda q: f"INSERT INTO {qualified(name)} {q}", parameters)

    def copy_to(
        self,
        connection: Connection,
        path: str | os.PathLike[str],
        *,
        parameters: Mapping[str, object] | None = None,
        **options: object,
    ) -> list[tuple[Any, ...]]:
        """`COPY (this plan) TO path (options)`, returning the rows the statement returns.

        The options are COPY's own, under their SQL names, `format` among
        them: `copy_to(con, "x.parquet", format="parquet", compression="zstd")`;
        with none, the engine picks the format from the path's extension.
        A value is written as SQL writes it: a list is a column list, so
        `partition_by=["grp", "id"]`; `star()` is `*`, so
        `force_quote=star()`; a dict is a struct, as `kv_metadata` takes.

        What comes back is what COPY returns: one row with the count, or
        with `return_files=True` the count and the files, or with
        `return_stats=True` one row per file written.
        """
        clause = _options_clause(options)
        target = render_literal(os.fspath(path))
        connection = self._on(connection)
        sql, values = self._sql_and_values(lambda q: f"COPY ({q}) TO {target}{clause}", connection, parameters)
        with connection._execute(sql, values) as result:
            return result.fetch_all()

    def to_parquet(
        self,
        connection: Connection,
        path: str | os.PathLike[str],
        *,
        parameters: Mapping[str, object] | None = None,
        **options: object,
    ) -> list[tuple[Any, ...]]:
        """Write a Parquet file: `copy_to` with `format="parquet"`."""
        return self.copy_to(connection, path, parameters=parameters, format="parquet", **options)

    def to_csv(
        self,
        connection: Connection,
        path: str | os.PathLike[str],
        *,
        parameters: Mapping[str, object] | None = None,
        **options: object,
    ) -> list[tuple[Any, ...]]:
        """Write a CSV file: `copy_to` with `format="csv"`."""
        return self.copy_to(connection, path, parameters=parameters, format="csv", **options)

    def to_json(
        self,
        connection: Connection,
        path: str | os.PathLike[str],
        *,
        parameters: Mapping[str, object] | None = None,
        **options: object,
    ) -> list[tuple[Any, ...]]:
        """Write newline-delimited JSON, or one array with `array=True`: `copy_to` with `format="json"`."""
        return self.copy_to(connection, path, parameters=parameters, format="json", **options)

    # -- looking at it

    def explain(
        self, connection: Connection, *, analyze: bool = False, parameters: Mapping[str, object] | None = None
    ) -> str:
        """The query plan, as text.

        With `analyze` the query runs and the plan carries what each step cost.
        """
        connection = self._on(connection)
        keyword = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
        sql, values = self._sql_and_values(lambda q: f"{keyword} {q}", connection, parameters)
        with connection._execute(sql, values) as result:
            rows = result.fetch_all()
        return str(rows[0][1]) if rows else ""

    def show(self, connection: Connection, limit: int = 10, *, parameters: Mapping[str, object] | None = None) -> None:
        """Print the first rows."""
        print(self.preview(connection, limit, parameters=parameters))

    def preview(
        self, connection: Connection, limit: int = 10, *, parameters: Mapping[str, object] | None = None
    ) -> str:
        """The first rows drawn as a table. What `show` prints."""
        # One row past the limit, so the footer can say there are more without
        # counting them all.
        head = self.limit(limit + 1)
        rows = head.rows(connection, parameters=parameters)
        return _box(head.columns(connection), head.types(connection), rows[:limit], more=len(rows) > limit)

    def __repr__(self) -> str:
        """The SQL. A plan holds no connection, so there are no rows to show."""
        try:
            rendered = self.render()
        except NeedsConnection as reason:
            # The one step that cannot render blind is a suffixed join. A repr
            # that raised would make a debugger useless, so say why instead.
            return f"<Frame, renders with a connection: {reason}>"
        first = rendered.splitlines()
        shown = first[0] if len(first) == 1 else f"{first[0]} ... ({len(first)} lines)"
        return f"<Frame {shown}>"


class Bound:
    """A plan on a connection: every terminal, with the connection filled in.

    Built by `plan.on(con)`. Holds the plan and the connection and nothing
    else; the plan is unchanged and still runs anywhere.
    """

    __slots__ = ("connection", "plan")

    def __init__(self, plan: Frame, connection: Connection) -> None:
        self.plan = plan
        self.connection = connection

    def rows(self, *, parameters: Mapping[str, object] | None = None) -> list[tuple[Any, ...]]:
        """Every row, as tuples."""
        return self.plan.rows(self.connection, parameters=parameters)

    def first(self, *, parameters: Mapping[str, object] | None = None) -> tuple[Any, ...] | None:
        """The first row, or None if there are none."""
        return self.plan.first(self.connection, parameters=parameters)

    def iter_rows(self, *, parameters: Mapping[str, object] | None = None) -> Iterator[tuple[Any, ...]]:
        """Every row, a batch at a time."""
        return self.plan.iter_rows(self.connection, parameters=parameters)

    def to_dicts(self, *, parameters: Mapping[str, object] | None = None) -> list[dict[str, Any]]:
        """Every row, as a dict keyed by column name."""
        return self.plan.to_dicts(self.connection, parameters=parameters)

    def count(self, *, parameters: Mapping[str, object] | None = None) -> int:
        """How many rows the plan produces."""
        return self.plan.count(self.connection, parameters=parameters)

    def columns(self) -> list[str]:
        """The column names."""
        return self.plan.columns(self.connection)

    def types(self) -> list[str]:
        """The column types."""
        return self.plan.types(self.connection)

    def schema(self) -> list[tuple[str, str]]:
        """The columns as (name, type)."""
        return self.plan.schema(self.connection)

    def resolve(self) -> Shape:
        """The columns as `Column` records."""
        return self.plan.resolve(self.connection)

    def render(self) -> str:
        """The SQL as it will run on this connection."""
        return self.plan.render(self.connection)

    def explain(self, *, analyze: bool = False, parameters: Mapping[str, object] | None = None) -> str:
        """The query plan, as text."""
        return self.plan.explain(self.connection, analyze=analyze, parameters=parameters)

    def show(self, limit: int = 10, *, parameters: Mapping[str, object] | None = None) -> None:
        """Print the first rows."""
        self.plan.show(self.connection, limit, parameters=parameters)

    def preview(self, limit: int = 10, *, parameters: Mapping[str, object] | None = None) -> str:
        """The first rows drawn as a table."""
        return self.plan.preview(self.connection, limit, parameters=parameters)

    def create(
        self,
        name: str,
        *,
        replace: bool = False,
        temporary: bool = False,
        parameters: Mapping[str, object] | None = None,
    ) -> int:
        """Store the rows in a new table."""
        return self.plan.create(self.connection, name, replace=replace, temporary=temporary, parameters=parameters)

    def insert_into(self, name: str, *, parameters: Mapping[str, object] | None = None) -> int:
        """Append the rows to a table that exists."""
        return self.plan.insert_into(self.connection, name, parameters=parameters)

    def copy_to(
        self, path: str | os.PathLike[str], *, parameters: Mapping[str, object] | None = None, **options: object
    ) -> list[tuple[Any, ...]]:
        """`COPY (this plan) TO path (options)`, returning what the statement returns."""
        return self.plan.copy_to(self.connection, path, parameters=parameters, **options)

    def to_parquet(
        self, path: str | os.PathLike[str], *, parameters: Mapping[str, object] | None = None, **options: object
    ) -> list[tuple[Any, ...]]:
        """Write a Parquet file."""
        return self.plan.to_parquet(self.connection, path, parameters=parameters, **options)

    def to_csv(
        self, path: str | os.PathLike[str], *, parameters: Mapping[str, object] | None = None, **options: object
    ) -> list[tuple[Any, ...]]:
        """Write a CSV file."""
        return self.plan.to_csv(self.connection, path, parameters=parameters, **options)

    def to_json(
        self, path: str | os.PathLike[str], *, parameters: Mapping[str, object] | None = None, **options: object
    ) -> list[tuple[Any, ...]]:
        """Write JSON."""
        return self.plan.to_json(self.connection, path, parameters=parameters, **options)

    def __repr__(self) -> str:
        try:
            return self.preview()
        except (ValueError, Error) as reason:
            # A debugger pane or a notebook shows this unasked, so a plan
            # that cannot run here is described rather than raised.
            return f"<Bound {self.plan!r}, does not run here: {reason}>"

    def _repr_html_(self) -> str:
        """The first rows as an HTML table, for notebooks."""
        try:
            return self._html_table()
        except (ValueError, Error) as reason:
            return f"<pre>{html.escape(repr(self.plan))}\ndoes not run here: {html.escape(str(reason))}</pre>"

    def _html_table(self) -> str:
        head = self.plan.limit(11)
        rows = head.rows(self.connection)
        names, types = head.columns(self.connection), head.types(self.connection)
        cells = "".join(
            f"<th>{html.escape(n)}<br><small>{html.escape(t)}</small></th>" for n, t in zip(names, types, strict=True)
        )
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(_cell(v))}</td>" for v in row) + "</tr>" for row in rows[:10]
        )
        more = "<p>10 rows shown, there are more</p>" if len(rows) > 10 else ""
        return f"<table><thead><tr>{cells}</tr></thead><tbody>{body}</tbody></table>{more}"


class GroupedFrame:
    """A frame with group keys chosen, waiting for aggregates."""

    def __init__(self, frame: Frame, keys: list[Expr]) -> None:
        self._frame = frame
        self._keys = keys

    def agg(self, *aggregates: object) -> Frame:
        """Apply aggregates to each group."""
        return self._frame.aggregate(*aggregates, group_by=self._keys)


_CELL_LIMIT = 32


def _cell(value: object) -> str:
    """One value as text, shortened if it would stretch the table."""
    if value is None:
        return "NULL"
    text = str(value)
    return text if len(text) <= _CELL_LIMIT else text[: _CELL_LIMIT - 1] + "\u2026"


def _box(columns: list[str], types: list[str], rows: list[tuple[Any, ...]], *, more: bool) -> str:
    """Rows drawn as a table, the way DuckDB's own shell draws them."""
    heading = [columns, types]
    body = [[_cell(v) for v in row] for row in rows]
    widths = [max(len(line[i]) for line in [*heading, *body]) for i in range(len(columns))]

    def rule(left: str, join: str, right: str) -> str:
        return left + join.join("\u2500" * (w + 2) for w in widths) + right

    def line(cells: list[str]) -> str:
        padded = (c.ljust(w) for c, w in zip(cells, widths, strict=True))
        return "\u2502 " + " \u2502 ".join(padded) + " \u2502"

    drawn = [rule("\u250c", "\u252c", "\u2510"), line(columns), line(types), rule("\u251c", "\u253c", "\u2524")]
    drawn += [line(cells) for cells in body]
    drawn.append(rule("\u2514", "\u2534", "\u2518"))
    if more:
        drawn.append(f"({len(body)} rows shown, there are more)")
    return "\n".join(drawn)


def sql(query: str) -> Frame:
    """A plan from SQL text.

    Nothing is checked here and no connection is involved. The text is used as
    written, so never build one from input you do not trust.
    """
    return Frame(Sql(query))


def table(name: str) -> Frame:
    """A plan reading a table or view, by name.

    The name is quoted, so a name from outside the program cannot turn into
    more SQL. What the table holds is the catalog's to say, and is asked of
    whichever connection the plan runs on.
    """
    return Frame(Table(name))


def values(rows: Iterable[Iterable[object]], columns: Iterable[str | tuple[str, str]]) -> Frame:
    """A plan over rows given here, in memory.

    `columns` names them, as names or as (name, type) pairs. Every value is
    bound as a parameter when the plan runs, so it exists before any
    database does and carries no text into the query.

        values([(1, "nl"), (2, "be")], columns=["id", "country"])
        values([], columns=[("id", "INTEGER")])   # no rows needs the types
    """
    heading = tuple(Column(c, None) if isinstance(c, str) else Column(*c) for c in columns)
    return Frame(Values(tuple(tuple(Lit(v) for v in row) for row in rows), heading))
