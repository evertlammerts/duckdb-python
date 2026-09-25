"""Build a query a step at a time, without a connection.

A query built this way is called a plan, and each verb added to it is a step. A plan holds no connection and no
database, so a connection is passed only to the calls that need one, such as running the query.

    plan = table("people").filter(col("age") > 30).select(col("name"))
    plan.rows(con)                # rows, from a connection you pass
    plan.on(con).rows()           # the same, with the connection filled in

Nothing a connection reports is stored on a plan, because what a database holds changes; column names and types are
worked out afresh against whichever connection runs the query, and a plan used twice is built once.
"""

from __future__ import annotations

import dataclasses
import html
import os
from collections.abc import Iterable, Sized
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .._expressions.expr import (
    Col,
    Expr,
    FamilyExpr,
    Lit,
    ParamSink,
    PlanBase,
    Side,
    Star,
    SubQuery,
    col,
    count_all,
    fold_name,
    function_name,
    identifier,
    name_parts,
    parameters_in,
    plain_identifier,
    quote,
    render_literal,
    rendering_steps,
    subqueries,
    suspended_sinks,
)
from ..exceptions import Error
from .connection import Connection, LiveResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = [
    "Bound",
    "Column",
    "Frame",
    "NeedsConnection",
    "Step",
    "read_csv",
    "read_json",
    "read_parquet",
    "sql",
    "table",
    "table_function",
    "values",
]


class Column(NamedTuple):
    """One output column, whose `type` is None when only DuckDB can say what it is."""

    name: str
    type: str | None = None


#: The columns a step produces, name and type, in order.
Shape = tuple[Column, ...]

#: How many remembered column-type answers a connection keeps, so a long run cannot grow without bound.
_STUB_LIMIT = 512


def _stub_shape(connection: Connection, sql: str) -> Shape:
    """Ask DuckDB for a query's columns without running it, remembered per connection because settings change it."""
    with connection._stub_lock:
        _forget_if_moved(connection)
        remembered = connection._stub_answers.get(sql)
    if remembered is not None:
        return cast("Shape", remembered)
    output, _ = connection._engine().bind(sql)
    shape = tuple(Column(name, type_text) for name, type_text in output)
    with connection._stub_lock:
        _forget_if_moved(connection)
        # The oldest goes, one at a time: clearing everything at the limit would throw away a working set.
        while len(connection._stub_answers) >= _STUB_LIMIT:
            connection._stub_answers.pop(next(iter(connection._stub_answers)))
        connection._stub_answers[sql] = shape
    return shape


def _forget_if_moved(connection: Connection) -> None:
    """Drop the remembered answers if any connection to this database has changed the catalog since."""
    current = connection._catalog.generation
    if connection._stub_generation != current:
        connection._stub_answers.clear()
        connection._stub_generation = current


def _completer(known: dict[int, Shape], connection: Connection | None) -> Callable[[Frame], Shape]:
    """A function giving one step's columns with every type filled in, asking DuckDB one step at a time."""

    def typed(node: Frame) -> Shape:
        shape = known[id(node)]
        if all(column.type is not None for column in shape):
            return shape
        if connection is None:
            message = "types are the engine's to say; pass a connection"
            raise NeedsConnection(message)
        answered = node._ask(connection, typed)
        # The names are this library's and the types DuckDB's; they line up because it answered our own select list.
        if len(answered) == len(shape):
            answered = tuple(Column(ours.name, theirs.type) for ours, theirs in zip(shape, answered, strict=True))
        known[id(node)] = answered
        return answered

    return typed


def _uses_of(expressions: Iterable[Expr]) -> tuple[Frame, ...]:
    """The plans a step's expressions refer to, each once, in order."""
    found: dict[int, Frame] = {}
    for expression in expressions:
        for plan in subqueries(expression):
            if isinstance(plan, Frame):
                found.setdefault(id(plan), plan)
    return tuple(found.values())


def _as_stub(shape: Shape) -> str:
    """A select list producing no rows with these column names and types."""
    return ", ".join(f"NULL::{column.type} AS {quote(column.name)}" for column in shape)


def _duplicates(names: list[str]) -> list[str]:
    """The names appearing more than once as DuckDB compares them, by first spelling, in order of appearance."""
    seen: dict[str, list[str]] = {}
    for name in names:
        seen.setdefault(fold_name(name), []).append(name)
    return [spellings[0] for spellings in seen.values() if len(spellings) > 1]


def _has(shape: Shape, name: str) -> bool:
    """Whether these columns include a name, compared the way DuckDB compares names."""
    wanted = fold_name(name)
    return any(fold_name(column.name) == wanted for column in shape)


def _require_present(shape: Shape, names: Iterable[str], verb: str) -> None:
    """Refuse a step naming a column its input does not have, which DuckDB would not always report."""
    missing = [name for name in names if not _has(shape, name)]
    if missing:
        listed = ", ".join(repr(name) for name in missing)
        message = f"{verb}: no column {listed} in the input"
        raise ValueError(message)


def _require_unique(shape: Shape, verb: str) -> Shape:
    """Refuse a step producing one name twice, which DuckDB accepts behind a WITH by silently taking the first."""
    repeated = _duplicates([column.name for column in shape])
    if repeated:
        listed = ", ".join(repr(name) for name in repeated)
        message = f"{verb} would produce {listed} more than once, and `col` could not tell them apart"
        raise ValueError(message)
    return shape


def _type_of(name: str, shape: Shape) -> str | None:
    """The type of one column, if it is known."""
    wanted = fold_name(name)
    for column in shape:
        if fold_name(column.name) == wanted:
            return column.type
    return None


def _contributed(expression: Expr, source: Shape) -> list[Column] | None:
    """What one select-list item adds, or None when only DuckDB names it, as with an unaliased `x * 2.5`."""
    alias = expression._alias
    while isinstance(expression, FamilyExpr):
        # A namespace such as .str() produces the SQL of what it wraps, so it names its column the same way.
        expression = expression.inner
    if isinstance(expression, Star):
        if alias:
            return None
        excluded = {fold_name(name) for name in expression.exclude}
        renamed = {fold_name(old): new for old, new in expression.rename.items()}
        kept = [c for c in source if fold_name(c.name) not in excluded]
        return [Column(renamed.get(fold_name(c.name), c.name), c.type) for c in kept]
    if isinstance(expression, Col):
        # A join side's column has two parts; the column is the last one.
        bare = expression.parts[-1]
        return [Column(alias or bare, _type_of(bare, source))]
    if alias:
        # The caller named it, but an expression computed it, so only DuckDB can say the type.
        return [Column(alias, None)]
    return None


class JoinKind(NamedTuple):
    """One join kind: its keyword, whether it needs a condition, and whether it keeps the right side's columns."""

    keyword: str
    needs_on: bool
    keeps_right: bool


#: The join kinds, listed once and closed, so `how` can never carry caller text into the SQL.
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

_OPTION_NAME = plain_identifier


def _option(name: str) -> str:
    """A COPY option name. Checked, because it cannot be quoted."""
    if not _OPTION_NAME.match(name):
        message = f"not a copy option name: {name!r}"
        raise ValueError(message)
    return name.upper()


def _option_value(name: str, value: object) -> str:
    """A COPY option value, written into the SQL because COPY takes no parameters, so a `param()` is refused."""
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
    """The `(NAME value, ...)` of a COPY, with values written into the text because COPY cannot bind any."""
    if not options:
        return ""
    with suspended_sinks():
        return " (" + ", ".join(f"{_option(k)} {_option_value(k, v)}" for k, v in options.items()) + ")"


def _as_expr(value: object) -> Expr:
    """One expression, for verbs with a single slot to fill, where a list can only have been meant as a value."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, str):
        return col(value)
    message = f"expected a column name or expression, got {value!r}; wrap a value in lit()"
    raise TypeError(message)


def _as_exprs(values: Iterable[object] | object) -> list[Expr]:
    """Accept an expression, a column name, or a sequence of either."""
    # Any iterable but text is a sequence of columns, and a string is one name, not its characters.
    items = list(values) if isinstance(values, Iterable) and not isinstance(values, (str, bytes, Expr)) else [values]
    out: list[Expr] = []
    for item in items:
        if isinstance(item, Expr):
            out.append(item)
        elif isinstance(item, str):
            # A bare string is a column name here, unlike inside an expression, where it is a value.
            out.append(col(item))
        else:
            message = f"expected a column name or expression, got {item!r}"
            raise TypeError(message)
    return out


def _type_name(value: object) -> str:
    """The type of a value by name, qualified by its module unless it is a builtin."""
    kind = type(value)
    return kind.__qualname__ if kind.__module__ == "builtins" else f"{kind.__module__}.{kind.__qualname__}"


class NeedsConnection(ValueError):
    """Working this out means asking DuckDB, and no connection was given."""


# --- steps: one record per verb, whose SQL is a function of that record ------


@dataclasses.dataclass(frozen=True, eq=False)
class Step:
    """One verb of a plan and its arguments, held as plain data so a plan can be pickled, compared and read."""

    def __post_init__(self) -> None:
        """Checks on the arguments. Steps with arguments to check override this."""

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore a pickled step and run the checks construction runs."""
        self.__dict__.update(state)
        self.__post_init__()

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        """The SQL for this step, given the names of its inputs and their columns where those are known."""
        raise NotImplementedError

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        """The columns this step produces, from its inputs' columns, or None when only DuckDB knows."""
        return None

    def needs_shapes(self) -> bool:
        """Whether the SQL for this step cannot be built without knowing its inputs' columns."""
        return False

    def expressions(self) -> tuple[Expr, ...]:
        """Every expression this step holds."""
        return ()

    def __eq__(self, other: object) -> bool:
        """Same verb and arguments; written out because a generated one would use `==`, which builds an expression."""
        if type(other) is not type(self):
            return NotImplemented
        return _comparable(self) == _comparable(other)

    def __hash__(self) -> int:
        return hash(_comparable(self))


def _comparable(value: object) -> object:
    """A step's fields as plain data, expressions as SQL text plus their bound values, so different values differ."""
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
    """Refuse a step given nothing to work on, which would leave a dangling clause in the SQL."""
    if not items:
        message = f"{verb} needs at least one {what}"
        raise TypeError(message)


@dataclasses.dataclass(frozen=True, eq=False)
class Table(Step):
    """A table or view, by its name's parts."""

    name: tuple[str, ...]

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        """Read the table; its name is quoted, so it cannot turn into more SQL."""
        return f"SELECT * FROM {identifier(self.name)}"


@dataclasses.dataclass(frozen=True, eq=False)
class Sql(Step):
    """A query, as SQL text."""

    text: str

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        """The text, as written."""
        return self.text


@dataclasses.dataclass(frozen=True, eq=False)
class TableFunction(Step):
    """A table function in a FROM clause; DuckDB may open the file to learn the columns, so a `param()` is refused."""

    name: tuple[str, ...]
    args: tuple[Expr, ...]
    named: tuple[tuple[str, Expr], ...]

    def __post_init__(self) -> None:
        for label, argument in [(str(i), a) for i, a in enumerate(self.args)] + list(self.named):
            held = parameters_in(argument)
            if held:
                message = (
                    f"{function_name(self.name)}() cannot take param({held[0]!r}) as argument {label!r}: the engine "
                    f"works out a table function's columns from its arguments, which a parameter does not have yet"
                )
                raise TypeError(message)

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        arguments = [a.fragment() for a in self.args] + [f"{quote(k)} := {v.fragment()}" for k, v in self.named]
        return f"SELECT * FROM {function_name(self.name)}({', '.join(arguments)})"

    def expressions(self) -> tuple[Expr, ...]:
        return (*self.args, *(v for _, v in self.named))


def _argument(value: object) -> Expr:
    """A table function argument from a Python value, with a path turned into its text and anything else a literal."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, os.PathLike):
        return Lit(os.fspath(value))
    if isinstance(value, (list, tuple)):
        return Lit([os.fspath(v) if isinstance(v, os.PathLike) else v for v in value])
    return Lit(value)


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
        # A type given becomes a cast, so the columns this step reports are the ones DuckDB produces.
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
            # The input's columns are unknown here, so a replaced column is re-added and moves to the end.
            excluded = ", ".join(render_literal(name) for name, _ in self.columns)
            added = ", ".join(e.alias(n).as_select() for n, e in self.columns)
            return f"SELECT COLUMNS(lambda c: c NOT IN ({excluded})), {added} FROM {source}"
        existing = {fold_name(column.name) for column in shapes[0]}
        # REPLACE keeps a column in place and cannot serve a new name, since excluding one that is absent is an error.
        replaced = [f"{e.fragment()} AS {quote(n)}" for n, e in self.columns if fold_name(n) in existing]
        appended = [e.alias(n).as_select() for n, e in self.columns if fold_name(n) not in existing]
        star = f"* REPLACE ({', '.join(replaced)})" if replaced else "*"
        return f"SELECT {', '.join([star, *appended])} FROM {source}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        setting = {fold_name(name) for name, _ in self.columns}
        existing = {fold_name(column.name) for column in shapes[0]}
        # A replaced column keeps its place and a new one is appended; DuckDB types both, since an expression made them.
        kept = [Column(c.name, None) if fold_name(c.name) in setting else c for c in shapes[0]]
        new = (Column(name, None) for name, _ in self.columns if fold_name(name) not in existing)
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
        _require_present(shapes[0], self.names, "drop")
        dropped = {fold_name(name) for name in self.names}
        return tuple(c for c in shapes[0] if fold_name(c.name) not in dropped)


@dataclasses.dataclass(frozen=True, eq=False)
class Rename(Step):
    pairs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _at_least_one(self.pairs, "rename")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        rendered = ", ".join(f"{quote(old)} AS {quote(new)}" for old, new in self.pairs)
        return f"SELECT * RENAME ({rendered}) FROM {names[0]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        _require_present(shapes[0], [old for old, _ in self.pairs], "rename")
        renamed = {fold_name(old): new for old, new in self.pairs}
        return _require_unique(
            tuple(Column(renamed.get(fold_name(c.name), c.name), c.type) for c in shapes[0]), "rename"
        )


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
        # Names and order are untouched, and an opened column takes its element type, which only DuckDB knows.
        opened = {fold_name(name) for name in self.columns}
        return tuple(Column(c.name, None) if fold_name(c.name) in opened else c for c in shapes[0])


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
        folded = {fold_name(name) for name in self.columns}
        kept = (c for c in shapes[0] if fold_name(c.name) not in folded)
        return _require_unique((*kept, Column(self.name, "VARCHAR"), Column(self.value, None)), "unpivot")


@dataclasses.dataclass(frozen=True, eq=False)
class Aggregate(Step):
    keys: tuple[Expr, ...]
    aggregates: tuple[Expr, ...]

    def __post_init__(self) -> None:
        _at_least_one((*self.keys, *self.aggregates), "aggregate", "aggregate or group key")

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        selected = [k.as_select() for k in self.keys] + [e.as_select() for e in self.aggregates]
        if any(isinstance(k, Star) for k in self.keys):
            # A star spans an unknown number of positions.
            grouped = [k.fragment() for k in self.keys]
        else:
            # By position, since rendering a key twice would give each of its values a second parameter number.
            grouped = [str(i) for i in range(1, len(self.keys) + 1)]
        clause = " GROUP BY " + ", ".join(grouped) if self.keys else ""
        return f"SELECT {', '.join(selected)} FROM {names[0]}{clause}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        return _projected((*self.keys, *self.aggregates), shapes[0], "aggregate")

    def expressions(self) -> tuple[Expr, ...]:
        return (*self.keys, *self.aggregates)


@dataclasses.dataclass(frozen=True, eq=False)
class Describe(Step):
    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        # Wrapped in a SELECT because a bare SUMMARIZE cannot follow a WITH, and every later step sits behind one.
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
        """Right-hand names the left already carries; USING keys merge into one column and do not clash."""
        left_names = {fold_name(column.name) for column in left}
        folded = {fold_name(key) for key in self.using}
        return [c.name for c in right if fold_name(c.name) in left_names and fold_name(c.name) not in folded]

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
            # Listed out only when renaming, so a plain star keeps USING's merging intact; the keys are already in l.*.
            excluded = f" EXCLUDE ({', '.join(keys)})" if keys else ""
            renamed = ", ".join(f"{quote(name)} AS {quote(name + str(self.suffix))}" for name in shared)
            projection = f"l.*, r.*{excluded} RENAME ({renamed})"
        # Both sides are aliased so a frame can join to itself and a condition can tell the sides apart, as l["id"].
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
        renamed = {fold_name(name): name + str(self.suffix) for name in shared}
        folded = {fold_name(key) for key in self.using}
        right_types = {fold_name(c.name): c.type for c in right_shape}
        # A merged key takes the type both sides promote to, which only DuckDB knows unless the two already agree.
        left = [
            Column(c.name, None) if fold_name(c.name) in folded and right_types.get(fold_name(c.name)) != c.type else c
            for c in left_shape
        ]
        carried = [
            Column(renamed.get(fold_name(c.name), c.name), c.type)
            for c in right_shape
            if fold_name(c.name) not in folded
        ]
        # The renamed copies must be free too: suffixing onto a name the left already holds only moves the clash.
        return _require_unique((*left, *carried), "join")

    def expressions(self) -> tuple[Expr, ...]:
        return (self.on,) if self.on is not None else ()


@dataclasses.dataclass(frozen=True, eq=False)
class SetOp(Step):
    keyword: str
    by_name: bool = False

    def render(self, names: tuple[str, ...], shapes: tuple[Shape | None, ...]) -> str:
        return f"SELECT * FROM {names[0]} {self.keyword} SELECT * FROM {names[1]}"

    def shape(self, shapes: tuple[Shape, ...]) -> Shape | None:
        # Matching by name can add a column from the right, so only DuckDB knows the result.
        if self.by_name:
            return None
        left, right = shapes
        # Each column takes the type both sides promote to, which only DuckDB knows unless the two already agree.
        if len(left) != len(right):
            return tuple(Column(c.name, None) for c in left)
        return tuple(Column(c.name, c.type if c.type == r.type else None) for c, r in zip(left, right, strict=True))


def _projected(chosen: tuple[Expr, ...], source: Shape, verb: str) -> Shape | None:
    """The columns a select list produces, or None where only DuckDB can name one."""
    out: list[Column] = []
    for expression in chosen:
        columns = _contributed(expression, source)
        if columns is None:
            return None
        out.extend(columns)
    return _require_unique(tuple(out), verb)


def _merge_streams(order: list[Frame], connection: Connection) -> tuple[list[Frame], dict[int, int]]:
    """One CTE per registered stream however many `table()` steps name it, since a stream can be read once.

    Returns the order without the duplicates and, for each duplicate, the id of the step standing in for it.
    """
    first: dict[str, int] = {}
    merged: dict[int, int] = {}
    kept: list[Frame] = []
    for node in order:
        step = node._step
        if isinstance(step, Table) and len(step.name) == 1:
            key = fold_name(step.name[0])
            if key in first:
                merged[id(node)] = first[key]
                continue
            if connection._engine().registered_kind(step.name[0]) == "stream":
                first[key] = id(node)
        kept.append(node)
    return kept, merged


class Frame(PlanBase):
    """One step of a plan. Immutable, so every verb returns a new frame."""

    def __init__(self, step: Step, inputs: tuple[Frame, ...] = ()) -> None:
        #: This step's verb and arguments; its expressions become SQL only when the plan does, so values are bound.
        self._step = step
        self._inputs = inputs
        #: Plans reached through subqueries here, which become steps of their own so each is built once.
        self._uses = _uses_of(step.expressions())

    @property
    def step(self) -> Step:
        """This step's verb and arguments."""
        return self._step

    @property
    def inputs(self) -> tuple[Frame, ...]:
        """The plans this step reads."""
        return self._inputs

    # -- building the SQL

    def _order(self) -> list[Frame]:
        """Every node this frame depends on, inputs before the nodes using them."""
        # An explicit stack, not recursion: a chain built in a loop can be thousands of steps deep.
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
            # Reversed, so inputs are visited left to right; step numbering and parameter order follow that.
            stack.extend((parent, False) for parent in reversed(node._inputs + node._uses))
        return order

    def render(self, connection: Connection | None = None) -> str:
        """The whole query as SQL, one CTE per step, values written in and any frame used twice appearing once.

        Without a connection nothing is checked, and a step needing its input's columns falls back to a form that
        does not, which can move a replaced column to the end; a join with a suffix has no such form and raises
        `NeedsConnection`.
        """
        order = self._order()
        shapes = self._shapes(connection, order) if connection is not None else None
        return self._render(shapes, order, connection)

    def render_parameterized(
        self, connection: Connection | None = None, *, parameters: Mapping[str, object] | None = None
    ) -> tuple[str, list[Any]]:
        """The SQL with `$n` placeholders and the values in order; every `param()` must be given one."""
        sql, values = self._sql_and_values(connection=connection, parameters=parameters)
        return sql, values or []

    def _render(
        self,
        shapes: dict[int, Shape] | None = None,
        order: list[Frame] | None = None,
        connection: Connection | None = None,
    ) -> str:
        """The SQL, given whatever column names and types are known."""
        order = self._order() if order is None else order
        merged: dict[int, int] = {}
        if connection is not None:
            order, merged = _merge_streams(order, connection)
        names = {id(node): quote(f"_s{i}") for i, node in enumerate(order)}
        names.update({duplicate: names[first] for duplicate, first in merged.items()})

        def body_of(node: Frame) -> str:
            # Columns are handed down, never read off the node; a step needing its input's says so by using them.
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
        """The SQL for a macro body, worked out only as far as building it needs."""
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
        return self._render(shapes, order, connection)

    # -- column names worked out here, types asked of DuckDB

    def _shapes(
        self,
        connection: Connection | None = None,
        order: list[Frame] | None = None,
        only: set[int] | None = None,
    ) -> dict[int, Shape]:
        """Every step's columns, asked afresh each time since what a source holds changes."""
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
        """Ask DuckDB about this step alone, over empty inputs; an answer naming no table is remembered."""
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
            # A subquery's plan is emptied out too, so the question still names no table.
            stub = quote(f"_use{position}")
            stubs.append(f"{stub} AS (SELECT {_as_stub(typed(used))} WHERE FALSE)")
            names[id(used)] = stub
        with suspended_sinks(), rendering_steps(names):
            body = self._step.render(tuple(names[id(p)] for p in self._inputs), tuple(given))
        if not stubs:
            # A source's answer belongs to one database, so it is asked every time rather than remembered.
            output, _ = connection._engine().bind(body)
            return tuple(Column(name, type_text) for name, type_text in output)
        return _stub_shape(connection, f"WITH {', '.join(stubs)}\n{body}")

    def resolve(self, connection: Connection | None = None) -> Shape:
        """The columns this plan produces, in order; their types are known only once DuckDB is asked."""
        return self._shapes(connection)[id(self)]

    def schema(self, connection: Connection | None = None) -> list[tuple[str, str]]:
        """The columns this plan produces, as (name, type), asking DuckDB for the types, unlike `.columns`."""
        shapes = self._shapes(connection)
        # Filled in the way an input is for the step above it: one question, over inputs already typed.
        shape = _completer(shapes, connection)(self)
        unresolved = [column.name for column in shape if column.type is None]
        if unresolved:  # pragma: no cover (DuckDB types every column it reports)
            message = f"no type reported for {unresolved}"
            raise Error(message)
        return [(column.name, type_text) for column in shape if (type_text := column.type) is not None]

    def columns(self, connection: Connection | None = None) -> list[str]:
        """The column names; a connection is only needed when DuckDB named something or a source must be read."""
        return [column.name for column in self.resolve(connection)]

    def types(self, connection: Connection | None = None) -> list[str]:
        """The column types, which only DuckDB can say, so it is asked."""
        return [type_text for _, type_text in self.schema(connection)]

    def describe(self) -> Frame:
        """Per-column statistics as a plan: count, min, max, average and quartiles. Reads the rows, unlike `.schema`."""
        return self._derive(Describe())

    # -- verbs

    def filter(self, predicate: Expr) -> Frame:
        """Keep the rows where `predicate` holds; for a condition written as SQL, use `filter(sql_expr("..."))`."""
        if not isinstance(predicate, Expr):  # the annotation says Expr, and callers do not always listen
            message = (  # type: ignore[unreachable]
                f"filter takes an expression, not {type(predicate).__name__}; "
                f"for a condition written as SQL, use filter(sql_expr(...))"
            )
            raise TypeError(message)
        return self._derive(Filter(predicate))

    def select(self, *columns: object) -> Frame:
        """Keep only these columns or expressions; it projects and never groups."""
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
        """A random subset: `n` rows, or `percent` of them, with `seed` repeating a draw."""
        if (n is None) == (percent is None):
            message = "sample takes either n or percent"
            raise TypeError(message)
        # Only reservoir can hit an exact count; the others sample each row independently.
        size = f"{int(n)} ROWS" if n is not None else f"{float(percent or 0)} PERCENT"
        chosen = _option(method) if method else ("RESERVOIR" if n is not None else "BERNOULLI")
        arguments = f"{chosen}, {int(seed)}" if seed is not None else chosen
        return self._derive(Sample(size, arguments))

    def unnest(self, *columns: str) -> Frame:
        """Expand list columns to one row per element; two unnested together walk in step, not paired."""
        return self._derive(Unnest(tuple(columns)))

    def unpivot(self, *columns: str, name: str = "name", value: str = "value") -> Frame:
        """Turn the named columns into rows; the others repeat, and `name` and `value` name the two new columns."""
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
        on: Expr | str | Iterable[str] | Callable[[Side, Side], object] | None = None,
        how: str = "inner",
        suffix: str | None = None,
    ) -> Frame:
        """Join to another frame, on a column name or a list of them for a USING join, which merges the column.

        Any other condition is a function of the two sides, `on=lambda l, r: l["id"] == r["order_id"]`, which runs
        once when the join is built and must return an expression.
        `how` is inner, left, right, outer, semi, anti, cross, positional, natural or asof.
        Both sides' columns are carried through, so a name on both is refused when the plan is resolved or run, since
        `col` could not tell the two apart; pass `suffix` to rename the right side's copies.
        """
        using: tuple[str, ...] = ()
        condition: Expr | None = None
        if isinstance(on, str):
            using = (on,)
        elif isinstance(on, Expr):
            condition = on
        elif callable(on):
            built = on(Side("l"), Side("r"))
            if not isinstance(built, Expr):
                message = (
                    f"the join condition returned {built!r}, which is not an expression; it runs once, when the "
                    f"join is built, on the two sides, so its body must be built from their columns"
                )
                raise TypeError(message)
            condition = built
        elif on is not None:
            keys = list(on)
            if not all(isinstance(key, str) for key in keys):
                message = (
                    "a USING join takes column names; for a condition pass a function of the two sides, "
                    "on=lambda left, right: ..."
                )
                raise TypeError(message)
            using = tuple(keys)
        return Frame(Join(how.lower(), condition, using, suffix), (self, other))

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
        """A single column as a plan; `plan["x"]` narrows the plan, where `expr["x"]` reaches inside a value."""
        if not isinstance(name, str):
            # Without this, `for row in plan` would fall back to plan[0], plan[1] and never stop.
            message = f"a plan is indexed by column name, not by {type(name).__name__}"
            raise TypeError(message)
        return self.select(col(name))

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """Refuse to iterate without a connection, and stop the old sequence protocol falling back to `__getitem__`."""
        message = "iterating a plan needs a connection; use rows(connection)"
        raise TypeError(message)

    # -- as a value inside another query

    def scalar(self) -> Expr:
        """This query where a single value is expected; it must give one row and one column."""
        return SubQuery(self)

    # -- execution

    def _sql_and_values(
        self,
        wrap: Callable[[str], str] | None = None,
        connection: Connection | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> tuple[str, list[Any] | None]:
        """The SQL with literals pulled out as `$n`, and the values, each of which must be used."""
        order = self._order()
        shapes = self._resolution(connection, order) if connection is not None else None
        with ParamSink() as sink:
            sql = self._render(shapes, order, connection)
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
        """The connection a plan runs on; a `dbapi.Connection` is refused because it tracks its cursors' live result."""
        if not isinstance(connection, Connection):
            message = (
                f"a plan runs on a duckdb.frame.Connection, not {_type_name(connection)}; "
                f"to run it through the DB-API, execute plan.render() on a cursor"
            )
            raise TypeError(message)
        return connection

    def _resolution(self, connection: Connection, order: list[Frame]) -> dict[int, Shape] | None:
        """The columns the SQL is built with; a lone source skips it, DuckDB cannot describe a PIVOT."""
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

    def to_numpy(self, connection: Connection, *, parameters: Mapping[str, object] | None = None) -> dict[str, Any]:
        """Every column as a numpy array; NULLs come back masked and values come straight from DuckDB's buffers."""
        # Imported here so that importing duckdb does not require numpy.
        from ._numpy import fetch_numpy

        connection = self._on(connection)
        sql, values = self._sql_and_values(connection=connection, parameters=parameters)
        with connection._execute(sql, values) as result:
            return fetch_numpy(result.result)

    def on(self, connection: Connection) -> Bound:
        """This plan with a connection filled in, so `plan.on(con).rows()` takes no argument; the plan is unchanged."""
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
        name: str | tuple[str, ...],
        *,
        replace: bool = False,
        temporary: bool = False,
        parameters: Mapping[str, object] | None = None,
    ) -> int:
        """Store the rows in a new table, a tuple naming a qualified one. Returns how many were written."""
        prefix = "CREATE OR REPLACE" if replace else "CREATE"
        kind = "TEMPORARY TABLE" if temporary else "TABLE"
        target = identifier(name_parts(name, "table name"))
        return self._run(connection, lambda q: f"{prefix} {kind} {target} AS {q}", parameters)

    def insert_into(
        self, connection: Connection, name: str | tuple[str, ...], *, parameters: Mapping[str, object] | None = None
    ) -> int:
        """Append the rows to a table that exists, a tuple naming a qualified one. Returns how many were added."""
        target = identifier(name_parts(name, "table name"))
        return self._run(connection, lambda q: f"INSERT INTO {target} {q}", parameters)

    def copy_to(
        self,
        connection: Connection,
        path: str | os.PathLike[str],
        *,
        parameters: Mapping[str, object] | None = None,
        **options: object,
    ) -> list[tuple[Any, ...]]:
        """`COPY (this plan) TO path (options)`, returning the rows the statement returns.

        The options are COPY's own under their SQL names, `format` among them; with none, the path's extension decides.
        A value is written as SQL writes it: a list is a column list, `star()` is `*`, and a dict is a struct.
        COPY's own result comes back: the count, the files with `return_files`, or a row per file with `return_stats`.
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
        """Write newline-delimited JSON, or one array with `array=True`; `copy_to` with `format="json"`."""
        return self.copy_to(connection, path, parameters=parameters, format="json", **options)

    # -- looking at it

    def explain(
        self, connection: Connection, *, analyze: bool = False, parameters: Mapping[str, object] | None = None
    ) -> str:
        """How DuckDB will run the query, as text; with `analyze` it runs and each step carries what it cost."""
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
        # One row past the limit, so the footer can say there are more without counting them all.
        head = self.limit(limit + 1)
        rows = head.rows(connection, parameters=parameters)
        return _box(head.columns(connection), head.types(connection), rows[:limit], more=len(rows) > limit)

    def __repr__(self) -> str:
        """The SQL. A plan holds no connection, so there are no rows to show."""
        try:
            rendered = self.render()
        except NeedsConnection as reason:
            # A repr that raised would make a debugger useless, so a join with a suffix says why instead.
            return f"<Frame, renders with a connection: {reason}>"
        first = rendered.splitlines()
        shown = first[0] if len(first) == 1 else f"{first[0]} ... ({len(first)} lines)"
        return f"<Frame {shown}>"


class Bound:
    """A plan and a connection together, from `plan.on(con)`, so the calls that run it take no connection."""

    __slots__ = ("connection", "plan")

    def __init__(self, plan: Frame, connection: Connection) -> None:
        self.plan = plan
        self.connection = connection

    def render_parameterized(self, *, parameters: Mapping[str, object] | None = None) -> tuple[str, list[Any]]:
        """The query as it is sent to run: SQL with `$n` placeholders, and the values in order."""
        return self.plan.render_parameterized(self.connection, parameters=parameters)

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

    def to_numpy(self, *, parameters: Mapping[str, object] | None = None) -> dict[str, Any]:
        """Every column as a numpy array; columns holding NULLs come back masked."""
        return self.plan.to_numpy(self.connection, parameters=parameters)

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
        """How DuckDB will run the query, as text."""
        return self.plan.explain(self.connection, analyze=analyze, parameters=parameters)

    def show(self, limit: int = 10, *, parameters: Mapping[str, object] | None = None) -> None:
        """Print the first rows."""
        self.plan.show(self.connection, limit, parameters=parameters)

    def preview(self, limit: int = 10, *, parameters: Mapping[str, object] | None = None) -> str:
        """The first rows drawn as a table."""
        return self.plan.preview(self.connection, limit, parameters=parameters)

    def create(
        self,
        name: str | tuple[str, ...],
        *,
        replace: bool = False,
        temporary: bool = False,
        parameters: Mapping[str, object] | None = None,
    ) -> int:
        """Store the rows in a new table."""
        return self.plan.create(self.connection, name, replace=replace, temporary=temporary, parameters=parameters)

    def insert_into(self, name: str | tuple[str, ...], *, parameters: Mapping[str, object] | None = None) -> int:
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
            # A notebook or debugger shows this unasked, so a plan that cannot run here is described, not raised.
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
    """A plan from SQL text, used exactly as written, so never build one from input you do not trust."""
    return Frame(Sql(query))


def table(name: str | tuple[str, ...]) -> Frame:
    """A plan reading a table, view or file by name; a string is never split, so a qualified name is a tuple."""
    return Frame(Table(name_parts(name, "table name")))


def table_function(name: str | tuple[str, ...], *args: object, **named: object) -> Frame:
    """A plan over a table function: `table_function("read_csv", "x.csv", header=True)`, `table_function("range", 10)`.

    Any table function DuckDB has, with positional arguments as given and named ones written as `name := value`.
    A string, number or date is bound when the query runs, a list is a list, and a dict is a struct.
    """
    parts = name_parts(name, "function name")
    return Frame(
        TableFunction(parts, tuple(_argument(a) for a in args), tuple((k, _argument(v)) for k, v in named.items()))
    )


def read_csv(path: object, **named: object) -> Frame:
    """A plan over `read_csv(path, ...)`: one file, a list of them or a glob, with the reader's own named arguments."""
    return table_function("read_csv", path, **named)


def read_parquet(path: object, **named: object) -> Frame:
    """A plan over `read_parquet(path, ...)`: one file, a list of them or a glob."""
    return table_function("read_parquet", path, **named)


def read_json(path: object, **named: object) -> Frame:
    """A plan over `read_json(path, ...)`: one file, a list of them or a glob."""
    return table_function("read_json", path, **named)


def values(rows: Iterable[Iterable[object]], columns: Iterable[str | tuple[str, str]]) -> Frame:
    """A plan over rows given here; `columns` are names or (name, type) pairs."""
    heading = tuple(Column(c, None) if isinstance(c, str) else Column(*c) for c in columns)
    return Frame(Values(tuple(tuple(Lit(v) for v in row) for row in rows), heading))
