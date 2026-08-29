"""A query you build up a step at a time.

A frame is a plan, and only a plan. It holds no connection and no database, so
it renders to SQL that will run wherever that SQL is valid, and the same plan
can be a macro body, a query on one connection, and a query on another.

    plan = table("people").filter(col("age") > 30).select(col("name"))
    plan.render()                 # SQL, no engine involved
    plan.fetchall(con)            # rows, from a connection you pass

A connection is an argument to the things that need one: resolving a schema,
running the query, writing the rows somewhere. Nothing else.

The plan is a graph rather than a tree, so a frame used in two places is
computed once. Column names are worked out here, because this library decided
them. Types, and any name the engine chose, are asked for.

Nothing a connection said is ever stored on a plan. A schema is one catalog's
answer at one moment, and a plan that remembered it would render one way here
and the same way somewhere the answer differs. Shapes are worked out afresh
against whichever connection is running, and a source may instead be given a
schema outright, which is a statement by the caller rather than a memory.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from .connection import Connection, LiveResult
from .exceptions import Error
from .expr import (
    Col,
    Expr,
    ParamSink,
    PlanBase,
    Star,
    SubQuery,
    col,
    qualified,
    quote,
    render_literal,
    rendering_steps,
    subqueries,
    suspended_sinks,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = ["Column", "Declared", "Frame", "declare", "sql", "table"]


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

#: How a step works out its own shape from its inputs' shapes. Returning None
#: means only the engine knows, so the shape is bound instead.
ShapeRule = "Callable[[tuple[Shape, ...]], Shape | None]"


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
        remembered = connection._stub_answers.get(sql)
    if remembered is not None:
        return cast("Shape", remembered)
    output, _ = connection._engine().bind(sql)
    shape = tuple(Column(name, type_text) for name, type_text in output)
    with connection._stub_lock:
        # The oldest answer goes, one at a time. Clearing everything at the
        # limit would throw away a working set at the busiest moment.
        while len(connection._stub_answers) >= _STUB_LIMIT:
            connection._stub_answers.pop(next(iter(connection._stub_answers)))
        connection._stub_answers[sql] = shape
    return shape


def _step_names(named: dict[Frame, str]) -> dict[int, str]:
    """Identity to step name, answering for earlier versions of a step too.

    A subquery built before a `bind()` holds the version it saw; the rebuilt
    step carries that version in `_aliases`, so the reference resolves. An
    alias redirects only when the earlier version is not itself a step of
    this render: a graph holding both the bound and the unbound descendant
    of one hole names each as itself.
    """
    names = {id(node): name for node, name in named.items()}
    for node, name in named.items():
        for earlier in node._aliases:
            names.setdefault(id(earlier), name)
    return names


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


def _check_heading(name: str, declared: Declared, plan: Frame) -> None:
    """Refuse a plan whose columns do not fit the heading it is bound to.

    Checked as far as the plan can say without a connection: names always
    when they are derivable, types where both sides know them. A plan whose
    columns only the engine knows is checked by the engine when it runs.
    """
    try:
        supplied = plan.resolve()
    except NeedsConnection:
        return
    expected = [column.name for column in declared.heading]
    actual = [column.name for column in supplied]
    if actual != expected:
        message = f"{name!r} is declared with columns {expected}, but the plan bound to it has {actual}"
        raise ValueError(message)
    for wanted, got in zip(declared.heading, supplied, strict=True):
        if wanted.type is not None and got.type is not None and wanted.type.upper() != got.type.upper():
            message = f"{name!r} declares {wanted.name} as {wanted.type}, but the plan bound to it has {got.type}"
            raise ValueError(message)


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


def _identity(shapes: tuple[Shape, ...]) -> Shape:
    """Steps that keep every column: filter, sort, limit, distinct, sample."""
    return shapes[0]


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


def _projection(chosen: list[Expr], verb: str = "select") -> Callable[[tuple[Shape, ...]], Shape | None]:
    """Shape of a select list."""

    def rule(shapes: tuple[Shape, ...]) -> Shape | None:
        out: list[Column] = []
        for expression in chosen:
            columns = _contributed(expression, shapes[0])
            if columns is None:
                return None
            out.extend(columns)
        return _require_unique(tuple(out), verb)

    return rule


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


class Declared(NamedTuple):
    """A relation variable: a name with a heading, standing for a relation.

    The heading is a signature, not a claim about any table, so nothing can
    make it stale. `Frame.bind()` closes the hole with a plan and checks that
    plan against the heading.
    """

    name: str
    heading: Shape


class Frame(PlanBase):
    """One step of a query. Immutable: every verb returns a new frame."""

    def __init__(
        self,
        body: Callable[..., str],
        inputs: tuple[Frame, ...] = (),
        shape: Callable[[tuple[Shape, ...]], Shape | None] | None = None,
        declared: Declared | None = None,
        uses: tuple[Frame, ...] = (),
        aliases: tuple[Frame, ...] = (),
    ) -> None:
        #: SQL for this step. Always a callable, taking the rendered names of
        #: this step's inputs, so that expressions render while the parameter
        #: sink is active. Rendering them when the verb was called would bake
        #: their literals into the SQL text instead of binding them.
        #:
        #: Never a format template. A caller's SQL and their column names can
        #: contain braces, and `str.format` would read those as fields.
        self._body: Callable[..., str] = body
        self._inputs = inputs
        #: Plans this step refers to through subqueries in its expressions.
        #: Part of the graph, so they render once as steps of their own, but
        #: not named to the body: an expression finds its plan by identity.
        self._uses = uses
        #: How this step works out its own shape. None, or a rule that returns
        #: None, means the engine decides and the shape is asked for.
        self._shape_rule = shape
        #: A relation variable, if this step is one. Its heading is the shape.
        self._declared = declared
        #: Earlier versions of this step, when `bind()` rebuilt it. A subquery
        #: built before the bind still holds the version it saw; rendering
        #: answers for any of them.
        self._aliases = aliases

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
        return self._render(self._shapes(connection) if connection is not None else None)

    def _render(self, shapes: dict[int, Shape] | None = None) -> str:
        """The SQL, given whatever shapes are known."""
        order = self._order()
        names = _step_names({node: quote(f"_s{i}") for i, node in enumerate(order)})

        def body_of(node: Frame) -> str:
            # Shapes are handed down, never read off the node. A step that
            # needs to know its input's columns says so by using them.
            given = tuple((shapes or {}).get(id(parent)) for parent in node._inputs)
            return node._body(*(names[id(p)] for p in node._inputs), shapes=given)

        with rendering_steps(names):
            if len(order) == 1:
                return body_of(self)
            ctes = ",\n".join(f"{names[id(n)]} AS (\n{body_of(n)}\n)" for n in order[:-1])
            return f"WITH {ctes}\n{body_of(order[-1])}"

    def _derive(
        self,
        template: Callable[..., str],
        *inputs: Frame,
        shape: Callable[[tuple[Shape, ...]], Shape | None] | None = None,
        expressions: Iterable[Expr] = (),
    ) -> Frame:
        return Frame(template, inputs or (self,), shape, uses=_uses_of(expressions))

    # -- shape: names derived here, types from the binder

    def _shapes(self, connection: Connection | None = None) -> dict[int, Shape]:
        """The shape of every step, worked out fresh, keyed by identity.

        Returned rather than stored. Which columns a source has is a fact
        about one catalog at one moment: it changes with the connection and it
        changes under DDL, so a plan that kept it would render by whatever it
        was told first.
        """
        known: dict[int, Shape] = {}
        typed = _completer(known, connection)
        for node in self._order():
            given = tuple(known[id(parent)] for parent in node._inputs)
            shape = node._derived(given)
            if shape is None:
                if connection is None:
                    message = (
                        "working out these columns means asking the engine, so it needs a connection; "
                        "pass one, or build on a declared relation"
                    )
                    raise NeedsConnection(message)
                shape = node._ask(connection, typed)
            known[id(node)] = shape
        return known

    def _derived(self, given: tuple[Shape, ...]) -> Shape | None:
        """This step's shape from its inputs' shapes alone.

        None when only the engine knows.
        """
        if self._declared is not None:
            return self._declared.heading
        if self._shape_rule is not None:
            return self._shape_rule(given)
        return None

    def _ask(self, connection: Connection, typed: Callable[[Frame], Shape]) -> Shape:
        """Ask the engine about this step alone.

        A step with inputs is asked over empty tables of the right shape, so
        the question stays small however long the chain is, and names no
        catalog object. That makes its answer a function of the text, so it is
        remembered on the connection. A source does name the catalog, and its
        answer is only true of the connection that gave it, so it is not.
        """
        stubs: list[str] = []
        named: dict[Frame, str] = {}
        given: list[Shape] = []
        for position, parent in enumerate(self._inputs):
            shape = typed(parent)
            given.append(shape)
            stub = quote(f"_in{position}")
            stubs.append(f"{stub} AS (SELECT {_as_stub(shape)} WHERE FALSE)")
            named[parent] = stub
        for position, used in enumerate(self._uses):
            # A subquery's plan is stubbed too, so the question stays free of
            # the catalog even when an expression refers to another plan.
            stub = quote(f"_use{position}")
            stubs.append(f"{stub} AS (SELECT {_as_stub(typed(used))} WHERE FALSE)")
            named[used] = stub
        names = _step_names(named)
        with suspended_sinks(), rendering_steps(names):
            body = self._body(*(names[id(p)] for p in self._inputs), shapes=tuple(given))
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
        # Wrapped in a SELECT because a bare SUMMARIZE cannot follow a WITH,
        # and every step but the first is reached through one.
        return self._derive(lambda source, **_: f"SELECT * FROM (SUMMARIZE SELECT * FROM {source})")

    # -- relation variables

    def bind(self, **relations: Frame) -> Frame:
        """Close declared relations with plans, by name.

        Substitution, done here and before any engine is involved. Each plan
        is checked against the heading it fills, as far as its columns can be
        worked out without a connection. A hole left open is still a hole;
        executing such a plan means the catalog supplies the name.
        """
        order = self._order()
        holes: dict[str, Frame] = {}
        for node in order:
            if node._declared is None:
                continue
            earlier = holes.get(node._declared.name)
            if earlier is not None and earlier._declared != node._declared:
                message = (
                    f"{node._declared.name!r} is declared twice with different headings: "
                    f"{list(earlier._declared.heading)} and {list(node._declared.heading)}"  # type: ignore[union-attr]
                )
                raise ValueError(message)
            holes[node._declared.name] = node
        unknown = sorted(set(relations) - set(holes))
        if unknown:
            message = f"no declared relation named {', '.join(repr(n) for n in unknown)}"
            raise ValueError(message)
        for name, plan in relations.items():
            _check_heading(name, holes[name]._declared, plan)  # type: ignore[arg-type]
        rebuilt: dict[int, Frame] = {}
        for node in order:
            replacement = relations.get(node._declared.name) if node._declared is not None else None
            if replacement is not None:
                # A copy of the replacement's top step, carrying the hole as an
                # alias: a subquery that holds the hole itself renders as a
                # reference to this step. The replacement is not touched.
                rebuilt[id(node)] = Frame(
                    replacement._body,
                    replacement._inputs,
                    replacement._shape_rule,
                    replacement._declared,
                    replacement._uses,
                    (*replacement._aliases, node),
                )
                continue
            inputs = tuple(rebuilt[id(p)] for p in node._inputs)
            uses = tuple(rebuilt[id(u)] for u in node._uses)
            if inputs == node._inputs and uses == node._uses:
                rebuilt[id(node)] = node
                continue
            rebuilt[id(node)] = Frame(
                node._body, inputs, node._shape_rule, node._declared, uses, (*node._aliases, node)
            )
        return rebuilt[id(self)]

    # -- verbs

    def filter(self, predicate: Expr | str) -> Frame:
        """Keep the rows where `predicate` holds."""

        def body(source: str, **_: object) -> str:
            rendered = predicate if isinstance(predicate, str) else predicate.fragment()
            return f"SELECT * FROM {source} WHERE {rendered}"

        used = [predicate] if isinstance(predicate, Expr) else []
        return self._derive(body, shape=_identity, expressions=used)

    def select(self, *columns: object) -> Frame:
        """Keep only these columns or expressions.

        Pure projection: it does not group, unlike the older client's `select`.
        """
        chosen = _as_exprs(list(columns))

        def body(source: str, **_: object) -> str:
            return f"SELECT {', '.join(e.as_select() for e in chosen)} FROM {source}"

        return self._derive(body, shape=_projection(chosen), expressions=chosen)

    def with_columns(self, **columns: object) -> Frame:
        """Add columns, replacing any of the same name."""
        chosen = {name: _as_expr(value) for name, value in columns.items()}

        def body(source: str, *, shapes: tuple[Shape | None, ...], **_: object) -> str:
            if shapes[0] is None:
                # Nothing is known about the input, so the SQL cannot say which
                # names it replaces. This form needs no such knowledge: keep
                # every input column but the ones being set, then add those.
                # The price is order: a replaced column moves to the end.
                excluded = ", ".join(render_literal(name) for name in chosen)
                added = ", ".join(e.alias(n).as_select() for n, e in chosen.items())
                return f"SELECT COLUMNS(lambda c: c NOT IN ({excluded})), {added} FROM {source}"
            existing = {column.name for column in shapes[0]}
            # A name already present is replaced in place, keeping column
            # order; a new one is appended. EXCLUDE cannot serve both, because
            # excluding a name that is not there is an error.
            replaced = [f"{e.fragment()} AS {quote(n)}" for n, e in chosen.items() if n in existing]
            appended = [e.alias(n).as_select() for n, e in chosen.items() if n not in existing]
            star = f"* REPLACE ({', '.join(replaced)})" if replaced else "*"
            return f"SELECT {', '.join([star, *appended])} FROM {source}"

        def shape(shapes: tuple[Shape, ...]) -> Shape:
            existing = {column.name for column in shapes[0]}
            # A replaced column keeps its position, a new one is appended, and
            # both take their type from the engine because an expression made it.
            kept = [Column(c.name, None) if c.name in chosen else c for c in shapes[0]]
            new = (Column(name, None) for name in chosen if name not in existing)
            return _require_unique((*kept, *new), "with_columns")

        return self._derive(body, shape=shape, expressions=chosen.values())

    def drop(self, *columns: str) -> Frame:
        """Remove columns."""
        rendered = ", ".join(quote(c) for c in columns)
        dropped = set(columns)
        return self._derive(
            lambda source, **_: f"SELECT * EXCLUDE ({rendered}) FROM {source}",
            shape=lambda shapes: tuple(c for c in shapes[0] if c.name not in dropped),
        )

    def rename(self, **columns: str) -> Frame:
        """Rename columns, given as old=new."""
        rendered = ", ".join(f"{quote(old)} AS {quote(new)}" for old, new in columns.items())
        return self._derive(
            lambda source, **_: f"SELECT * RENAME ({rendered}) FROM {source}",
            shape=lambda shapes: _require_unique(
                tuple(Column(columns.get(c.name, c.name), c.type) for c in shapes[0]), "rename"
            ),
        )

    def sort(self, *columns: object) -> Frame:
        """Order the rows. Use `.desc()` and `.nulls_last()` on the columns."""
        ordering = _as_exprs(list(columns))

        def body(source: str, **_: object) -> str:
            return f"SELECT * FROM {source} ORDER BY {', '.join(e.as_order() for e in ordering)}"

        return self._derive(body, shape=_identity, expressions=ordering)

    def limit(self, count: int, offset: int = 0) -> Frame:
        """Keep at most `count` rows, optionally skipping `offset` first."""
        tail = f" OFFSET {int(offset)}" if offset else ""
        return self._derive(lambda source, **_: f"SELECT * FROM {source} LIMIT {int(count)}{tail}", shape=_identity)

    def head(self, count: int = 5) -> Frame:
        """The first `count` rows."""
        return self.limit(count)

    def offset(self, count: int) -> Frame:
        """Skip `count` rows."""
        return self._derive(lambda source, **_: f"SELECT * FROM {source} OFFSET {int(count)}", shape=_identity)

    def distinct(self, on: Iterable[object] | object | None = None) -> Frame:
        """Remove duplicate rows, or duplicates of `on` only."""
        if on is None:
            return self._derive(lambda source, **_: f"SELECT DISTINCT * FROM {source}", shape=_identity)
        keys = _as_exprs(on)

        def body(source: str, **_: object) -> str:
            return f"SELECT DISTINCT ON ({', '.join(e.fragment() for e in keys)}) * FROM {source}"

        return self._derive(body, shape=_identity, expressions=keys)

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
        return self._derive(
            lambda source, **_: f"SELECT * FROM {source} USING SAMPLE {size} ({arguments})", shape=_identity
        )

    def unnest(self, *columns: str) -> Frame:
        """Expand list columns to one row per element, repeating the rest.

        Unnesting two columns together walks them in step rather than making
        every pairing; the shorter one runs out as NULL.
        """
        if not columns:
            message = "unnest needs at least one column"
            raise TypeError(message)
        expanded = ", ".join(f"unnest({quote(c)}) AS {quote(c)}" for c in columns)
        opened = set(columns)
        return self._derive(
            lambda source, **_: f"SELECT * REPLACE ({expanded}) FROM {source}",
            # Names and order are untouched; an opened column becomes its element
            # type, which only the engine knows.
            shape=lambda shapes: tuple(Column(c.name, None) if c.name in opened else c for c in shapes[0]),
        )

    def unpivot(self, *columns: str, name: str = "name", value: str = "value") -> Frame:
        """Turn columns into rows, one row per column named.

        The columns not named are kept and repeat down the rows. `name` and
        `value` name the two columns that replace the ones folded away.
        """
        if not columns:
            message = "unpivot needs at least one column"
            raise TypeError(message)
        folded = ", ".join(quote(c) for c in columns)
        folded_names = set(columns)
        return self._derive(
            lambda source, **_: (
                f"UNPIVOT (SELECT * FROM {source}) ON {folded} INTO NAME {quote(name)} VALUE {quote(value)}"
            ),
            shape=lambda shapes: _require_unique(
                (
                    *(c for c in shapes[0] if c.name not in folded_names),
                    Column(name, "VARCHAR"),
                    Column(value, None),
                ),
                "unpivot",
            ),
        )

    def aggregate(self, *aggregates: object, group_by: Iterable[object] | object = ()) -> Frame:
        """Group and aggregate. Group keys come first in the output."""
        keys = _as_exprs(group_by) if group_by else []
        computed = _as_exprs(list(aggregates))

        def body(source: str, **_: object) -> str:
            # Rendered here, not when the verb was called: every expression has
            # to reach the sink so its literals are bound rather than written
            # into the text.
            selected = [k.as_select() for k in keys] + [e.as_select() for e in computed]
            clause = " GROUP BY " + ", ".join(k.fragment() for k in keys) if keys else ""
            return f"SELECT {', '.join(selected)} FROM {source}{clause}"

        return self._derive(body, shape=_projection([*keys, *computed], "aggregate"), expressions=[*keys, *computed])

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
        kind = _JOIN_KINDS.get(how.lower())
        if kind is None:
            message = f"unknown join kind {how!r}; one of {', '.join(sorted(_JOIN_KINDS))}"
            raise ValueError(message)
        keyword = kind.keyword
        if on is None and kind.needs_on:
            message = f"a {how} join needs `on`"
            raise TypeError(message)

        # Narrowed here rather than inside the closure, where mypy cannot see
        # that the None case was already refused above.
        keys: list[str] = []
        using: list[str] = []
        if on is not None and not isinstance(on, (Expr, str)):
            using = [str(c) for c in list(on)]
        elif isinstance(on, str):
            using = [on]
        keys = [quote(c) for c in using]

        # A semi or anti join keeps only the left side, so nothing can collide.
        keeps_right = kind.keeps_right
        folded = set(using)

        def clashing(left_shape: Shape, right_shape: Shape) -> list[str]:
            """Right-hand names the left already carries.

            USING folds its own keys into one column, so those are not a clash.
            """
            left_names = {column.name for column in left_shape}
            return [c.name for c in right_shape if c.name in left_names and c.name not in folded]

        def body(left: str, right: str, *, shapes: tuple[Shape | None, ...], **_: object) -> str:
            if not kind.needs_on:
                clause = ""
            elif isinstance(on, Expr):
                clause = f" ON {on.fragment()}"
            elif isinstance(on, str):
                clause = f" USING ({quote(on)})"
            else:
                clause = " USING (" + ", ".join(keys) + ")"
            # Both sides are aliased `l` and `r`, so joining a frame to itself
            # works: the graph is keyed by identity, so a self-join names the
            # same CTE twice and the FROM clause would otherwise be ambiguous.
            # The aliases are also how an ON expression tells the sides apart,
            # as col("l.id").
            # Worked out from the shapes handed to this step, not remembered.
            left_shape, right_shape = shapes
            if keeps_right and suffix is not None and (left_shape is None or right_shape is None):
                message = (
                    "a join with a suffix renames the right side's clashing columns, and which "
                    "ones clash depends on both sides' columns; this needs a connection to render"
                )
                raise ValueError(message)
            shared = (
                clashing(left_shape, right_shape)
                if keeps_right and left_shape is not None and right_shape is not None
                else []
            )
            projection = "*"
            if shared:
                # Only when renaming: the plain star keeps the SQL closest to
                # what the reader wrote, and USING's own folding intact.
                parts = [f"{quote(n)} AS {quote(n + str(suffix))}" for n in shared]
                excluded = f" EXCLUDE ({', '.join(keys)})" if keys else ""
                projection = f"l.*, r.*{excluded} RENAME ({', '.join(parts)})"
            return f"SELECT {projection} FROM {left} AS l {keyword} JOIN {right} AS r{clause}"

        def shape(shapes: tuple[Shape, ...]) -> Shape:
            left_shape, right_shape = shapes
            if not keeps_right:
                return left_shape
            shared = clashing(left_shape, right_shape)
            if shared and suffix is None:
                listed = ", ".join(repr(name) for name in shared)
                message = (
                    f"both sides of this join have {listed}; the result would carry each name "
                    f"twice and `col` could not tell them apart. Pass suffix=... to rename the "
                    f"right side, or rename before joining."
                )
                raise ValueError(message)
            renamed = {name: name + str(suffix) for name in shared}
            carried = [Column(renamed.get(c.name, c.name), c.type) for c in right_shape if c.name not in folded]
            # The renamed copies have to be free too: suffixing onto a name the
            # left already holds only moves the clash.
            return _require_unique((*left_shape, *carried), "join")

        return Frame(body, (self, other), shape, uses=_uses_of([on] if isinstance(on, Expr) else []))

    def cross(self, other: Frame) -> Frame:
        """Every combination of rows from both frames."""
        return self.join(other, how="cross")

    def union(self, other: Frame, *, all: bool = True) -> Frame:
        """Rows from both frames, keeping duplicates unless `all` is false."""
        keyword = "UNION ALL" if all else "UNION"
        return Frame(
            lambda left, right, **_: f"SELECT * FROM {left} {keyword} SELECT * FROM {right}",
            (self, other),
            _identity,
        )

    def union_by_name(self, other: Frame, *, all: bool = True) -> Frame:
        """Union, matching columns by name rather than position."""
        keyword = "UNION ALL BY NAME" if all else "UNION BY NAME"
        return Frame(
            lambda left, right, **_: f"SELECT * FROM {left} {keyword} SELECT * FROM {right}",
            (self, other),
        )

    def intersect(self, other: Frame) -> Frame:
        """Rows present in both frames."""
        return Frame(
            lambda left, right, **_: f"SELECT * FROM {left} INTERSECT SELECT * FROM {right}",
            (self, other),
            _identity,
        )

    def except_(self, other: Frame) -> Frame:
        """Rows in this frame and not the other."""
        return Frame(
            lambda left, right, **_: f"SELECT * FROM {left} EXCEPT SELECT * FROM {right}",
            (self, other),
            _identity,
        )

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
        it lands in, so however many times it is used, it is computed once.

            budget = totals.aggregate(col("amount").mean().alias("m"))
            orders.filter(col("amount") > budget.scalar())
        """
        return SubQuery(self)

    # -- execution

    def _sql_and_values(
        self,
        wrap: Callable[[str], str] | None = None,
        shapes: dict[int, Shape] | None = None,
        parameters: Mapping[str, object] | None = None,
    ) -> tuple[str, list[Any] | None]:
        """The SQL for this frame and the values it binds.

        Rendered inside a sink, so lifted literals are bound as `$n` instead of
        being written into the text. A `param(name)` takes its value from
        `parameters`; every name must be supplied, and every name supplied must
        be used, so a typo cannot pass silently either way.
        """
        with ParamSink() as sink:
            sql = self._render(shapes)
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

    def _resolution(self, connection: Connection) -> dict[int, Shape] | None:
        """The shapes execution renders with, worked out on this connection.

        Every time, because it is what tells a join which columns clash and
        what refuses a step that would produce one name twice, and both
        answers belong to the catalog being read. A lone source is the one
        plan that needs none: its text is the query, and some statements the
        engine will run it cannot describe in advance (PIVOT is one).
        """
        return None if not self._inputs and not self._uses else self._shapes(connection)

    def _execute(self, connection: Connection, parameters: Mapping[str, object] | None) -> LiveResult:
        connection = self._on(connection)
        sql, values = self._sql_and_values(shapes=self._resolution(connection), parameters=parameters)
        return connection._execute(sql, values)

    def _run(self, connection: Connection, wrap: Callable[[str], str], parameters: Mapping[str, object] | None) -> int:
        """Run a statement built around this frame, reporting rows changed."""
        connection = self._on(connection)
        sql, values = self._sql_and_values(wrap, shapes=self._resolution(connection), parameters=parameters)
        with connection._execute(sql, values) as result:
            return result.drain()

    def fetchall(
        self, connection: Connection, *, parameters: Mapping[str, object] | None = None
    ) -> list[tuple[Any, ...]]:
        """Every row, as tuples."""
        with self._execute(connection, parameters) as result:
            return result.fetch_all()

    def fetchone(
        self, connection: Connection, *, parameters: Mapping[str, object] | None = None
    ) -> tuple[Any, ...] | None:
        """The first row, or None if there are none."""
        with self._execute(connection, parameters) as result:
            rows = result.fetch_rows(1)
        return rows[0] if rows else None

    def rows(
        self, connection: Connection, *, parameters: Mapping[str, object] | None = None
    ) -> Iterator[tuple[Any, ...]]:
        """Every row, a batch at a time."""
        with self._execute(connection, parameters) as result:
            while batch := result.fetch_rows(1024):
                yield from batch

    def count(self, connection: Connection, *, parameters: Mapping[str, object] | None = None) -> int:
        """How many rows this plan produces. Runs a count."""
        counted = self._derive(
            lambda source, **_: f"SELECT count(*) FROM {source}", shape=lambda _: (Column("count", "BIGINT"),)
        )
        row = counted.fetchone(connection, parameters=parameters)
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

    def to_parquet(
        self, connection: Connection, path: str, *, parameters: Mapping[str, object] | None = None, **options: object
    ) -> int:
        """Write a Parquet file. Returns how many rows were written."""
        return self._copy(connection, path, {"format": "parquet", **options}, parameters)

    def to_csv(
        self, connection: Connection, path: str, *, parameters: Mapping[str, object] | None = None, **options: object
    ) -> int:
        """Write a CSV file. Returns how many rows were written."""
        return self._copy(connection, path, {"format": "csv", **options}, parameters)

    def _copy(
        self, connection: Connection, path: str, options: dict[str, object], parameters: Mapping[str, object] | None
    ) -> int:
        """COPY TO. The path is a literal because COPY will not take a parameter."""
        rendered = ", ".join(f"{_option(k)} {render_literal(v)}" for k, v in options.items())
        return self._run(connection, lambda q: f"COPY ({q}) TO {render_literal(path)} ({rendered})", parameters)

    # -- looking at it

    def explain(
        self, connection: Connection, *, analyze: bool = False, parameters: Mapping[str, object] | None = None
    ) -> str:
        """The query plan, as text.

        With `analyze` the query runs and the plan carries what each step cost.
        """
        connection = self._on(connection)
        keyword = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
        sql, values = self._sql_and_values(
            lambda q: f"{keyword} {q}", shapes=self._resolution(connection), parameters=parameters
        )
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
        rows = head.fetchall(connection, parameters=parameters)
        return _box(head.columns(connection), head.types(connection), rows[:limit], more=len(rows) > limit)

    def __repr__(self) -> str:
        """The SQL. A plan holds no connection, so there are no rows to show."""
        try:
            rendered = self.render()
        except ValueError as reason:
            # The one step that cannot render blind is a suffixed join. A repr
            # that raised would make a debugger useless, so say why instead.
            return f"<Frame, renders with a connection: {reason}>"
        first = rendered.splitlines()
        shown = first[0] if len(first) == 1 else f"{first[0]} ... ({len(first)} lines)"
        return f"<Frame {shown}>"


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
    return Frame(lambda **_: query)


def table(name: str) -> Frame:
    """A plan reading a table or view, by name.

    The name is quoted, so a name from outside the program cannot turn into
    more SQL. What the table holds is the catalog's to say, and is asked of
    whichever connection the plan runs on.
    """
    source = f"SELECT * FROM {qualified(name)}"
    return Frame(lambda **_: source)


def declare(name: str, heading: Iterable[tuple[str, str | None]]) -> Frame:
    """A relation the caller will supply, with the columns it will have.

    A plan built on one is a function over relations: it works out its
    columns from the heading with no engine, renders with the hole as a bare
    name, and is closed with `plan.bind(name=some_plan)`. Executed with the
    hole still open, the catalog supplies the name.

        lineitem = declare("lineitem", [("l_shipdate", "DATE"), ("l_quantity", "DECIMAL(15,2)")])
        recent = lineitem.filter(col("l_shipdate") > date(1998, 9, 1))
        recent.bind(lineitem=table("lineitem")).fetchall(con)
    """
    shape = tuple(Column(column, type_text) for column, type_text in heading)
    source = f"SELECT * FROM {qualified(name)}"
    return Frame(lambda **_: source, declared=Declared(name, _require_unique(shape, "declare")))
