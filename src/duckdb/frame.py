"""A query you build up a step at a time.

A frame is a plan, not a result. Each verb returns a new frame, and nothing
runs until you ask for rows. The plan is a graph: reuse a frame in two places
and it is computed once.

    people.filter(col("age") > 30).select(col("name")).fetchall()

Rendering never consults a schema. Column names and types come from the
binder, through `.schema`, because it is the only thing that knows them.

Rows leave through `fetchall` and its neighbours, or through a sink, which
writes them to a table or a file without them passing through Python.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, NamedTuple

from .exceptions import Error
from .expr import Col, Expr, ParamSink, Star, SubQuery, col, qualified, quote, render_literal, suspended_sinks

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from . import _duckdb

__all__ = ["Column", "Frame"]


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


def _projection(chosen: list[Expr]) -> Callable[[tuple[Shape, ...]], Shape | None]:
    """Shape of a select list."""

    def rule(shapes: tuple[Shape, ...]) -> Shape | None:
        out: list[Column] = []
        for expression in chosen:
            columns = _contributed(expression, shapes[0])
            if columns is None:
                return None
            out.extend(columns)
        return tuple(out)

    return rule


_OPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _option(name: str) -> str:
    """A COPY option name. Checked, because it cannot be quoted."""
    if not _OPTION_NAME.match(name):
        message = f"not a copy option name: {name!r}"
        raise ValueError(message)
    return name.upper()


def _as_exprs(values: Iterable[object] | object) -> list[Expr]:
    """Accept an expression, a column name, or a sequence of either."""
    items = values if isinstance(values, (list, tuple)) else [values]
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


class Frame:
    """One step of a query. Immutable: every verb returns a new frame."""

    def __init__(
        self,
        connection: _duckdb.Connection,
        body: str | Callable[..., str],
        inputs: tuple[Frame, ...] = (),
        shape: Callable[[tuple[Shape, ...]], Shape | None] | None = None,
    ) -> None:
        self._connection = connection
        #: SQL for this step. A callable, taking the rendered names of this
        #: step's inputs, so that expressions render while the parameter sink
        #: is active. Rendering them when the verb was called would bake their
        #: literals into the SQL text instead of binding them.
        self._body: Callable[..., str] = (lambda *names, _t=body: _t.format(*names)) if isinstance(body, str) else body
        self._inputs = inputs
        #: How this step works out its own shape. None, or a rule that returns
        #: None, means the engine decides and the shape is bound instead.
        self._shape_rule = shape
        self._cached_shape: Shape | None = None

    # -- graph and rendering

    def _order(self) -> list[Frame]:
        """Every node this frame depends on, inputs before the nodes using them."""
        seen: dict[int, Frame] = {}
        order: list[Frame] = []

        def visit(node: Frame) -> None:
            if id(node) in seen:
                return
            seen[id(node)] = node
            for parent in node._inputs:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def render(self) -> str:
        """The whole query as SQL, with one CTE per step.

        A frame used twice appears once: the graph is walked by identity, so
        the engine sees the reuse rather than a duplicated subtree.
        """
        order = self._order()
        names = {id(node): f"_s{i}" for i, node in enumerate(order)}

        def body_of(node: Frame) -> str:
            return node._body(*(quote(names[id(p)]) for p in node._inputs))

        if len(order) == 1:
            return body_of(self)
        ctes = ",\n".join(f"{quote(names[id(n)])} AS (\n{body_of(n)}\n)" for n in order[:-1])
        return f"WITH {ctes}\n{body_of(order[-1])}"

    def _derive(
        self,
        template: str | Callable[..., str],
        *inputs: Frame,
        shape: Callable[[tuple[Shape, ...]], Shape | None] | None = None,
    ) -> Frame:
        return Frame(self._connection, template, inputs or (self,), shape)

    # -- shape: names derived here, types from the binder

    @property
    def shape(self) -> Shape:
        """What this step produces, in order.

        Names are derived wherever this library decided them, which is most
        verbs. Where DuckDB decides a name, such as the `(x * 2.5)` it invents
        for an unaliased computation, the step is bound instead. Types are left
        unknown until something asks, because nothing about rendering needs
        them.

        Computed once for the whole graph, inputs first, and cached per step.
        """
        if self._cached_shape is not None:
            return self._cached_shape
        computed: Shape = ()
        for node in self._order():
            if node._cached_shape is None:
                node._cached_shape = node._derive_shape()
            computed = node._cached_shape
        return computed

    def _derive_shape(self) -> Shape:
        """This step's shape, its inputs' shapes already being known."""
        if self._shape_rule is not None:
            given = tuple(node._cached_shape or () for node in self._inputs)
            derived = self._shape_rule(given)
            if derived is not None:
                return derived
        return self._bind_shape()

    def _bind_shape(self) -> Shape:
        """Ask the engine about this step alone.

        The inputs become empty tables of the right shape, so the query put to
        the binder stays small however long the chain is. Falls back to binding
        the whole thing when a type below is not known, since a stub cannot be
        written without one.
        """
        stubs, names = [], []
        for position, node in enumerate(self._inputs):
            shape = node._cached_shape
            if shape is None or any(column.type is None for column in shape):
                return self._bind_whole()
            columns = ", ".join(f"NULL::{column.type} AS {quote(column.name)}" for column in shape)
            stub = f"_in{position}"
            stubs.append(f"{quote(stub)} AS (SELECT {columns} WHERE FALSE)")
            names.append(quote(stub))
        with suspended_sinks():
            body = self._body(*names)
            sql = f"WITH {', '.join(stubs)}\n{body}" if stubs else body
            output, _ = self._connection.bind(sql)
        return tuple(Column(name, type_text) for name, type_text in output)

    def _bind_whole(self) -> Shape:
        """Bind the entire query.

        Correct always, but the cost grows with the chain, so it is the
        fallback rather than the way.
        """
        with suspended_sinks():
            output, _ = self._connection.bind(self.render())
        return tuple(Column(name, type_text) for name, type_text in output)

    @property
    def schema(self) -> list[tuple[str, str]]:
        """The columns this frame produces, as (name, type).

        Types are the engine's to say, so this asks it if it has not already.
        `.columns` alone usually costs nothing.
        """
        shape = self.shape
        if any(column.type is None for column in shape):
            self._cached_shape = shape = self._bind_shape()
        unresolved = [column.name for column in shape if column.type is None]
        if unresolved:  # pragma: no cover - the binder types everything it returns
            message = f"no type reported for {unresolved}"
            raise Error(message)
        return [(column.name, type_text) for column in shape if (type_text := column.type) is not None]

    @property
    def columns(self) -> list[str]:
        """The column names. Derived, so this usually asks the engine nothing."""
        return [column.name for column in self.shape]

    @property
    def types(self) -> list[str]:
        """The column types."""
        return [type_text for _, type_text in self.schema]

    def describe(self) -> Frame:
        """Per-column statistics, as a frame: count, min, max, average, quartiles.

        This reads the rows, unlike `.schema`, which only asks the binder.
        """
        # Wrapped in a SELECT because a bare SUMMARIZE cannot follow a WITH,
        # and every step but the first is reached through one.
        return self._derive("SELECT * FROM (SUMMARIZE SELECT * FROM {0})")

    # -- verbs

    def filter(self, predicate: Expr | str) -> Frame:
        """Keep the rows where `predicate` holds."""

        def body(source: str) -> str:
            rendered = predicate if isinstance(predicate, str) else predicate.fragment()
            return f"SELECT * FROM {source} WHERE {rendered}"

        return self._derive(body, shape=_identity)

    def select(self, *columns: object) -> Frame:
        """Keep only these columns or expressions.

        Pure projection: it does not group, unlike the older client's `select`.
        """
        chosen = _as_exprs(list(columns))

        def body(source: str) -> str:
            return f"SELECT {', '.join(e.as_select() for e in chosen)} FROM {source}"

        return self._derive(body, shape=_projection(chosen))

    def with_columns(self, **columns: object) -> Frame:
        """Add columns, replacing any of the same name."""
        chosen = {name: _as_exprs(value)[0] for name, value in columns.items()}
        existing = set(self.columns)

        def body(source: str) -> str:
            # A name already present is replaced in place, keeping column
            # order; a new one is appended. EXCLUDE cannot serve both, because
            # excluding a name that is not there is an error.
            replaced = [f"{e.fragment()} AS {quote(n)}" for n, e in chosen.items() if n in existing]
            appended = [e.alias(n).as_select() for n, e in chosen.items() if n not in existing]
            star = f"* REPLACE ({', '.join(replaced)})" if replaced else "*"
            return f"SELECT {', '.join([star, *appended])} FROM {source}"

        def shape(shapes: tuple[Shape, ...]) -> Shape:
            # A replaced column keeps its position, a new one is appended, and
            # both take their type from the engine because an expression made it.
            kept = [Column(c.name, None) if c.name in chosen else c for c in shapes[0]]
            return (*kept, *(Column(name, None) for name in chosen if name not in existing))

        return self._derive(body, shape=shape)

    def drop(self, *columns: str) -> Frame:
        """Remove columns."""
        rendered = ", ".join(quote(c) for c in columns)
        dropped = set(columns)
        return self._derive(
            f"SELECT * EXCLUDE ({rendered}) FROM {{0}}",
            shape=lambda shapes: tuple(c for c in shapes[0] if c.name not in dropped),
        )

    def rename(self, **columns: str) -> Frame:
        """Rename columns, given as old=new."""
        rendered = ", ".join(f"{quote(old)} AS {quote(new)}" for old, new in columns.items())
        return self._derive(
            f"SELECT * RENAME ({rendered}) FROM {{0}}",
            shape=lambda shapes: tuple(Column(columns.get(c.name, c.name), c.type) for c in shapes[0]),
        )

    def sort(self, *columns: object) -> Frame:
        """Order the rows. Use `.desc()` and `.nulls_last()` on the columns."""
        ordering = _as_exprs(list(columns))

        def body(source: str) -> str:
            return f"SELECT * FROM {source} ORDER BY {', '.join(e.as_order() for e in ordering)}"

        return self._derive(body, shape=_identity)

    def limit(self, count: int, offset: int = 0) -> Frame:
        """Keep at most `count` rows, optionally skipping `offset` first."""
        tail = f" OFFSET {int(offset)}" if offset else ""
        return self._derive(f"SELECT * FROM {{0}} LIMIT {int(count)}{tail}", shape=_identity)

    def head(self, count: int = 5) -> Frame:
        """The first `count` rows."""
        return self.limit(count)

    def offset(self, count: int) -> Frame:
        """Skip `count` rows."""
        return self._derive(f"SELECT * FROM {{0}} OFFSET {int(count)}", shape=_identity)

    def distinct(self, on: Iterable[object] | object | None = None) -> Frame:
        """Remove duplicate rows, or duplicates of `on` only."""
        if on is None:
            return self._derive("SELECT DISTINCT * FROM {0}", shape=_identity)
        keys = _as_exprs(on)

        def body(source: str) -> str:
            return f"SELECT DISTINCT ON ({', '.join(e.fragment() for e in keys)}) * FROM {source}"

        return self._derive(body, shape=_identity)

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
        return self._derive(f"SELECT * FROM {{0}} USING SAMPLE {size} ({arguments})", shape=_identity)

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
            f"SELECT * REPLACE ({expanded}) FROM {{0}}",
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
        # Wrapped for the same reason as `describe`: a bare UNPIVOT cannot
        # follow a WITH.
        folded_names = set(columns)
        return self._derive(
            f"SELECT * FROM (UNPIVOT (SELECT * FROM {{0}}) ON {folded} INTO NAME {quote(name)} VALUE {quote(value)})",
            shape=lambda shapes: (
                *(c for c in shapes[0] if c.name not in folded_names),
                Column(name, "VARCHAR"),
                Column(value, None),
            ),
        )

    def aggregate(self, *aggregates: object, group_by: Iterable[object] | object = ()) -> Frame:
        """Group and aggregate. Group keys come first in the output."""
        keys = _as_exprs(group_by) if group_by else []
        selected = [k.as_select() for k in keys] + [e.as_select() for e in _as_exprs(list(aggregates))]
        clause = " GROUP BY " + ", ".join(k.fragment() for k in keys) if keys else ""
        return self._derive(
            f"SELECT {', '.join(selected)} FROM {{0}}{clause}",
            shape=_projection([*keys, *_as_exprs(list(aggregates))]),
        )

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
        would then silently mean whichever came first. That is refused; pass
        `suffix` to rename the right side's copies instead.
        """
        keyword = {"outer": "FULL OUTER", "semi": "SEMI", "anti": "ANTI"}.get(how.lower(), how.upper())
        if on is None and how.lower() not in {"cross", "natural", "positional"}:
            message = f"a {how} join needs `on`"
            raise TypeError(message)

        # Narrowed here rather than inside the closure, where mypy cannot see
        # that the None case was already refused above.
        keys: list[str] = []
        using: list[str] = []
        if on is not None and not isinstance(on, (Expr, str)):
            using = [str(c) for c in list(on)]
            keys = [quote(c) for c in using]
        elif isinstance(on, str):
            using = [on]

        # A semi or anti join keeps only the left side, so nothing can collide.
        keeps_right = how.lower() not in {"semi", "anti"}
        shared: list[str] = []
        if keeps_right:
            # USING already folds its keys into one column, so those are not a
            # clash. Everything else two sides share is.
            left_names = {c.name for c in self.shape}
            shared = [c.name for c in other.shape if c.name in left_names and c.name not in set(using)]
        if shared and suffix is None:
            message = (
                f"both sides of this join have {', '.join(repr(n) for n in shared)}; "
                f"the result would carry each name twice and `col` could not tell them apart. "
                f"Pass suffix=... to rename the right side, or rename before joining."
            )
            raise ValueError(message)

        def body(left: str, right: str) -> str:
            if how.lower() in {"cross", "natural", "positional"}:
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
            folded = set(using)
            renamed = {n: n + str(suffix) for n in shared}
            carried = [Column(renamed.get(c.name, c.name), c.type) for c in right_shape if c.name not in folded]
            return (*left_shape, *carried)

        return Frame(self._connection, body, (self, other), shape)

    def cross(self, other: Frame) -> Frame:
        """Every combination of rows from both frames."""
        return self.join(other, how="cross")

    def union(self, other: Frame, *, all: bool = True) -> Frame:
        """Rows from both frames, keeping duplicates unless `all` is false."""
        keyword = "UNION ALL" if all else "UNION"
        return Frame(self._connection, f"SELECT * FROM {{0}} {keyword} SELECT * FROM {{1}}", (self, other), _identity)

    def union_by_name(self, other: Frame, *, all: bool = True) -> Frame:
        """Union, matching columns by name rather than position."""
        keyword = "UNION ALL BY NAME" if all else "UNION BY NAME"
        return Frame(self._connection, f"SELECT * FROM {{0}} {keyword} SELECT * FROM {{1}}", (self, other))

    def intersect(self, other: Frame) -> Frame:
        """Rows present in both frames."""
        return Frame(self._connection, "SELECT * FROM {0} INTERSECT SELECT * FROM {1}", (self, other), _identity)

    def except_(self, other: Frame) -> Frame:
        """Rows in this frame and not the other."""
        return Frame(self._connection, "SELECT * FROM {0} EXCEPT SELECT * FROM {1}", (self, other), _identity)

    def __getitem__(self, name: str) -> Frame:
        """A single column, as a frame."""
        return self.select(col(name))

    # -- as a value inside another query

    def scalar(self) -> Expr:
        """This query used where a single value is expected.

        It must return one row and one column. It is not correlated: it cannot
        see the columns of the query it lands in, so it is computed once.

            budget = totals.aggregate(col("amount").mean().alias("m"))
            orders.filter(col("amount") > budget.scalar())
        """
        return SubQuery(self)

    # -- execution

    def _bind(self, wrap: Callable[[str], str] | None = None) -> tuple[str, list[Any] | None]:
        """The SQL for this frame and the values it binds.

        Rendered inside a sink, so lifted literals are bound as `$n` instead of
        being written into the text.
        """
        with ParamSink() as sink:
            sql = self.render()
            if wrap is not None:
                sql = wrap(sql)
        values = [value if kind == "literal" else None for kind, value, _ in sink.entries]
        return sql, values or None

    def _execute(self) -> _duckdb.Result:
        sql, values = self._bind()
        return self._connection.execute(sql, values)

    def _run(self, wrap: Callable[[str], str]) -> int:
        """Run a statement built around this frame, reporting rows changed."""
        sql, values = self._bind(wrap)
        result = self._connection.execute(sql, values)
        try:
            return result.drain()
        finally:
            result.close()

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every row, as tuples."""
        return self._execute().fetch_all()

    def fetchone(self) -> tuple[Any, ...] | None:
        """The first row, or None if there are none."""
        rows = self._execute().fetch_rows(1)
        return rows[0] if rows else None

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        result = self._execute()
        while batch := result.fetch_rows(1024):
            yield from batch

    def __len__(self) -> int:
        """How many rows this frame produces. Runs a count."""
        counted = self._derive("SELECT count(*) FROM {0}", shape=lambda _: (Column("count", "BIGINT"),))
        row = counted.fetchone()
        return int(row[0]) if row else 0

    # -- writing the rows somewhere

    def create(self, name: str, *, replace: bool = False, temporary: bool = False) -> int:
        """Store the rows in a new table. Returns how many were written."""
        prefix = "CREATE OR REPLACE" if replace else "CREATE"
        kind = "TEMPORARY TABLE" if temporary else "TABLE"
        return self._run(lambda q: f"{prefix} {kind} {qualified(name)} AS {q}")

    def insert_into(self, name: str) -> int:
        """Append the rows to a table that exists. Returns how many were added."""
        return self._run(lambda q: f"INSERT INTO {qualified(name)} {q}")

    def to_parquet(self, path: str, **options: object) -> int:
        """Write a Parquet file. Returns how many rows were written."""
        return self._copy(path, {"format": "parquet", **options})

    def to_csv(self, path: str, **options: object) -> int:
        """Write a CSV file. Returns how many rows were written."""
        return self._copy(path, {"format": "csv", **options})

    def _copy(self, path: str, options: dict[str, object]) -> int:
        """COPY TO. The path is a literal because COPY will not take a parameter."""
        rendered = ", ".join(f"{_option(k)} {render_literal(v)}" for k, v in options.items())
        return self._run(lambda q: f"COPY ({q}) TO {render_literal(path)} ({rendered})")

    # -- looking at it

    def explain(self, *, analyze: bool = False) -> str:
        """The query plan, as text.

        With `analyze` the query runs and the plan carries what each step cost.
        """
        keyword = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
        sql, values = self._bind(lambda q: f"{keyword} {q}")
        rows = self._connection.execute(sql, values).fetch_all()
        return str(rows[0][1]) if rows else ""

    def show(self, limit: int = 10) -> None:
        """Print the first rows."""
        print(self.preview(limit))

    def preview(self, limit: int = 10) -> str:
        """The first rows drawn as a table. What `show` prints."""
        # One row past the limit, so the footer can say there are more without
        # counting them all.
        head = self.limit(limit + 1)
        rows = head.fetchall()
        return _box(head.columns, head.types, rows[:limit], more=len(rows) > limit)

    def __repr__(self) -> str:
        try:
            return self.preview()
        except Error as error:
            # A repr that raises is unusable in a debugger and a repr that hides
            # the reason is worse, so name it and carry on.
            reason = str(error).splitlines()[0] if str(error) else ""
            return f"<Frame, unrunnable: {type(error).__name__}: {reason}>"


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
