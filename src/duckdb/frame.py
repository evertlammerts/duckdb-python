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
from typing import TYPE_CHECKING, Any

from .exceptions import Error
from .expr import Expr, ParamSink, SubQuery, col, qualified, quote, render_literal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from . import _duckdb

__all__ = ["Frame"]


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
    ) -> None:
        self._connection = connection
        #: SQL for this step. A callable, taking the rendered names of this
        #: step's inputs, so that expressions render while the parameter sink
        #: is active. Rendering them when the verb was called would bake their
        #: literals into the SQL text instead of binding them.
        self._body: Callable[..., str] = (lambda *names, _t=body: _t.format(*names)) if isinstance(body, str) else body
        self._inputs = inputs
        self._cached_schema: list[tuple[str, str]] | None = None

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

    def _derive(self, template: str | Callable[..., str], *inputs: Frame) -> Frame:
        return Frame(self._connection, template, inputs or (self,))

    # -- schema, from the binder

    @property
    def schema(self) -> list[tuple[str, str]]:
        """The columns this frame produces, as (name, type).

        Answered by the binder, not predicted here. Rendered with literals in
        place so it can infer their types.
        """
        if self._cached_schema is None:
            output, _ = self._connection.bind(self.render())
            self._cached_schema = output
        return self._cached_schema

    @property
    def columns(self) -> list[str]:
        """The column names."""
        return [name for name, _ in self.schema]

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

        return self._derive(body)

    def select(self, *columns: object) -> Frame:
        """Keep only these columns or expressions.

        Pure projection: it does not group, unlike the older client's `select`.
        """
        chosen = _as_exprs(list(columns))

        def body(source: str) -> str:
            return f"SELECT {', '.join(e.as_select() for e in chosen)} FROM {source}"

        return self._derive(body)

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

        return self._derive(body)

    def drop(self, *columns: str) -> Frame:
        """Remove columns."""
        rendered = ", ".join(quote(c) for c in columns)
        return self._derive(f"SELECT * EXCLUDE ({rendered}) FROM {{0}}")

    def rename(self, **columns: str) -> Frame:
        """Rename columns, given as old=new."""
        rendered = ", ".join(f"{quote(old)} AS {quote(new)}" for old, new in columns.items())
        return self._derive(f"SELECT * RENAME ({rendered}) FROM {{0}}")

    def sort(self, *columns: object) -> Frame:
        """Order the rows. Use `.desc()` and `.nulls_last()` on the columns."""
        ordering = _as_exprs(list(columns))

        def body(source: str) -> str:
            return f"SELECT * FROM {source} ORDER BY {', '.join(e.as_order() for e in ordering)}"

        return self._derive(body)

    def limit(self, count: int, offset: int = 0) -> Frame:
        """Keep at most `count` rows, optionally skipping `offset` first."""
        tail = f" OFFSET {int(offset)}" if offset else ""
        return self._derive(f"SELECT * FROM {{0}} LIMIT {int(count)}{tail}")

    def head(self, count: int = 5) -> Frame:
        """The first `count` rows."""
        return self.limit(count)

    def offset(self, count: int) -> Frame:
        """Skip `count` rows."""
        return self._derive(f"SELECT * FROM {{0}} OFFSET {int(count)}")

    def distinct(self, on: Iterable[object] | object | None = None) -> Frame:
        """Remove duplicate rows, or duplicates of `on` only."""
        if on is None:
            return self._derive("SELECT DISTINCT * FROM {0}")
        keys = _as_exprs(on)

        def body(source: str) -> str:
            return f"SELECT DISTINCT ON ({', '.join(e.fragment() for e in keys)}) * FROM {source}"

        return self._derive(body)

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
        return self._derive(f"SELECT * FROM {{0}} USING SAMPLE {size} ({arguments})")

    def unnest(self, *columns: str) -> Frame:
        """Expand list columns to one row per element, repeating the rest.

        Unnesting two columns together walks them in step rather than making
        every pairing; the shorter one runs out as NULL.
        """
        if not columns:
            message = "unnest needs at least one column"
            raise TypeError(message)
        expanded = ", ".join(f"unnest({quote(c)}) AS {quote(c)}" for c in columns)
        return self._derive(f"SELECT * REPLACE ({expanded}) FROM {{0}}")

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
        return self._derive(
            f"SELECT * FROM (UNPIVOT (SELECT * FROM {{0}}) ON {folded} INTO NAME {quote(name)} VALUE {quote(value)})"
        )

    def aggregate(self, *aggregates: object, group_by: Iterable[object] | object = ()) -> Frame:
        """Group and aggregate. Group keys come first in the output."""
        keys = _as_exprs(group_by) if group_by else []
        selected = [k.as_select() for k in keys] + [e.as_select() for e in _as_exprs(list(aggregates))]
        clause = " GROUP BY " + ", ".join(k.fragment() for k in keys) if keys else ""
        return self._derive(f"SELECT {', '.join(selected)} FROM {{0}}{clause}")

    def group_by(self, *columns: object) -> GroupedFrame:
        """Begin a grouped aggregation. Continue with `.agg(...)`."""
        return GroupedFrame(self, _as_exprs(list(columns)))

    def join(
        self,
        other: Frame,
        on: Expr | str | Iterable[str] | None = None,
        how: str = "inner",
    ) -> Frame:
        """Join to another frame.

        `on` is a column name, a list of them for a USING join, or an
        expression for an ON join. Inside that expression the two sides are
        `l` and `r`, so `col("l.id") == col("r.order_id")` is unambiguous even
        when both carry the same column name.

        `how` is inner, left, right, outer, semi, anti, cross, positional,
        natural or asof.
        """
        keyword = {"outer": "FULL OUTER", "semi": "SEMI", "anti": "ANTI"}.get(how.lower(), how.upper())
        if on is None and how.lower() not in {"cross", "natural", "positional"}:
            message = f"a {how} join needs `on`"
            raise TypeError(message)

        # Narrowed here rather than inside the closure, where mypy cannot see
        # that the None case was already refused above.
        keys: list[str] = []
        if on is not None and not isinstance(on, (Expr, str)):
            keys = [quote(c) for c in list(on)]

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
            return f"SELECT * FROM {left} AS l {keyword} JOIN {right} AS r{clause}"

        return Frame(self._connection, body, (self, other))

    def cross(self, other: Frame) -> Frame:
        """Every combination of rows from both frames."""
        return self.join(other, how="cross")

    def union(self, other: Frame, *, all: bool = True) -> Frame:
        """Rows from both frames, keeping duplicates unless `all` is false."""
        keyword = "UNION ALL" if all else "UNION"
        return Frame(self._connection, f"SELECT * FROM {{0}} {keyword} SELECT * FROM {{1}}", (self, other))

    def union_by_name(self, other: Frame, *, all: bool = True) -> Frame:
        """Union, matching columns by name rather than position."""
        keyword = "UNION ALL BY NAME" if all else "UNION BY NAME"
        return Frame(self._connection, f"SELECT * FROM {{0}} {keyword} SELECT * FROM {{1}}", (self, other))

    def intersect(self, other: Frame) -> Frame:
        """Rows present in both frames."""
        return Frame(self._connection, "SELECT * FROM {0} INTERSECT SELECT * FROM {1}", (self, other))

    def except_(self, other: Frame) -> Frame:
        """Rows in this frame and not the other."""
        return Frame(self._connection, "SELECT * FROM {0} EXCEPT SELECT * FROM {1}", (self, other))

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
        counted = self._derive("SELECT count(*) FROM {0}")
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
