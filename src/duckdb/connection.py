"""Connecting, and running statements.

A connection is somewhere to run things, and nothing else holds one. Plans are
built with `duckdb.table` and `duckdb.sql`, which need no connection, and are
run by passing one: `plan.rows(con)`, or `plan.on(con).rows()`.

`run()` executes a statement and reports how many rows it changed. There is no
cursor here and no fetch family; those belong to PEP 249 and live in
`duckdb.dbapi`.
"""

from __future__ import annotations

import contextlib
import threading
import weakref
from typing import TYPE_CHECKING, Any

from . import _duckdb
from .exceptions import Error, InterfaceError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from types import TracebackType

__all__ = ["Connection", "connect"]


class LiveResult:
    """A result a plan is reading, known to its connection.

    Exists so that `close()` can reach every result still open. The engine
    handle behind a result is what keeps a database open, and a result held by
    a paused iterator would otherwise outlive the connection that made it.
    """

    __slots__ = ("__weakref__", "result")

    def __init__(self, result: _duckdb.Result) -> None:
        self.result = result

    def fetch_all(self) -> list[tuple[Any, ...]]:
        return self.result.fetch_all()

    def fetch_rows(self, count: int) -> list[tuple[Any, ...]]:
        return self.result.fetch_rows(count)

    def drain(self) -> int:
        return self.result.drain()

    def close(self) -> None:
        self.result.close()

    def __enter__(self) -> LiveResult:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _Catalog:
    """What connections to one database share: a count of the statements that may have changed it.

    A stub answer depends on the functions and types the catalog holds, and
    a sibling connection can change those. Each connection remembers the
    count its answers were given under and forgets them when it has moved.
    """

    def __init__(self) -> None:
        self.generation = 0
        self.lock = threading.Lock()

    def changed(self) -> None:
        """Note a statement that may have changed what a query binds to."""
        with self.lock:
            self.generation += 1


class Connection:
    """A connection to a database."""

    def __init__(self, database: _duckdb.Database, catalog: _Catalog | None = None) -> None:
        self._database: _duckdb.Database | None = database
        self._raw: _duckdb.Connection | None = database.connect()
        #: Shared with the connections `duplicate()` makes from this one, so a
        #: change through any of them is seen by all. Two connections opened
        #: separately to one file share nothing here, and a change through
        #: one is not seen by the other until it runs a changing statement.
        self._catalog = catalog if catalog is not None else _Catalog()
        #: Answers to stub questions asked of this connection. A stub names no
        #: table, so its answer survives DDL, but it does depend on this
        #: connection's settings, which is why it is kept here and not shared.
        self._stub_answers: dict[str, object] = {}
        self._stub_lock = threading.Lock()
        self._stub_generation = self._catalog.generation
        #: Results plans are still reading. Weak, so a result that has been
        #: consumed and dropped leaves on its own; `close()` closes the rest.
        #: Guarded, because plans run from any thread and close from another.
        self._live: weakref.WeakSet[LiveResult] = weakref.WeakSet()
        self._live_lock = threading.Lock()

    def _execute(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> LiveResult:
        """Run a statement and track its result. Every execution comes through here.

        A statement can change what a stub answer depends on: SET
        integer_division changes what `x / y` binds to, LOAD adds functions,
        CREATE MACRO redefines one. A query cannot, so a plan's own SELECT
        leaves the answers alone and anything else forgets them.
        """
        if _may_change_binding(sql):
            self._catalog.changed()
        return self._track(self._engine().execute(sql, parameters))

    def _track(self, result: _duckdb.Result) -> LiveResult:
        live = LiveResult(result)
        with self._live_lock:
            if self._raw is None:
                # Closed between this result's execute and now. A result left
                # untracked would hold the engine past a close that returned.
                live.close()
                message = "connection is closed"
                raise InterfaceError(message)
            self._live.add(live)
        return live

    def _engine(self) -> _duckdb.Connection:
        if self._raw is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return self._raw

    def run(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> int:
        """Run a statement and report how many rows it changed.

        For statements that produce rows, use `sql()` instead.
        """
        # Closed on the way out whatever happens: a failure in drain would
        # otherwise leave the result open, and one live result per connection
        # means the next statement could not start.
        with self._execute(sql, parameters) as result:
            return result.drain()

    def interrupt(self) -> None:
        """Cancel the query this connection is running.

        Made to be called from another thread while a query runs; the query
        fails with `InterruptError`. A Ctrl-C in the querying thread does
        the same without this being called.
        """
        self._engine().interrupt()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the block as one transaction: COMMIT on success, ROLLBACK on any error.

        Statements and plans inside the block run on this connection, so
        they share the transaction. The engine refuses a nested BEGIN in
        its own words.
        """
        self.run("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            # A rollback that fails must not hide the error that caused it;
            # a closed connection has discarded the transaction already.
            with contextlib.suppress(Error):
                self.run("ROLLBACK")
            raise
        else:
            self.run("COMMIT")

    def create_macro(
        self,
        name: str,
        parameters: Iterable[str | tuple[str, object]],
        body: object,
        *,
        replace: bool = False,
        temporary: bool = False,
    ) -> None:
        """Define a macro from an expression or a plan.

        An expression makes a scalar macro, a plan a table macro. `parameters`
        are names, or (name, default) pairs. The body is rendered as written,
        with `col(name)` referring to a parameter, and the engine checks it
        when the macro is defined. Literals in the body are written into it,
        since a definition has no parameters to bind.
        """
        from .expr import Expr, qualified, quote, render_literal, suspended_sinks
        from .frame import Frame, NeedsConnection

        signature = ", ".join(
            quote(p) if isinstance(p, str) else f"{quote(p[0])} := {render_literal(p[1])}" for p in parameters
        )
        with suspended_sinks():
            if isinstance(body, Expr):
                definition = body.fragment()
            elif isinstance(body, Frame):
                # Rendered blind first, so `col(name)` can mean a parameter
                # the engine only knows once the macro exists. A body that
                # needs its inputs' columns, a suffixed join, renders here.
                try:
                    definition = "TABLE " + body.render()
                except NeedsConnection:
                    definition = "TABLE " + body._definition(self)
            else:
                message = f"a macro body is an expression or a plan, not {type(body).__name__}"
                raise TypeError(message)
        prefix = "CREATE OR REPLACE" if replace else "CREATE"
        kind = "TEMP MACRO" if temporary else "MACRO"
        self.run(f"{prefix} {kind} {qualified(name)}({signature}) AS {definition}")

    def duplicate(self) -> Connection:
        """A second connection to the same database, with its own transaction."""
        if self._database is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return Connection(self._database, self._catalog)

    def close(self) -> None:
        """Close the connection, releasing the database. Idempotent.

        Every result still being read is closed first: an iterator paused
        half-way through `rows()` holds an engine handle, and closing has to
        mean the database is released, not released once that iterator is
        garbage collected.
        """
        with self._live_lock:
            pending = list(self._live)
            self._live.clear()
            # Marked closed under the lock, so a result tracked from now on is
            # refused rather than orphaned.
            self._raw = None
            self._database = None
        failures: list[BaseException] = []
        for live in pending:
            try:
                live.close()
            except Exception as error:  # every result must be tried
                failures.append(error)
        if failures:
            raise failures[0]

    def __enter__(self) -> Connection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


#: The first word of a statement that only reads. Anything else may change
#: what the binder would answer, so it forgets the stub answers.
_READ_ONLY = frozenset({"SELECT", "WITH", "FROM", "VALUES", "DESCRIBE", "SUMMARIZE", "EXPLAIN", "SHOW"})


def _may_change_binding(sql: str) -> bool:
    """Whether a statement could change how a later query binds.

    Decided by the first keyword, looking past `EXPLAIN` and `EXPLAIN
    ANALYZE`: the latter runs the statement it wraps, so `EXPLAIN ANALYZE
    SET ...` changes as much as the `SET` would.
    """
    words = sql.upper().split()
    if words and words[0] == "EXPLAIN":
        words = words[1:]
        if words and words[0] == "ANALYZE":
            words = words[1:]
    return bool(words) and words[0] not in _READ_ONLY


def connect(database: str = ":memory:", **options: str) -> Connection:
    """Open a connection.

    Args:
        database: A database file, or ":memory:".
        **options: Settings applied as the database is opened.
    """
    return Connection(_duckdb.Database(database, [(k, str(v)) for k, v in options.items()]))
