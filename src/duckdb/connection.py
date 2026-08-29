"""Connecting, and running statements.

A connection is somewhere to run things, and nothing else holds one. Plans are
built with `duckdb.table` and `duckdb.sql`, which need no connection, and are
run by passing one: `plan.fetchall(con)`.

`run()` executes a statement and reports how many rows it changed. There is no
cursor here and no fetch family; those belong to PEP 249 and live in
`duckdb.dbapi`.
"""

from __future__ import annotations

import threading
import weakref
from typing import TYPE_CHECKING, Any

from . import _duckdb
from .exceptions import InterfaceError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
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


class Connection:
    """A connection to a database."""

    def __init__(self, database: _duckdb.Database) -> None:
        self._database: _duckdb.Database | None = database
        self._raw: _duckdb.Connection | None = database.connect()
        #: Answers to stub questions asked of this connection. A stub names no
        #: table, so its answer survives DDL, but it does depend on this
        #: connection's settings, which is why it is kept here and not shared.
        self._stub_answers: dict[str, object] = {}
        self._stub_lock = threading.Lock()
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
            with self._stub_lock:
                self._stub_answers.clear()
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

    def duplicate(self) -> Connection:
        """A second connection to the same database, with its own transaction."""
        if self._database is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return Connection(self._database)

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
    """Whether a statement could change how a later query binds."""
    first = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
    return first not in _READ_ONLY


def connect(database: str = ":memory:", **options: str) -> Connection:
    """Open a connection.

    Args:
        database: A database file, or ":memory:".
        **options: Settings applied as the database is opened.
    """
    return Connection(_duckdb.Database(database, [(k, str(v)) for k, v in options.items()]))
