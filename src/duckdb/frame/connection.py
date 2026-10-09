"""Connections, and running statements. Queries are built without one and take a connection when they run."""

from __future__ import annotations

import contextlib
import os
import threading
import weakref
from typing import TYPE_CHECKING, Any

from .. import _duckdb
from .._sources import NumpyScanSource, Source, adapt
from ..exceptions import Error, InterfaceError, InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from types import TracebackType

__all__ = ["Connection", "connect"]


class LiveResult:
    """A result still being read, tracked so `close()` can reach it: an open result keeps the database open."""

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


class LiveArrowStream:
    """An Arrow stream still being read, tracked by its connection and consumable through the capsule protocol.

    Tracked strongly, unlike a `LiveResult`: the capsule's consumer holds no Python reference, and a stream
    `close()` cannot reach is a query the engine ends quietly, so a read past the close would look complete.
    """

    __slots__ = ("_owner", "stream")

    def __init__(self, stream: _duckdb.ArrowStream, owner: Connection) -> None:
        self.stream = stream
        self._owner = weakref.ref(owner)

    def __arrow_c_stream__(self, requested_schema: object | None = None) -> Any:  # noqa: ANN401
        return self.stream.__arrow_c_stream__(requested_schema)

    @property
    def error(self) -> tuple[int, str] | None:
        """What ended the stream, or None; codes above zero are the engine's."""
        return self.stream.error

    def close(self) -> None:
        self.stream.close()
        if self.stream.close_pending:
            # Deferred by a close on an engine thread: stay tracked, so the connection completes it.
            return
        owner = self._owner()
        if owner is not None:
            with owner._live_lock:
                owner._live_streams.discard(self)

    def __enter__(self) -> LiveArrowStream:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _Catalog:
    """A counter of catalog changes, shared by connections to one database so each sees a sibling's change."""

    def __init__(self) -> None:
        self.generation = 0
        self.lock = threading.Lock()

    def changed(self) -> None:
        """Record a statement that may have changed which columns or functions a name refers to."""
        with self.lock:
            self.generation += 1


class Connection:
    """A connection to a database."""

    def __init__(self, database: _duckdb.Database, catalog: _Catalog | None = None) -> None:
        self._database: _duckdb.Database | None = database
        self._raw: _duckdb.Connection | None = database.connect()
        #: Shared by duplicate() only, so those connections see each other's changes and separately opened ones do not.
        self._catalog = catalog if catalog is not None else _Catalog()
        #: Column types DuckDB reported without running a query, kept per connection since settings change the answer.
        self._stub_answers: dict[str, object] = {}
        self._stub_lock = threading.Lock()
        self._stub_generation = self._catalog.generation
        #: Results still being read; weak so finished ones drop out, guarded because close() may run in another thread.
        self._live: weakref.WeakSet[LiveResult] = weakref.WeakSet()
        #: Open Arrow streams, held strongly: their consumers keep no Python reference close() could find.
        self._live_streams: set[LiveArrowStream] = set()
        self._live_lock = threading.Lock()

    def _finish_pending_closes(self) -> None:
        """Complete closes that an engine thread could only request.

        A close reached from inside a query (a scalar function, a source callback) cancels and defers its
        teardown; when nothing will ever read that stream again, this connection is the cleanup owner of last
        resort, finishing the job before it runs anything else.
        """
        with self._live_lock:
            pending = [held for held in self._live_streams if held.stream.close_pending]
        for held in pending:
            held.close()

    def _execute(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> LiveResult:
        """Run a statement and track its result; every execution comes through here."""
        self._finish_pending_closes()
        if _may_change_binding(sql):
            self._catalog.changed()
        return self._track(self._engine().execute(sql, parameters))

    def _track(self, result: _duckdb.Result) -> LiveResult:
        live = LiveResult(result)
        with self._live_lock:
            if self._raw is None:
                # An untracked result would keep the database open past a close that already returned.
                live.close()
                message = "connection is closed"
                raise InterfaceError(message)
            self._live.add(live)
        return live

    def _execute_arrow(
        self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None, batch_size: int
    ) -> LiveArrowStream:
        """Run a statement into an Arrow stream and track it, like `_execute` tracks a result."""
        self._finish_pending_closes()
        if _may_change_binding(sql):
            self._catalog.changed()
        live = LiveArrowStream(self._engine().execute_arrow(sql, parameters, batch_size), self)
        with self._live_lock:
            if self._raw is None:
                live.stream.close()
                message = "connection is closed"
                raise InterfaceError(message)
            # Streams whose consumer finished or released them are done; pruning here bounds the set.
            self._live_streams = {held for held in self._live_streams if held.stream.live}
            self._live_streams.add(live)
        return live

    def _engine(self) -> _duckdb.Connection:
        # Read once: close() may run in another thread between a check and a second read.
        raw = self._raw
        if raw is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return raw

    def run(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> int:
        """Run a statement and report how many rows it changed; use `sql()` for statements that produce rows."""
        # A result left open by a failed run would block the connection's next statement.
        with self._execute(sql, parameters) as result:
            return result.drain()

    def interrupt(self) -> None:
        """Cancel the query this connection is running, from another thread; Ctrl-C in the running one does the same."""
        self._engine().interrupt()

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the block as one transaction: COMMIT on success, ROLLBACK on any error."""
        self.run("BEGIN TRANSACTION")
        try:
            yield
        except BaseException:
            # A failing rollback must not hide the error that caused it.
            with contextlib.suppress(Error):
                self.run("ROLLBACK")
            raise
        else:
            self.run("COMMIT")

    def create_macro(
        self,
        name: str | tuple[str, ...],
        parameters: Iterable[str | tuple[str, object]],
        body: object,
        *,
        replace: bool = False,
        temporary: bool = False,
    ) -> None:
        """Define a macro from an expression (scalar) or a query (table); literals are written into the body.

        Args:
            name: The macro name, or a tuple for a schema-qualified one.
            parameters: Parameter names, or (name, default) pairs; `col(name)` in the body refers to one.
            body: The expression or query the macro stands for; a `param()` in it is refused.
            replace: Redefine a macro of that name if one exists.
            temporary: Make the macro last only for this database session.
        """
        from .._expressions.expr import (
            Expr,
            identifier,
            name_parts,
            quote,
            refusing_parameters,
            render_literal,
            suspended_sinks,
        )
        from .plan import Frame

        signature = ", ".join(
            quote(p) if isinstance(p, str) else f"{quote(p[0])} := {render_literal(p[1])}" for p in parameters
        )
        refusal = "param() has no value inside a macro body; a macro parameter is col(name)"
        with suspended_sinks(), refusing_parameters(refusal):
            if isinstance(body, Expr):
                definition = body.fragment()
            elif isinstance(body, Frame):
                # Asks DuckDB only for the parts of the plan that need their input's columns, since a parameter
                # only exists once the macro does.
                definition = "TABLE " + body._definition(self)
            else:
                message = f"a macro body is an expression or a plan, not {type(body).__name__}"
                raise TypeError(message)
        prefix = "CREATE OR REPLACE" if replace else "CREATE"
        kind = "TEMP MACRO" if temporary else "MACRO"
        self.run(f"{prefix} {kind} {identifier(name_parts(name, 'macro name'))}({signature}) AS {definition}")

    def create_function(
        self,
        name: str,
        function: Callable[..., Any],
        parameters: Iterable[str],
        returns: str,
        *,
        null_handling: str = "default",
        stability: str = "consistent",
    ) -> None:
        """Register a Python callable as a scalar SQL function; a macro is orders of magnitude faster where it fits.

        Args:
            name: The function's bare, unqualified name.
            function: Called once per row, from DuckDB's own threads.
            parameters: SQL type texts the arguments are cast to before the call.
            returns: The SQL type text the returned value is cast to.
            null_handling: "default" makes a NULL argument a NULL result without a call; "special" passes None in.
            stability: "consistent" lets DuckDB reuse results, "volatile" never, "consistent_within_query" per query.
        """
        if not isinstance(name, str):
            message = (  # type: ignore[unreachable]
                f"a function's name is a string, not {name!r}: the engine registers a Python function by its "
                f"bare name, so it cannot be schema-qualified"
            )
            raise TypeError(message)
        nulls = _NULL_HANDLING.get(null_handling)
        if nulls is None:
            message = "Invalid Input Error: null_handling must be 'default' or 'special'"
            raise InvalidInputError(message)
        level = _STABILITY.get(stability)
        if level is None:
            message = "Invalid Input Error: stability must be 'consistent', 'volatile' or 'consistent_within_query'"
            raise InvalidInputError(message)
        # ANY would leave arguments uncast while values are read by their declared type; DuckDB refuses it obscurely.
        texts = list(parameters)
        if any(_is_any(text) for text in texts):
            message = "Invalid Input Error: ANY parameters are not supported yet"
            raise InvalidInputError(message)
        if _is_any(returns):
            message = "Invalid Input Error: an ANY return type is not supported yet"
            raise InvalidInputError(message)
        self._engine().create_scalar_function(name, function, texts, returns, nulls, level)
        # A new function changes what a name in a later query can refer to.
        self._catalog.changed()

    def register(self, name: str, obj: object) -> None:
        """Make a Python object readable as the table `name`, on every connection to this database.

        Accepted: anything exporting an Arrow stream or array (a pyarrow Table, RecordBatch or Array, a polars
        DataFrame or Series, a pandas Series), a pyarrow Dataset or Scanner, a polars LazyFrame, a pandas DataFrame
        (its index left out), a numpy array or a dict, list or tuple of one-dimensional numpy arrays, a
        RecordBatchReader, or an Arrow stream capsule. A reader or a capsule is a stream, read once; register it again
        to read it again. Everything else is read as often as it is queried, a LazyFrame by collecting it per query.
        Registering a name again replaces the earlier object, and a real table of that name takes precedence.
        """
        if not isinstance(name, str):
            message = f"a registered name is a string, not {name!r}"  # type: ignore[unreachable]
            raise TypeError(message)
        self._register_source(name, adapt(obj))

    def _register_source(self, name: str, source: Source) -> None:
        """Register a source as `name` as it is, without adapting it."""
        self._engine().register_object(name, source, source.one_shot, isinstance(source, NumpyScanSource))
        # A new name changes what a later query resolves it to.
        self._catalog.changed()

    def unregister(self, name: str) -> None:
        """Forget the object registered as `name`; a query bound over it before this still runs."""
        if not self._engine().unregister_object(name):
            message = f"Invalid Input Error: nothing is registered as {name!r}"
            raise InvalidInputError(message)
        self._catalog.changed()

    def duplicate(self) -> Connection:
        """A second connection to the same database, with its own transaction; a subclass duplicates as itself."""
        database = self._database
        if database is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return type(self)(database, self._catalog)

    def close(self) -> None:
        """Close the connection and release the database. Idempotent, and results still being read are closed too."""
        with self._live_lock:
            pending: list[LiveResult | LiveArrowStream] = [*self._live, *self._live_streams]
            self._live.clear()
            # The streams stay tracked: each discards itself as its close completes, so a Ctrl-C breaking out
            # mid-loop leaves the rest for a repeated close() instead of losing them.
            # Marked closed under the lock, so a result tracked from now on is refused, not orphaned. The engine
            # connection stays alive in `raw` until the results are closed: ending a stream mid-read cancels its
            # query through the connection, which must still exist for that.
            raw = self._raw
            self._raw = None
            self._database = None
        failures: list[BaseException] = []
        try:
            for live in pending:
                try:
                    live.close()
                except Exception as error:  # every result must be tried
                    failures.append(error)
        finally:
            # A Ctrl-C breaking out of a stream's close must still release the engine connection.
            if raw is not None:
                raw.close()
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


_NULL_HANDLING = {
    "default": _duckdb.FunctionNullHandling.DEFAULT,
    "special": _duckdb.FunctionNullHandling.SPECIAL,
}

_STABILITY = {
    "consistent": _duckdb.FunctionStability.CONSISTENT,
    "volatile": _duckdb.FunctionStability.VOLATILE,
    "consistent_within_query": _duckdb.FunctionStability.CONSISTENT_WITHIN_QUERY,
}


def _is_any(text: str) -> bool:
    """Whether a type text is the word ANY, trimmed of ASCII blanks and in any case."""
    return text.strip(" \t\n\r\f\v").upper() == "ANY"


#: First words of a statement that only reads; anything else may change what a later query resolves to.
_READ_ONLY = frozenset({"SELECT", "WITH", "FROM", "VALUES", "DESCRIBE", "SUMMARIZE", "EXPLAIN", "SHOW"})


def _may_change_binding(sql: str) -> bool:
    """Whether a statement could change what a later query resolves names to, judged by its first keyword."""
    words = sql.upper().split()
    # EXPLAIN ANALYZE runs the statement it wraps, so the wrapped one is what counts.
    if words and words[0] == "EXPLAIN":
        words = words[1:]
        if words and words[0] == "ANALYZE":
            words = words[1:]
    return bool(words) and words[0] not in _READ_ONLY


def connect(database: str | os.PathLike[str] = ":memory:", **options: str) -> Connection:
    """Open a connection to a database file or ":memory:", applying any settings given as keyword arguments."""
    return Connection(_duckdb.Database(os.fspath(database), [(k, str(v)) for k, v in options.items()]))
