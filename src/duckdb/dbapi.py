"""A strict PEP 249 interface, kept in its own module so it and the rest of the package share no habits."""

from __future__ import annotations

import contextlib
import datetime
import os
import threading
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from . import _duckdb
from ._sources import NumpyScanSource, Source, adapt
from .exceptions import (
    DatabaseError,
    DataError,
    Error,
    IntegrityError,
    InterfaceError,
    InternalError,
    NotSupportedError,
    OperationalError,
    ProgrammingError,
    Warning,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

__all__ = [
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
    "Binary",
    "Connection",
    "Cursor",
    "DataError",
    "DatabaseError",
    "Date",
    "DateFromTicks",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "Time",
    "TimeFromTicks",
    "Timestamp",
    "TimestampFromTicks",
    "Warning",
    "apilevel",
    "connect",
    "paramstyle",
    "threadsafety",
]

#: PEP 249 compliance level.
apilevel = "2.0"

#: Threads may share the module, but not connections.
threadsafety = 1

#: DuckDB also accepts $1 and $name, but PEP 249 wants one answer.
paramstyle = "qmark"


# --- type objects ----------------------------------------------------------


class _TypeSet:
    """A DB-API type object: equal to every SQL type name it stands for."""

    def __init__(self, name: str, *members: str) -> None:
        self.name = name
        self._members = frozenset(members)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            # DECIMAL(18,3) and similar carry parameters; compare on the head.
            return other.split("(")[0].strip() in self._members
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name


STRING = _TypeSet("STRING", "VARCHAR", "CHAR", "TEXT", "ENUM", "UUID")
BINARY = _TypeSet("BINARY", "BLOB", "BIT")
NUMBER = _TypeSet(
    "NUMBER",
    "BOOLEAN",
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
)
DATETIME = _TypeSet(
    "DATETIME",
    "DATE",
    "TIME",
    "TIME_NS",
    "TIME WITH TIME ZONE",
    "TIMESTAMP",
    "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP_S",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "TIMESTAMPTZ_NS",
    "INTERVAL",
)
#: DuckDB has no row identifier type, so nothing ever equals this.
ROWID = _TypeSet("ROWID")

Date = datetime.date
Time = datetime.time
Timestamp = datetime.datetime


def DateFromTicks(ticks: float) -> datetime.date:
    """The date of a Unix timestamp."""
    # date.fromtimestamp reads local time, so the date comes from a UTC datetime instead.
    return datetime.datetime.fromtimestamp(ticks, tz=datetime.UTC).date()


def TimeFromTicks(ticks: float) -> datetime.time:
    """The time of day of a Unix timestamp."""
    return datetime.datetime.fromtimestamp(ticks, tz=datetime.UTC).time()


def TimestampFromTicks(ticks: float) -> datetime.datetime:
    """The moment of a Unix timestamp."""
    return datetime.datetime.fromtimestamp(ticks, tz=datetime.UTC)


def Binary(value: bytes | bytearray | memoryview) -> bytes:
    """Wrap a value for use as a binary parameter."""
    return bytes(value)


# --- cursor ----------------------------------------------------------------

Parameters = Sequence[Any] | Mapping[str, Any]

#: Statement types whose superseded remainder is dropped where it settled. Everything else drains to its
#: end: only the engine's binder could prove a non-SELECT read-only, and wrongly draining a read costs
#: time while wrongly cancelling a write fails the transaction.
_CANCEL_ON_SUPERSEDE = frozenset({"select", "explain"})

_BUSY = (
    "connection has an open result being driven by another call; "
    "finish that fetch, complete() or abort() the result, or use a separate connection"
)


class Cursor:
    """A PEP 249 cursor over its connection's transaction."""

    def __init__(self, connection: Connection) -> None:
        self._connection: Connection | None = connection
        self._result: _duckdb.Result | None = None
        self._description: list[tuple[Any, ...]] | None = None
        self._rowcount = -1
        #: How many rows fetchmany returns when not told otherwise.
        self.arraysize = 1

    # -- state

    @property
    def connection(self) -> Connection:
        """The connection this cursor belongs to."""
        return self._require_open()

    @property
    def description(self) -> list[tuple[Any, ...]] | None:
        """Column metadata for the last query, or None if it produced no rows."""
        return self._description

    @property
    def rowcount(self) -> int:
        """Rows affected by the last statement, or -1 when that does not apply."""
        return self._rowcount

    def _require_open(self) -> Connection:
        if self._connection is None:
            message = "cursor is closed"
            raise InterfaceError(message)
        return self._connection

    def _require_result(self) -> _duckdb.Result:
        self._require_open()
        if self._result is None:
            message = "no result set; call execute() first"
            raise InterfaceError(message)
        return self._result

    # -- execution

    def execute(self, operation: str, parameters: Parameters | None = None) -> Cursor:
        """Run one statement, first superseding any open result, since DuckDB allows one per connection.

        A row-returning statement settles: it runs to its first batch, so an early error or an interrupt
        raises here, and the rest waits for the fetches. Everything else runs to completion here.
        """
        connection = self._require_open()
        with connection._driving():
            # Cleared before anything runs: a failure must not leave the previous statement's metadata
            # looking current, and superseding the open result can itself fail loudly.
            self._description = None
            self._rowcount = -1
            try:
                connection._claim_result_slot(self)
                connection._begin_if_needed()
                result = connection._engine().execute(operation, parameters)
                try:
                    if result.result_type == "rows":
                        result.settle()
                        self._description = [
                            (name, type_text, None, None, None, None, None) for name, type_text in result.schema
                        ]
                        self._result = result
                        return self
                    self._rowcount = result.drain()
                except BaseException:
                    result.close()
                    raise
                result.close()
            except BaseException:
                connection._release_cursor(self)
                raise
        self._result = None
        connection._release_cursor(self)
        return self

    def executemany(self, operation: str, seq_of_parameters: Sequence[Parameters]) -> Cursor:
        """Run each parameter set in order; PEP 249 leaves rows undefined here, so none are kept."""
        connection = self._require_open()
        # A statement run zero times is still the last one asked for, so earlier metadata must not survive it.
        self._description = None
        self._rowcount = -1
        with connection._driving():
            connection._claim_result_slot(self)
        total = 0
        counted = False
        try:
            for parameters in seq_of_parameters:
                self.execute(operation, parameters)
                if self._rowcount >= 0:
                    total += self._rowcount
                    counted = True
        finally:
            try:
                # The last set's rows follow the supersede rule too, so a trailing RETURNING is never cut short.
                with connection._driving():
                    result, self._result = self._result, None
                    if result is not None:
                        connection._dispose_superseded(result)
            finally:
                self._description = None
                connection._release_cursor(self)
        self._rowcount = total if counted else -1
        return self

    # -- fetching

    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, or None when the result is exhausted."""
        result = self._require_result()
        with self._require_open()._driving():
            rows = result.fetch_rows(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int | None = None) -> list[tuple[Any, ...]]:
        """Up to `size` rows, defaulting to `arraysize`."""
        count = self.arraysize if size is None else size
        if count < 0:
            message = "fetchmany size cannot be negative"
            raise ProgrammingError(message)
        if count == 0:
            return []
        result = self._require_result()
        with self._require_open()._driving():
            return result.fetch_rows(count)

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every remaining row."""
        result = self._require_result()
        with self._require_open()._driving():
            return result.fetch_all()

    # -- disposal beyond PEP 249

    def complete(self) -> None:
        """Run the pending statement to its end without reading its rows; a no-op without one."""
        connection = self._connection
        if connection is None:
            return
        try:
            with connection._driving():
                result, self._result = self._result, None
                if result is None:
                    return
                try:
                    result.drain()
                finally:
                    result.close()
        finally:
            connection._release_cursor(self)

    def abort(self) -> None:
        """Drop the pending statement: unread rows never arrive, and a cancelled write fails an open transaction.

        A statement that settled keeps the side effects of what already ran; a no-op without a result.
        """
        connection = self._connection
        if connection is None:
            return
        try:
            with connection._driving():
                self._release_result()
        finally:
            connection._release_cursor(self)

    def __iter__(self) -> Cursor:
        return self

    def __next__(self) -> tuple[Any, ...]:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    # -- no-ops PEP 249 requires

    def setinputsizes(self, sizes: Sequence[Any]) -> None:
        """Ignored: DuckDB infers parameter types from the values."""

    def setoutputsize(self, size: int, column: int | None = None) -> None:
        """Ignored: DuckDB does not need output buffers sized ahead of time."""

    # -- lifecycle

    def _release_result(self) -> None:
        """Drop any open result, so the connection can run another query."""
        result, self._result = self._result, None
        if result is not None:
            result.close()

    def close(self) -> None:
        """Release the cursor, running out a pending write first. Idempotent, as PEP 249 requires.

        Refused while another call is driving the engine, with the cursor left open and usable.
        """
        connection = self._connection
        if connection is None:
            return
        with connection._driving():
            try:
                result, self._result = self._result, None
                if result is not None:
                    # The supersede rule, so routine cleanup never cuts a write short; abort() is the
                    # deliberate way to do that.
                    connection._dispose_superseded(result)
            finally:
                self._description = None
                self._connection = None
                connection._release_cursor(self)

    def __enter__(self) -> Cursor:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


# --- connection ------------------------------------------------------------


class Connection:
    """A PEP 249 connection. Its cursors share its transaction."""

    def __init__(self, raw: _duckdb.Connection, *, autocommit: bool = False) -> None:
        self._raw: _duckdb.Connection | None = raw
        self._autocommit = autocommit
        self._in_transaction = False
        #: The cursor currently holding the connection's one result slot.
        self._open_cursor: Cursor | None = None
        #: Set while a call drives the engine, so a re-entrant or concurrent call refuses instead of
        #: superseding work in flight or deadlocking; guarded only so misuse errors instead of racing.
        self._busy = False
        self._busy_lock = threading.Lock()

    @contextlib.contextmanager
    def _driving(self) -> Iterator[None]:
        """Hold the connection for a stretch of engine work; every other call refuses meanwhile."""
        claimed = False
        try:
            # The claim is taken inside the try, so a signal landing right after it still reaches the release.
            with self._busy_lock:
                if self._busy:
                    raise ProgrammingError(_BUSY)
                self._busy = True
                claimed = True
            yield
        finally:
            if claimed:
                with self._busy_lock:
                    self._busy = False

    def _engine(self) -> _duckdb.Connection:
        """The open DuckDB connection, or a clear error once it is closed."""
        if self._raw is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return self._raw

    def _dispose_superseded(self, result: _duckdb.Result) -> None:
        """Dispose of a superseded result: a plain read stops where it settled, anything else runs out first.

        An error in the drained remainder raises here, out of the superseding call, with the statement's
        effects applied as far as the engine got.
        """
        try:
            if result.statement_type not in _CANCEL_ON_SUPERSEDE:
                result.drain()
        finally:
            result.close()

    def _claim_result_slot(self, cursor: Cursor) -> None:
        """Dispose of whoever holds the connection's single result slot, then take it."""
        holder = self._open_cursor
        if holder is not None:
            result = holder._result
            holder._result = None
            if result is not None:
                self._dispose_superseded(result)
        self._open_cursor = cursor

    def _release_cursor(self, cursor: Cursor) -> None:
        if self._open_cursor is cursor:
            self._open_cursor = None

    def _run(self, sql: str) -> None:
        """Run a statement of our own; a BEGIN whose result is closed unread opens no transaction at all."""
        result = self._engine().execute(sql)
        try:
            result.drain()
        finally:
            result.close()

    def _begin_if_needed(self) -> None:
        """Open a transaction first: PEP 249 wants autocommit off, and DuckDB has no session-level switch for it."""
        if self._autocommit or self._in_transaction:
            return
        self._run("BEGIN TRANSACTION")
        self._in_transaction = True

    def _release_open_result(self, *, drain: bool) -> None:
        """Dispose of whichever cursor holds the result slot, if any.

        Draining follows the supersede rule, so a commit never cuts a write short; a rollback discards
        everything anyway, so it just cancels.
        """
        holder = self._open_cursor
        self._open_cursor = None
        if holder is None:
            return
        result = holder._result
        holder._result = None
        if result is None:
            return
        if drain:
            self._dispose_superseded(result)
        else:
            result.close()

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

    def unregister(self, name: str) -> None:
        """Forget the object registered as `name`."""
        if not self._engine().unregister_object(name):
            message = f"nothing is registered as {name!r}"
            raise ProgrammingError(message)

    def cursor(self) -> Cursor:
        """A new cursor sharing this connection's transaction."""
        self._engine()
        return Cursor(self)

    def interrupt(self) -> None:
        """Cancel the statement this connection is running, from another thread; it fails with `InterruptError`."""
        self._engine().interrupt()

    def commit(self) -> None:
        """Commit the open transaction, if there is one; a pending statement that may write runs out first."""
        self._engine()
        with self._driving():
            self._release_open_result(drain=True)
            if self._in_transaction:
                self._run("COMMIT")
                self._in_transaction = False

    def rollback(self) -> None:
        """Discard the open transaction, if there is one."""
        self._engine()
        with self._driving():
            self._release_open_result(drain=False)
            if self._in_transaction:
                self._run("ROLLBACK")
                self._in_transaction = False

    def close(self) -> None:
        """Close the connection; PEP 249 says an uncommitted transaction rolls back.

        Refused while another call is driving the engine, with the connection left open and usable:
        marking it closed under a live drive would strand that work and skip the rollback.
        """
        if self._raw is None:
            return
        with self._driving():
            self._release_open_result(drain=False)
            try:
                if self._in_transaction:
                    self._in_transaction = False
                    self._run("ROLLBACK")
            finally:
                self._raw = None

    def __enter__(self) -> Connection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # PEP 249 leaves this to the driver, and `with connection:` is meant as commit on success.
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def connect(database: str | os.PathLike[str] = ":memory:", *, autocommit: bool = False, **options: str) -> Connection:
    """Open a connection; unless `autocommit` is on, statements run inside a transaction that `commit()` ends."""
    engine = _duckdb.Database(os.fspath(database), [(k, str(v)) for k, v in options.items()])
    return Connection(engine.connect(), autocommit=autocommit)
