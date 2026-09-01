"""A strict PEP 249 interface, isolated from the native surface.

The execute/fetch/cursor vocabulary exists because PEP 249 asks for it, not
because it is the natural way to use DuckDB. Keeping it in its own module means
the native surface owes it nothing, and neither inherits the other's habits.

Cursors share their connection's engine connection, so they share its
transaction, which is what PEP 249 requires. The previous client gave every
cursor its own connection and so could not honour that.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from . import _duckdb
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
    """A DB-API type object: equal to every engine type it stands for."""

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
    "TIME WITH TIME ZONE",
    "TIMESTAMP",
    "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP_S",
    "TIMESTAMP_MS",
    "TIMESTAMP_NS",
    "INTERVAL",
)
#: DuckDB has no row identifier type, so nothing ever equals this.
ROWID = _TypeSet("ROWID")

Date = datetime.date
Time = datetime.time
Timestamp = datetime.datetime


def DateFromTicks(ticks: float) -> datetime.date:
    """The date of a Unix timestamp."""
    # date.fromtimestamp has no tz parameter and reads local time, so the
    # date is taken from an aware datetime instead.
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
        """Rows affected by the last statement, or -1 when not applicable.

        Real counts, where the previous client always reported -1.
        """
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
        """Run one statement.

        The engine allows one live result per connection, and cursors share
        theirs, so this releases whatever result is open first, including one
        belonging to a sibling cursor. PEP 249 permits that; it is what ODBC
        does without MARS.
        """
        connection = self._require_open()
        connection._claim_result_slot(self)
        connection._begin_if_needed()
        # Cleared before the call, not after: if it raises there is no last
        # query, and leaving the previous one's metadata in place would make
        # description and rowcount describe a statement that never ran.
        self._description = None
        self._rowcount = -1
        result = connection._engine().execute(operation, parameters)

        if result.result_type == "rows":
            self._result = result
            self._description = [(name, type_text, None, None, None, None, None) for name, type_text in result.schema]
            return self

        # Anything that is not a row-producing statement is drained here and
        # now. Side effects land on drain, so a result dropped without draining
        # means the INSERT simply never happened. Draining also yields a real
        # rowcount, where the previous client always reported -1.
        try:
            self._rowcount = result.drain()
        finally:
            result.close()
        self._result = None
        self._description = None
        connection._release_cursor(self)
        return self

    def executemany(self, operation: str, seq_of_parameters: Sequence[Parameters]) -> Cursor:
        """Run one statement once per parameter set.

        DuckDB has no batched bind, so the sets run in order. `rowcount` is the
        total across them, which is what sqlite3 and most drivers report; PEP
        249 leaves it undefined only when the statement produces rows, and it
        stays -1 in that case.
        """
        # A statement run zero times is still the last one asked for, so the
        # metadata of whatever ran before must not survive it.
        self._require_open()._claim_result_slot(self)
        self._description = None
        self._rowcount = -1
        total = 0
        counted = False
        for parameters in seq_of_parameters:
            self.execute(operation, parameters)
            if self._rowcount >= 0:
                total += self._rowcount
                counted = True
        self._rowcount = total if counted else -1
        return self

    # -- fetching

    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, or None when the result is exhausted."""
        rows = self._require_result().fetch_rows(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int | None = None) -> list[tuple[Any, ...]]:
        """Up to `size` rows, defaulting to `arraysize`."""
        count = self.arraysize if size is None else size
        if count < 0:
            message = "fetchmany size cannot be negative"
            raise ProgrammingError(message)
        if count == 0:
            return []
        return self._require_result().fetch_rows(count)

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every remaining row."""
        return self._require_result().fetch_all()

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
        if self._result is not None:
            self._result.close()
            self._result = None

    def close(self) -> None:
        """Release the cursor. Idempotent, as PEP 249 requires."""
        if self._connection is not None:
            self._connection._release_cursor(self)
        self._release_result()
        self._description = None
        self._connection = None

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

    def _engine(self) -> _duckdb.Connection:
        """The engine connection, or a clear error once closed."""
        if self._raw is None:
            message = "connection is closed"
            raise InterfaceError(message)
        return self._raw

    def _claim_result_slot(self, cursor: Cursor) -> None:
        """Release whoever holds the connection's single result slot, then take it."""
        if self._open_cursor is not None:
            self._open_cursor._release_result()
        self._open_cursor = cursor

    def _release_cursor(self, cursor: Cursor) -> None:
        if self._open_cursor is cursor:
            self._open_cursor = None

    def _run(self, sql: str) -> None:
        """Run a statement of our own, without disturbing a cursor's result.

        Drained, not just closed: a statement's effect lands when its result is
        drained, so closing BEGIN without draining leaves no transaction open
        and every later rollback silently does nothing.
        """
        result = self._engine().execute(sql)
        try:
            result.drain()
        finally:
            result.close()

    def _begin_if_needed(self) -> None:
        """Open a transaction lazily, so a read-only session never starts one."""
        if self._autocommit or self._in_transaction:
            return
        self._run("BEGIN TRANSACTION")
        self._in_transaction = True

    def _release_open_result(self) -> None:
        """Release whichever cursor holds the result slot, if any."""
        if self._open_cursor is not None:
            self._open_cursor._release_result()
            self._open_cursor = None

    def cursor(self) -> Cursor:
        """A new cursor sharing this connection's transaction."""
        self._engine()
        return Cursor(self)

    def commit(self) -> None:
        """Commit the open transaction, if there is one."""
        self._release_open_result()
        if self._in_transaction:
            self._run("COMMIT")
            self._in_transaction = False

    def rollback(self) -> None:
        """Discard the open transaction, if there is one."""
        self._release_open_result()
        if self._in_transaction:
            self._run("ROLLBACK")
            self._in_transaction = False

    def close(self) -> None:
        """Close the connection, discarding any open transaction.

        PEP 249: closing without committing rolls back.
        """
        if self._raw is None:
            return
        try:
            self.rollback()
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
        # PEP 249 leaves this to the driver. Commit on success, roll back on
        # error, which is what every caller means by `with connection:`.
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


def connect(database: str = ":memory:", *, autocommit: bool = False, **options: str) -> Connection:
    """Open a connection.

    Args:
        database: A database file, or ":memory:".
        autocommit: When false, statements run inside a transaction that
            commit() ends and close() rolls back.
        **options: Settings applied when the database is opened. Some can only
            be chosen here, before the database exists.
    """
    engine = _duckdb.Database(database, [(k, str(v)) for k, v in options.items()])
    return Connection(engine.connect(), autocommit=autocommit)
