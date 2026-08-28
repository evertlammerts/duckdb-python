"""Connecting, and running SQL.

`sql()` builds a query without running it. `run()` runs one and reports how
many rows it changed. There is no cursor here and no fetch family; those belong
to PEP 249 and live in `duckdb.dbapi`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import _duckdb
from .frame import Frame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

__all__ = ["Connection", "connect"]


class Connection:
    """A connection to a database."""

    def __init__(self, database: _duckdb.Database) -> None:
        self._database = database
        self._raw: _duckdb.Connection | None = database.connect()

    def _engine(self) -> _duckdb.Connection:
        if self._raw is None:
            message = "connection is closed"
            from .exceptions import InterfaceError

            raise InterfaceError(message)
        return self._raw

    def sql(self, query: str) -> Frame:
        """Build a query. Nothing runs until you ask the frame for rows."""
        return Frame(self._engine(), query)

    def run(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> int:
        """Run a statement and report how many rows it changed.

        For statements that produce rows, use `sql()` instead.
        """
        result = self._engine().execute(sql, parameters)
        changed = result.drain()
        result.close()
        return changed

    def duplicate(self) -> Connection:
        """A second connection to the same database, with its own transaction."""
        return Connection(self._database)

    def close(self) -> None:
        """Close the connection. Idempotent."""
        self._raw = None

    def __enter__(self) -> Connection:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def connect(database: str = ":memory:", **options: str) -> Connection:
    """Open a connection.

    Args:
        database: A database file, or ":memory:".
        **options: Settings applied as the database is opened.
    """
    return Connection(_duckdb.Database(database, [(k, str(v)) for k, v in options.items()]))
