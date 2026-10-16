"""The appender both connection kinds hand out: rows buffered on the client and written by one statement per flush."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING

from . import _duckdb
from ._expressions.expr import _as_names, _items, fold_name, name_parts
from .exceptions import Error, InterfaceError, NotSupportedError

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from .frame.typing import PythonValue, _AppenderRows


class Appender:
    """Rows buffered on the client and written to the database by one statement at each `flush()`, and never otherwise.

    A flush that fails drops its rows and raises, except one the connection refuses because it is busy and one
    interrupted, which keep them for a retry; `clear()` drops them too. Letting go of an appender that still holds rows
    drops them with a `ResourceWarning`. Each call converts its values as a query parameter is converted. One caller
    at a time: a second, from another thread or from inside a flush, raises `OperationalError` instead of waiting.
    A `Decimal` of a width and scale this process has not converted before cannot be appended from a Python function
    running inside a query on the appender's connection: the call waits for that query and never returns.
    """

    __slots__ = ("_flush", "_raw")

    def __init__(self, raw: _duckdb.Appender, flush: Callable[[_duckdb.Appender], None]) -> None:
        self._raw = raw
        #: Its connection's way of running a flush, which holds that connection alive as long as the appender is.
        self._flush = flush

    def append_row(self, *values: PythonValue) -> None:
        """Buffer one row, one value per column in the appender's order; a failed value buffers nothing of the row."""
        self._raw.append_row(values)

    def append_rows(self, rows: _AppenderRows) -> None:
        """Buffer each row of `rows`, read lazily; the first that fails raises, and the rows before it stay buffered.

        A row is any sequence of values but a string, bytes or a mapping.
        """
        self._raw.append_rows(rows)

    def flush(self) -> None:
        """Write every buffered row with one statement on the appender's connection, then empty the buffer.

        It commits by itself, or joins the transaction the connection holds open; with nothing buffered nothing runs.
        """
        self._flush(self._raw)

    def clear(self) -> None:
        """Drop every buffered row, and make an appender whose buffer failed usable again."""
        self._raw.clear()

    def __enter__(self) -> Appender:
        if self._raw.closed:
            message = "connection is closed"
            raise InterfaceError(message)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Flush on a normal exit; on an exception, drop the rows and let the exception propagate."""
        if exc_type is None:
            self.flush()
            return
        # A refusal raised here would replace the exception that ended the block.
        with contextlib.suppress(Error):
            self._raw.clear()

    def __del__(self) -> None:
        raw = getattr(self, "_raw", None)
        if raw is not None and raw.buffered:
            message = (
                f"an appender was let go with {raw.buffered} rows it never flushed, so they were dropped; "
                "call flush() or clear() before letting go of it"
            )
            warnings.warn(message, ResourceWarning, stacklevel=2, source=self)


def _read_once(names: object) -> object:
    """A list of names read now, before anything runs, since reading it may run code; checked later.

    A string, bytes, a mapping and anything that is not iterable are kept as they are, for the checks to refuse.
    """
    if names is None or isinstance(names, (str, bytes, Mapping)):
        return names
    items = _items(names)
    return names if items is None else tuple(items)


def _read_declared(columns: object) -> object:
    """A query appender's columns read now, as (name, type) pairs in order, before anything runs; checked later."""
    if isinstance(columns, Mapping):
        return tuple((name, columns[name]) for name in columns)
    return columns


_ACTIONS = {
    "insert": _duckdb.TableAction.INSERT_ROWS,
    "upsert": _duckdb.TableAction.UPSERT_ROWS,
    "update": _duckdb.TableAction.UPDATE_ROWS,
    "delete": _duckdb.TableAction.DELETE_ROWS,
}
_ACTION_WORDS = '"insert", "upsert", "update" or "delete"'
_MATCHES_ON_KEY = (_duckdb.TableAction.UPDATE_ROWS, _duckdb.TableAction.DELETE_ROWS)


def _column_names(names: object, argument: str) -> tuple[str, ...] | None:
    """`columns` or `key` of a table appender as names, read already; a mapping is refused, its values unused."""
    if names is None:
        return None
    if isinstance(names, Mapping):
        message = (
            f"{argument} takes names of the table's columns, not a mapping; the table gives each column's type, so "
            "its values would go unused"
        )
        raise TypeError(message)
    return _as_names(names, "column name")


def _table_appender(
    engine: _duckdb.Connection, table: object, columns: object, action: object, key: object
) -> _duckdb.Appender:
    """The engine's appender for the table form, its arguments checked after the connection and its transaction.

    `columns` and `key` were read already. The checks here need no table; those against the table are the engine's.
    """
    engine.check_transaction()
    parts = name_parts(table, "table name")
    if len(parts) > 3:
        message = f"a table name has at most three parts, catalog, schema and table, not {len(parts)}: {table!r}"
        raise ValueError(message)
    names = _column_names(columns, "columns")
    if not isinstance(action, str):
        message = f"action is one of the strings {_ACTION_WORDS}, not {action!r}"
        raise TypeError(message)
    word = fold_name(action)
    kind = _ACTIONS.get(word)
    if kind is None:
        message = f"action is {_ACTION_WORDS}, not {action!r}"
        raise ValueError(message)
    keyed = _column_names(key, "key")
    matches_on_key = kind in _MATCHES_ON_KEY
    if keyed is not None and not matches_on_key:
        message = f"key applies to the update and delete actions, not to {word}"
        raise ValueError(message)
    if matches_on_key and not keyed:
        message = f"the {word} action needs key, the columns that pick the rows to {word}"
        raise ValueError(message)
    if kind is _duckdb.TableAction.DELETE_ROWS and names is not None:
        message = "the delete action buffers exactly its key columns, so it takes no columns"
        raise ValueError(message)
    return engine.appender(parts, names, kind, keyed)


def _query_appender(engine: _duckdb.Connection, dml: object, columns: object, source: object) -> _duckdb.Appender:
    """The engine's appender for the query form, its arguments checked after the connection and its transaction."""
    engine.check_transaction()
    message = "the query form is not available"
    raise NotSupportedError(message)
