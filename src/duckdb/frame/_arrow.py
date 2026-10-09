"""Results leaving as Arrow: the table and reader egress over the engine's Arrow stream.

pyarrow is imported on use; the capsule protocol itself needs none of this module, so a consumer such as
polars reads a bound plan with no pyarrow installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from ..exceptions import InterfaceError, InvalidInputError, class_for_code

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .connection import Connection, LiveArrowStream

#: `ArrowStream.error` codes beside the engine's own, mirrored from the extension.
_CLOSED = -1
_KEYBOARD_INTERRUPT = -2


def _pyarrow() -> Any:
    try:
        import pyarrow
    except ImportError as error:
        message = "to_arrow and to_reader need pyarrow, which is not installed; the capsule protocol does not"
        raise ImportError(message) from error
    if not hasattr(pyarrow.RecordBatchReader, "from_stream"):
        message = "to_arrow and to_reader need pyarrow 15.0 or newer, for RecordBatchReader.from_stream"
        raise ImportError(message)
    return pyarrow


def _batch_size(batch_size: int | None) -> int:
    # The engine takes 0 as "use the default"; only this module says 0, so None stays the public spelling.
    if batch_size is None:
        return 0
    if batch_size <= 0:
        message = f"batch_size must be positive, not {batch_size}"
        raise InvalidInputError(message)
    return batch_size


def _raise_typed(stream: LiveArrowStream, error: Exception) -> NoReturn:
    """Re-raise a consumer's generic Arrow error as what actually ended the stream."""
    ended = stream.error
    if ended is None:
        raise error
    code, message = ended
    if code == _KEYBOARD_INTERRUPT:
        raise KeyboardInterrupt from error
    if code > 0:
        raise class_for_code(code)(message) from error
    raise InterfaceError(message) from error


def reader(connection: Connection, sql: str, values: Sequence[Any] | None, batch_size: int | None) -> Any:
    """A pyarrow RecordBatchReader over a fresh run; a failed read surfaces pyarrow's error with the engine's text."""
    pyarrow = _pyarrow()
    stream = connection._execute_arrow(sql, values, _batch_size(batch_size))
    try:
        return pyarrow.RecordBatchReader.from_stream(stream)
    except Exception as error:
        # from_stream reads the schema, which runs the statements an expanding one hides; a failure there
        # must free the connection, not leave the query live.
        stream.close()
        _raise_typed(stream, error)
    except BaseException:
        # A KeyboardInterrupt here would otherwise leave the query live and tracked.
        stream.close()
        raise


def table(connection: Connection, sql: str, values: Sequence[Any] | None) -> Any:
    """Every row as a pyarrow Table."""
    pyarrow = _pyarrow()
    # Closed on the way out: the table's arrays are independent of the stream, and the handle would otherwise
    # stay tracked until the connection closes.
    stream = connection._execute_arrow(sql, values, 0)
    try:
        return pyarrow.RecordBatchReader.from_stream(stream).read_all()
    except Exception as error:
        _raise_typed(stream, error)
    finally:
        stream.close()
