"""Results leaving as Arrow: the table, reader and polars egress over the engine's Arrow stream.

pyarrow and polars are imported on use; the capsule protocol itself needs neither, so any Arrow-native
consumer reads a bound plan with nothing extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

from ..exceptions import InterfaceError, InvalidInputError, NotSupportedError, OperationalError, class_for_code

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .connection import Connection, LiveArrowStream
    from .plan import Frame

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


def _polars() -> Any:
    try:
        import polars
    except ImportError as error:
        message = "to_polars needs polars, which is not installed"
        raise ImportError(message) from error
    version = tuple(int(part) for part in polars.__version__.split(".")[:2] if part.isdigit())
    if version < (1, 3):
        message = "to_polars needs polars 1.3 or newer, where the DataFrame constructor reads an Arrow stream"
        raise ImportError(message)
    return polars


def polars_frame(connection: Connection, sql: str, values: Sequence[Any] | None) -> Any:
    """Every row as a polars DataFrame, read batch by batch from the stream's capsule."""
    polars = _polars()
    # polars aborts on Arrow data it has no type for by panicking out of Rust, a BaseException.
    panics = getattr(polars.exceptions, "PanicException", ())
    stream = connection._execute_arrow(sql, values, 0)
    try:
        return polars.DataFrame(stream)
    except Exception as error:
        # polars raises its own error types; what ended the stream carries the engine's code.
        _raise_typed(stream, error)
    except panics as error:
        # A type polars lacks (INTERVAL, UNION) panics, and so does an empty result with a repeated name.
        message = f"polars cannot hold this result, panicking with: {error}; to_arrow and to_reader carry it"
        raise NotSupportedError(message) from error
    finally:
        stream.close()


def _plugins(polars: Any) -> Any:
    try:
        from polars.io import plugins
    except ImportError as error:
        message = f"to_polars(lazy=True) needs polars' IO plugins, which polars {polars.__version__} does not have"
        raise ImportError(message) from error
    return plugins


def _result_header(connection: Connection, sql: str, values: Sequence[Any] | None) -> list[tuple[str, str]]:
    """The names and types a statement's result carries, read before it runs where the statement allows."""
    with connection._execute(sql, values) as live:
        try:
            return live.result.schema
        except InvalidInputError:
            # The PIVOT family names its columns only once it has run.
            live.result.settle()
            return live.result.schema


def _busy(error: OperationalError) -> OperationalError:
    # Only a connection already running a query raises the bare class here; its subclasses mean other things.
    if type(error) is not OperationalError:
        return error
    message = (
        "Resource In Use Error: a LazyFrame runs its plan on its connection, which runs one query at a time, and "
        "polars scans the frames of one query at once; give each LazyFrame combined in one polars query a "
        "connection of its own, as Connection.duplicate() makes"
    )
    return OperationalError(message)


def polars_lazy(plan: Frame, connection: Connection, parameters: Mapping[str, object] | None) -> Any:
    """The plan as a polars LazyFrame whose scans run it on `connection`, with polars' pushdowns applied.

    polars trusts a source to apply the projection, the row limit and the predicate it hands over, and checks
    none of them, so every scan applies all three: the projection and the limit in the query, the predicate to
    the polars frame the query returns.
    """
    from .._expressions.expr import quote

    polars = _polars()
    plugins = _plugins(polars)
    sql, values = plan._sql_and_values(connection=connection, parameters=parameters)
    header = _result_header(connection, sql, values)
    names = [name for name, _ in header]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            # The eager path's error; polars itself panics on repeated names when the frame it is given is empty.
            message = f"could not create a new DataFrame: column with name '{name}' has more than one occurrence"
            raise polars.exceptions.DuplicateError(message)
        seen.add(name)

    def projected(columns: list[tuple[int, str]], limit: int | None) -> str:
        # By position, aliased back: DuckDB resolves a name regardless of case, to its first match, and a star
        # over the plan suffixes repeated names, so neither may stand between polars' names and the columns.
        select = ", ".join(f"#{position} AS {quote(name)}" for position, name in columns)
        return f"SELECT {select} FROM ({sql})" + (f" LIMIT {limit}" if limit is not None else "")

    every = list(enumerate(names, 1))
    # Through the eager path, so the dtypes are to_polars' and its refusals happen here, before any scan.
    schema = polars_frame(connection, projected(every, 0), values).schema
    position = {name: index for index, name in every}

    def scan(with_columns: list[str] | None, predicate: Any, n_rows: int | None, _batch_size: int | None) -> Any:
        if with_columns == []:
            # polars asks for no column only where it needs no row either, as for a select of literals.
            yield polars.DataFrame()
            return
        limit = n_rows if predicate is None else None
        if with_columns is None and limit is None:
            query = sql
        else:
            columns = every if with_columns is None else [(position[name], name) for name in with_columns]
            query = projected(columns, limit)
        try:
            # Positions and dtypes are the schema's, so a plan whose columns moved since must not be read by them.
            now = _result_header(connection, sql, values)
            if now != header:
                message = (
                    f"Invalid Input Error: the plan's columns changed since the LazyFrame was made, from {header} to "
                    f"{now}; make a new LazyFrame"
                )
                raise InvalidInputError(message)
            out = polars_frame(connection, query, values)
        except OperationalError as error:
            busy = _busy(error)
            if busy is error:
                raise
            raise busy from error
        if predicate is not None:
            out = out.filter(predicate)
            if n_rows is not None:
                out = out.head(n_rows)
        yield out

    return plugins.register_io_source(scan, schema=schema)
