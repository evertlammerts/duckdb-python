"""Results leaving as Arrow: the table, reader and polars egress over the engine's Arrow stream.

pyarrow and polars are imported on use; the capsule protocol itself needs neither, so any Arrow-native
consumer reads a bound plan with nothing extra installed.
"""

from __future__ import annotations

import functools
import io
import json
import math
import operator
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


def _raise_typed(stream: LiveArrowStream, error: BaseException) -> NoReturn:
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


def _version(module: Any) -> tuple[int, ...]:
    return tuple(int(part) for part in module.__version__.split(".")[:2] if part.isdigit())


def _polars() -> Any:
    try:
        import polars
    except ImportError as error:
        message = "to_polars needs polars, which is not installed"
        raise ImportError(message) from error
    if _version(polars) < (1, 3):
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
        # polars 1.36 and older panic on a stream that ended with an error, rather than raise.
        if stream.error is not None:
            _raise_typed(stream, error)
        # A type polars lacks (INTERVAL, UNION) panics, and so does an empty result with a repeated name.
        message = f"polars cannot hold this result, panicking with: {error}; to_arrow and to_reader carry it"
        raise NotSupportedError(message) from error
    finally:
        stream.close()


def _plugins(polars: Any) -> Any:
    # Before 1.22 polars applies no filter it fails to hand its IO source, and before 1.33 it cannot be told that
    # a source read twice in one query may be read once.
    if _version(polars) < (1, 33):
        message = f"to_polars(lazy=True) needs polars 1.33 or newer for its IO plugins, not {polars.__version__}"
        raise ImportError(message)
    from polars.io import plugins

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
        "polars scans the frames of one query at once, a frame used twice with different filters once per use; "
        "give each such scan a LazyFrame on a connection of its own, as Connection.duplicate() makes"
    )
    return OperationalError(message)


class _Untranslatable(Exception):
    """A predicate part with no exact SQL counterpart here; polars applies it instead."""


#: polars' comparison operators: their SQL spelling, and the operator read from the other side.
_COMPARISONS = {
    "Eq": ("=", "Eq"),
    "NotEq": ("<>", "NotEq"),
    "Lt": ("<", "Gt"),
    "LtEq": ("<=", "GtEq"),
    "Gt": (">", "Lt"),
    "GtEq": (">=", "LtEq"),
}

#: Bounds of polars' is_between: whether each end is included.
_BETWEEN = {"Both": (">=", "<="), "Left": (">=", "<"), "Right": (">", "<="), "None": (">", "<")}

#: Nesting past which a predicate stays with polars, well inside both Python's recursion limit and DuckDB's
#: max_expression_depth.
_DEPTH = 200


def _comparable(polars: Any, type_text: str, dtype: Any) -> bool:
    """Whether DuckDB compares a column exactly as polars does: its engine type and polars' dtype for it agree.

    The engine type decides, since polars sees a UUID or a JSON column as text it compares otherwise.
    """
    plain = {
        "BOOLEAN": polars.Boolean,
        "TINYINT": polars.Int8,
        "SMALLINT": polars.Int16,
        "INTEGER": polars.Int32,
        "BIGINT": polars.Int64,
        "UTINYINT": polars.UInt8,
        "USMALLINT": polars.UInt16,
        "UINTEGER": polars.UInt32,
        "UBIGINT": polars.UInt64,
        "FLOAT": polars.Float32,
        "DOUBLE": polars.Float64,
        "VARCHAR": polars.String,
        "DATE": polars.Date,
        "TIMESTAMP": polars.Datetime("us"),
    }.get(type_text)
    if plain is not None:
        return bool(dtype == plain)
    if type_text == "TIMESTAMP WITH TIME ZONE":
        return isinstance(dtype, polars.Datetime) and dtype.time_unit == "us" and dtype.time_zone is not None
    return False


class _Predicate:
    """SQL for the parts of a polars predicate that mean exactly the same in DuckDB, over positional aliases.

    Values are bound as neo's own literals are, numbered after the plan's.
    """

    def __init__(self, polars: Any, columns: dict[str, tuple[int, Any, bool]], values: Sequence[Any] | None) -> None:
        from .._expressions.expr import ParamSink

        self.polars = polars
        #: name: (position, dtype, comparable)
        self.columns = columns
        self.used: set[int] = set()
        self.sink = ParamSink()
        for value in values or ():
            self.sink.add_literal(value, None)

    @property
    def values(self) -> list[Any]:
        """The plan's values, then those the pushed SQL binds."""
        return [value for _, value, _ in self.sink.entries]

    def sql(self, node: object, depth: int = 0) -> str:
        if depth > _DEPTH:
            raise _Untranslatable
        kind, body = _node(node)
        if kind == "Column":
            position, dtype = self._column(node)
            if dtype != self.polars.Boolean:
                raise _Untranslatable
            return self._reference(position, dtype)
        if kind == "BinaryExpr" and isinstance(body, dict) and isinstance(body.get("op"), str):
            op = body["op"]
            if op in ("And", "Or"):
                left, right = self.sql(body.get("left"), depth + 1), self.sql(body.get("right"), depth + 1)
                return f"({left} {op.upper()} {right})"
            if op in _COMPARISONS:
                return self._compare(body.get("left"), op, body.get("right"))
        if kind == "Function" and isinstance(body, dict):
            return self._function(body.get("function"), body.get("input"), depth)
        raise _Untranslatable

    def _function(self, function: object, inputs: object, depth: int) -> str:
        if not isinstance(inputs, list) or not inputs:
            raise _Untranslatable
        if function == {"Boolean": "Not"} and len(inputs) == 1:
            return f"(NOT {self.sql(inputs[0], depth + 1)})"
        if function in ({"Boolean": "IsNull"}, {"Boolean": "IsNotNull"}) and len(inputs) == 1:
            position = self._any_column(inputs[0])
            return f'("_c{position}" IS {"NOT " if function == {"Boolean": "IsNotNull"} else ""}NULL)'
        boolean = function.get("Boolean") if isinstance(function, dict) else None
        if isinstance(boolean, dict) and isinstance(boolean.get("IsBetween"), dict) and len(inputs) == 3:
            low, high = _BETWEEN.get(str(boolean["IsBetween"].get("closed")), (None, None))
            if low is None or high is None:
                raise _Untranslatable
            position, dtype = self._column(inputs[0])
            column = self._reference(position, dtype)
            lower, upper = self._scalar(inputs[1], dtype), self._scalar(inputs[2], dtype)
            return f"({column} {low} {lower} AND {column} {high} {upper})"
        # nulls_equal makes a NULL match a NULL in the list, which SQL's IN never does.
        if boolean == {"IsIn": {"nulls_equal": False}} and len(inputs) == 2:
            position, dtype = self._column(inputs[0])
            return f"({self._reference(position, dtype)} IN ({', '.join(self._list(inputs[1], dtype))}))"
        raise _Untranslatable

    def _compare(self, left: object, op: str, right: object) -> str:
        if _node(left)[0] == "Column":
            column, literal, symbol = left, right, _COMPARISONS[op][0]
        elif _node(right)[0] == "Column":
            column, literal, symbol = right, left, _COMPARISONS[_COMPARISONS[op][1]][0]
        else:
            raise _Untranslatable
        position, dtype = self._column(column)
        return f"{self._reference(position, dtype)} {symbol} {self._scalar(literal, dtype)}"

    def _column(self, node: object) -> tuple[int, Any]:
        name = _node(node)[1]
        found = self.columns.get(name) if isinstance(name, str) else None
        if found is None or not found[2]:
            raise _Untranslatable
        return found[0], found[1]

    def _any_column(self, node: object) -> int:
        kind, name = _node(node)
        found = self.columns.get(name) if kind == "Column" and isinstance(name, str) else None
        if found is None:
            raise _Untranslatable
        self.used.add(found[0])
        return found[0]

    def _reference(self, position: int, dtype: Any) -> str:
        self.used.add(position)
        reference = f'"_c{position}"'
        # Byte order, polars' own, whatever collation the column or the session declares.
        return f"({reference} COLLATE C)" if dtype == self.polars.String else reference

    def _literal(self, node: object) -> Any:
        if _node(node)[0] != "Literal":
            raise _Untranslatable
        # polars evaluates its own literal, whatever its encoding: a list arrives as Arrow IPC bytes.
        try:
            source = io.StringIO(json.dumps(node))
            return self.polars.select(self.polars.Expr.deserialize(source, format="json")).to_series()
        except Exception as error:
            raise _Untranslatable from error

    def _scalar(self, node: object, dtype: Any) -> str:
        series = self._literal(node)
        if series.dtype != dtype or len(series) != 1:
            raise _Untranslatable
        return self._bound(_value(series[0]))

    def _list(self, node: object, dtype: Any) -> list[str]:
        series = self._literal(node)
        if series.dtype != self.polars.List(dtype) or len(series) != 1 or series[0] is None:
            raise _Untranslatable
        values = series[0].to_list()
        if not values:
            raise _Untranslatable
        return [self._bound(_value(value)) for value in values]

    def _bound(self, value: object) -> str:
        from .._expressions.expr import Lit

        # Text, bytes and dates bind as `$n`: written into the SQL, a NUL byte would end the statement there.
        with self.sink:
            return Lit(value).fragment()


def _node(node: object) -> tuple[object, object]:
    """A serialized expression node as (kind, body), or (None, None) for any other shape."""
    if isinstance(node, dict) and len(node) == 1:
        ((kind, body),) = node.items()
        return kind, body
    return None, None


def _value(value: object) -> object:
    """A literal's value, refusing what a comparison with it would not mean the same in both."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        raise _Untranslatable
    return value


def _conjuncts(expr: Any) -> list[tuple[Any, object]]:
    """The predicate's top-level AND operands, each as polars' own expression with its serialized form."""
    tree = json.loads(expr.meta.serialize(format="json"))
    kind, body = _node(tree)
    if kind == "BinaryExpr" and isinstance(body, dict) and body.get("op") == "And":
        return [part for side in expr.meta.pop() for part in _conjuncts(side)]
    return [(expr, tree)]


def _dynamic(node: object) -> bool:
    """A hint polars adds above a slice or a top-k: an optimisation only, and polars panics if asked to apply it."""
    kind, body = _node(node)
    return kind == "Display" and isinstance(body, dict) and str(body.get("fmt_str", "")).startswith("dynamic_pred")


def _split(
    polars: Any, predicate: Any, columns: dict[str, tuple[int, Any, bool]], values: Sequence[Any] | None
) -> tuple[list[str], set[int], Sequence[Any] | None, Any]:
    """The conjuncts DuckDB can apply as SQL, their columns, the values bound with them, and what polars applies."""
    try:
        parts = _conjuncts(predicate)
    except Exception:
        return [], set(), values, predicate
    translator = _Predicate(polars, columns, values)
    pushed: list[str] = []
    # polars' own expressions, not a rebuild from their serialized form, which a NaN literal cannot survive.
    kept: list[Any] = []
    for expr, tree in parts:
        if _dynamic(tree):
            continue
        try:
            pushed.append(translator.sql(tree))
        except _Untranslatable:
            kept.append(expr)
    residual = functools.reduce(operator.and_, kept) if kept else None
    return pushed, translator.used, translator.values, residual


def polars_lazy(plan: Frame, connection: Connection, parameters: Mapping[str, object] | None) -> Any:
    """The plan as a polars LazyFrame whose scans run it on `connection`, with polars' pushdowns applied.

    polars trusts a source to apply the projection, the row limit and the predicate it hands over, and checks
    none of them, so every scan applies all three: the projection and the limit in the query, the predicate's
    conjuncts that mean exactly the same in DuckDB there too, and the rest to the polars frame the query returns.
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

    def projected(
        columns: list[tuple[int, str]], limit: int | None, where: list[str] | None = None, used: set[int] | None = None
    ) -> str:
        # By position, aliased back: DuckDB resolves a name regardless of case, to its first match, and a star
        # over the plan suffixes repeated names, so neither may stand between polars' names and the columns.
        # polars' row limit takes the plan's first rows, and its predicate then filters those.
        tail = f" LIMIT {limit}" if limit is not None else ""
        if not where:
            select = ", ".join(f"#{position} AS {quote(name)}" for position, name in columns)
            return f"SELECT {select} FROM ({sql}){tail}"
        # A filter names its columns too, so it reads them under aliases no two of which can be confused.
        inner = ", ".join(f'#{p} AS "_c{p}"' for p in sorted({p for p, _ in columns} | (used or set())))
        outer = ", ".join(f'"_c{position}" AS {quote(name)}' for position, name in columns)
        return f"SELECT {outer} FROM (SELECT {inner} FROM ({sql}){tail}) WHERE {' AND '.join(where)}"

    every = list(enumerate(names, 1))
    # Through the eager path, so the dtypes are to_polars' and its refusals happen here, before any scan.
    schema = polars_frame(connection, projected(every, 0), values).schema
    position = {name: index for index, name in every}
    comparable = {
        name: (index, schema[name], _comparable(polars, type_text, schema[name]))
        for index, (name, type_text) in enumerate(header, 1)
    }

    def scan(with_columns: list[str] | None, predicate: Any, n_rows: int | None, _batch_size: int | None) -> Any:
        if with_columns == []:
            # polars asks for no column only where it needs no row either, as for a select of literals.
            yield polars.DataFrame()
            return
        pushed: list[str] = []
        used: set[int] = set()
        bound: Sequence[Any] | None = values
        residual = None
        if predicate is not None:
            pushed, used, bound, residual = _split(polars, predicate, comparable, values)
        if with_columns is None and n_rows is None and not pushed:
            query = sql
        else:
            columns = every if with_columns is None else [(position[name], name) for name in with_columns]
            query = projected(columns, n_rows, pushed, used)
        try:
            # Positions and dtypes are the schema's, so a plan whose columns moved since must not be read by them.
            now = _result_header(connection, sql, values)
            if now != header:
                message = (
                    f"Invalid Input Error: the plan's columns changed since the LazyFrame was made, from {header} to "
                    f"{now}; make a new LazyFrame"
                )
                raise InvalidInputError(message)
            out = polars_frame(connection, query, bound)
        except OperationalError as error:
            busy = _busy(error)
            if busy is error:
                raise
            raise busy from error
        yield out if residual is None else out.filter(residual)

    # Pure, so polars reads a frame used twice in one query once, rather than run two scans on one connection.
    return plugins.register_io_source(scan, schema=schema, is_pure=True)
