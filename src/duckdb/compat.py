"""The old client's connection surface, for migration.

The shipping client's `duckdb.connect()` returns a connection that executes
and fetches directly: `conn.execute(...).fetchall()`, with `description` and
`rowcount` on the connection, `cursor()` handing back a duplicate whose
lifetime is tied to its parent, and `sql()`/`table()`/`query()` returning
lazy relations. This face reproduces that contract over the new seam so
migrating code keeps running while it moves. The native surface and the
strict PEP 249 face in `duckdb.dbapi` stay what they are; new code has no
reason to be here.

Old habits are reproduced deliberately, the questionable ones included:
`cursor()` is an independent transaction unlike dbapi's shared one, closing
a connection closes its cursors, `rowcount` is always -1, relations compose
SQL text, `connect()` hands out connections into one shared database
instance per path (with the old config-mismatch refusal), and a module-level
default connection backs `execute()` and friends. What the face still lacks
is measured by the adopted suite under compat/, and every remaining
difference is a behavior-change-log entry, not an accident.
"""

from __future__ import annotations

import builtins
import itertools
import re
import sys
import threading
import weakref
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, cast

from . import _duckdb
from .connection import Connection, LiveResult, _Catalog
from .dbapi import BINARY, DATETIME, NUMBER, ROWID, STRING, apilevel, paramstyle, threadsafety
from .exceptions import (
    CatalogError,
    ConversionError,
    DatabaseError,
    DataError,
    Error,
    FatalError,
    IntegrityError,
    InterfaceError,
    InternalError,
    InterruptError,
    InvalidInputError,
    IOError,
    NotSupportedError,
    OperationalError,
    OutOfMemoryError,
    ParserError,
    ProgrammingError,
    TransactionError,
    Warning,
)
from .expr import qualified, quote

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    import pandas

    from .expr import Expr, ThenBuilder

__all__ = [
    "BINARY",
    "DATETIME",
    "NUMBER",
    "ROWID",
    "STRING",
    "BinderException",
    "CaseExpression",
    "CatalogException",
    "CoalesceOperator",
    "ColumnExpression",
    "CompatConnection",
    "CompatRelation",
    "ConnectionException",
    "ConstantExpression",
    "ConversionException",
    "DataError",
    "DatabaseError",
    "DefaultExpression",
    "Error",
    "Expression",
    "FatalException",
    "FunctionExpression",
    "HTTPException",
    "IOException",
    "IntegrityError",
    "InternalException",
    "InterruptException",
    "InvalidInputException",
    "InvalidTypeException",
    "NotImplementedException",
    "NotSupportedError",
    "OperationalError",
    "OutOfMemoryException",
    "OutOfRangeException",
    "ParserException",
    "PermissionException",
    "ProgrammingError",
    "SQLExpression",
    "SequenceException",
    "SerializationException",
    "StandardException",
    "StarExpression",
    "SyntaxException",
    "TransactionException",
    "TypeMismatchException",
    "Warning",
    "apilevel",
    "close",
    "connect",
    "default_connection",
    "description",
    "execute",
    "from_query",
    "paramstyle",
    "query",
    "set_default_connection",
    "sql",
    "table",
    "table_function",
    "threadsafety",
]

#: The old exception names, each aliased to the class this package raises for
#: the same engine error, or to the nearest ancestor when the old class drew a
#: finer line than the engine's error codes do. ConnectionException covered
#: "closed or unusable", which this package reports as InterfaceError.
BinderException = ProgrammingError
CatalogException = CatalogError
ConnectionException = InterfaceError
ConversionException = ConversionError
FatalException = FatalError
HTTPException = IOError
InternalException = InternalError
InterruptException = InterruptError
InvalidInputException = InvalidInputError
InvalidTypeException = ProgrammingError
IOException = IOError
NotImplementedException = NotSupportedError
OutOfMemoryException = OutOfMemoryError
OutOfRangeException = DataError
ParserException = ParserError
PermissionException = DatabaseError
SequenceException = DatabaseError
SerializationException = DatabaseError
StandardException = Error
SyntaxException = ParserError
TransactionException = TransactionError
TypeMismatchException = ProgrammingError

#: The old client advertised itself to the engine as python/<major.minor>.
__formatted_python_version__ = f"{sys.version_info.major}.{sys.version_info.minor}"


def _name(key: object) -> str:
    """A parameter name the old client's way: str as is, bytes decoded, anything else through str()."""
    if isinstance(key, str):
        return key
    if isinstance(key, bytes):
        return key.decode()
    return str(key)


#: Type texts describe() treats as numeric, mirroring the old client: those
#: get every summary statistic cast to DOUBLE, the rest get NULL for the
#: numeric-only ones and VARCHAR casts for the rest.
_NUMERIC_TYPES = frozenset({
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE", "DECIMAL",
})  # fmt: skip

#: A join condition that is only column names becomes USING, anything else
#: ON, the same reading the old client's parser applied.
_IDENTIFIER_LIST = re.compile(r'\s*[\w"]+(\s*,\s*[\w"]+)*\s*$')

_JOIN_KINDS = {
    "inner": "INNER JOIN",
    "left": "LEFT JOIN",
    "right": "RIGHT JOIN",
    "outer": "FULL JOIN",
    "semi": "SEMI JOIN",
    "anti": "ANTI JOIN",
}

#: Distinct default aliases per relation, so unaliased relations can join.
_relation_numbers = itertools.count()

#: Statement types the parser certifies as pure queries: those stay lazy
#: relations, re-running per bare fetchall as the old client's did. Anything
#: else executes exactly once at sql() time.
_LAZY_STATEMENTS = frozenset({"select", "explain", "relation", "logical_plan"})

_CLOSED_MESSAGE = "Connection Error: Connection has already been closed"


def _materialized_relation(
    connection: CompatConnection, schema: builtins.list[tuple[str, str]], rows: builtins.list[tuple[Any, ...]]
) -> CompatRelation:
    """Rows already produced, carried as a typed VALUES scan.

    The old client wrapped a RETURNING result as a materialized relation:
    fetches re-read the same rows and verbs compose on top. A VALUES scan
    of the rendered values gives both for free, since re-running it has no
    side effects. The text is O(rows) and re-parsed per run, so this suits
    the small row sets RETURNING produces, not bulk results.
    """
    from .expr import suspended_sinks
    from .frame import values as frame_values

    with suspended_sinks():
        sql = frame_values([tuple(row) for row in rows], builtins.list(schema)).render()
    return CompatRelation(connection, sql, kind="MATERIALIZED_RELATION")


def _quantile_parameter(q: object) -> str:
    if isinstance(q, str):
        return q
    if isinstance(q, (list, tuple)):
        return "[" + ", ".join(str(float(value)) for value in q) + "]"
    return str(float(cast("float", q)))


def _operand(value: object) -> str:
    """A verb operand as SQL text: expressions render, strings pass through."""
    from .expr import Expr, suspended_sinks

    if isinstance(value, CompatExpression):
        value = value._value()
    if isinstance(value, Expr):
        with suspended_sinks():
            return value.fragment()
    return str(value)


class CompatExpression:
    """An old-client expression: the native expression layer under the old names.

    Wraps a native `Expr` (or a CASE mid-build), so the semantics are the
    native layer's: a bare value operand is a value, matching the old
    Expression objects, whose divergent string-as-column reading lived in
    the text verbs, not here.
    """

    __slots__ = ("_case", "_expr")

    def __init__(self, expr: Expr | None = None, case: ThenBuilder | None = None) -> None:
        self._expr = expr
        self._case = case

    def _value(self) -> Expr:
        if self._expr is not None:
            return self._expr
        # A CASE nothing finished: unmatched rows are NULL, as the old
        # client's implicit else was.
        return cast("ThenBuilder", self._case).end()

    def __repr__(self) -> str:
        from .expr import suspended_sinks

        with suspended_sinks():
            return self._value().fragment()

    def _binary(self, other: object, operate: Callable[[Expr, Expr], Expr]) -> CompatExpression:
        return CompatExpression(operate(self._value(), _expression_operand(other)))

    def __eq__(self, other: object) -> CompatExpression:  # type: ignore[override]
        return self._binary(other, lambda a, b: a == b)

    def __ne__(self, other: object) -> CompatExpression:  # type: ignore[override]
        return self._binary(other, lambda a, b: a != b)

    __hash__ = None  # type: ignore[assignment]

    def __lt__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a < b)

    def __le__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a <= b)

    def __gt__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a > b)

    def __ge__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a >= b)

    def __add__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other: object) -> CompatExpression:
        return CompatExpression(_expression_operand(other) + self._value())

    def __sub__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other: object) -> CompatExpression:
        return CompatExpression(_expression_operand(other) - self._value())

    def __mul__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other: object) -> CompatExpression:
        return CompatExpression(_expression_operand(other) * self._value())

    def __truediv__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a / b)

    def __rtruediv__(self, other: object) -> CompatExpression:
        return CompatExpression(_expression_operand(other) / self._value())

    def __floordiv__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a // b)

    def __mod__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a % b)

    def __pow__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a**b)

    def __and__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a & b)

    def __or__(self, other: object) -> CompatExpression:
        return self._binary(other, lambda a, b: a | b)

    def __invert__(self) -> CompatExpression:
        return CompatExpression(~self._value())

    def __neg__(self) -> CompatExpression:
        return CompatExpression(-self._value())

    def alias(self, name: str) -> CompatExpression:
        """The expression under an output name."""
        return CompatExpression(self._value().alias(name))

    def cast(self, type: object) -> CompatExpression:
        """Cast to a SQL type, written as text."""
        return CompatExpression(self._value().cast(str(type)))

    def asc(self) -> CompatExpression:
        """Ascending sort order."""
        return CompatExpression(self._value().asc())

    def desc(self) -> CompatExpression:
        """Descending sort order."""
        return CompatExpression(self._value().desc())

    def nulls_first(self) -> CompatExpression:
        """NULLs before values in a sort."""
        return CompatExpression(self._value().nulls_first())

    def nulls_last(self) -> CompatExpression:
        """NULLs after values in a sort."""
        return CompatExpression(self._value().nulls_last())

    def between(self, lower: object, upper: object) -> CompatExpression:
        """Inclusive range."""
        return CompatExpression(self._value().between(_expression_operand(lower), _expression_operand(upper)))

    def isin(self, *values: object) -> CompatExpression:
        """Membership in the listed values."""
        return CompatExpression(self._value().isin([_expression_operand(v) for v in values]))

    def isnotin(self, *values: object) -> CompatExpression:
        """Absence from the listed values."""
        return ~self.isin(*values)

    def isnull(self) -> CompatExpression:
        """IS NULL, under its old name."""
        return CompatExpression(self._value().is_null())

    def isnotnull(self) -> CompatExpression:
        """IS NOT NULL, under its old name."""
        return CompatExpression(self._value().is_not_null())

    def collate(self, collation: str) -> CompatExpression:
        """The expression under a collation."""
        from .expr import sql_expr, suspended_sinks

        with suspended_sinks():
            rendered = self._value().fragment()
        return CompatExpression(sql_expr(f"({rendered} COLLATE {collation})"))

    def when(self, condition: object, value: object) -> CompatExpression:
        """Another CASE branch, the old two-argument form."""
        if self._case is None:
            message = "when() can only be chained on a CASE expression"
            raise InvalidInputException(message)
        return CompatExpression(case=self._case.when(_expression_operand(condition)).then(_expression_operand(value)))

    def otherwise(self, value: object) -> CompatExpression:
        """The CASE fallback, completing the expression."""
        if self._case is None:
            message = "otherwise() can only be chained on a CASE expression"
            raise InvalidInputException(message)
        return CompatExpression(self._case.otherwise(_expression_operand(value)))

    def get_name(self) -> str:
        """The rendered name, unquoted when it is one plain identifier."""
        rendered = repr(self)
        if rendered.startswith('"') and rendered.endswith('"') and '"' not in rendered[1:-1].replace('""', ""):
            return rendered[1:-1].replace('""', '"')
        return rendered

    def show(self) -> None:
        """Print the expression, as the old client's did."""
        print(repr(self))


def _expression_operand(value: object) -> Expr:
    from .expr import Expr, lit

    if isinstance(value, CompatExpression):
        return value._value()
    if isinstance(value, Expr):
        return value
    return lit(value)


def ColumnExpression(*parts: str) -> CompatExpression:
    """A column reference, dotted or given in parts."""
    from .expr import col

    return CompatExpression(col(".".join(parts)))


def ConstantExpression(value: object) -> CompatExpression:
    """A literal value."""
    from .expr import lit

    return CompatExpression(lit(value))


def FunctionExpression(function_name: str, *args: object) -> CompatExpression:
    """Any function by name."""
    from .expr import fn

    return CompatExpression(fn(function_name, *(_expression_operand(a) for a in args)))


def CaseExpression(condition: object, value: object) -> CompatExpression:
    """A CASE with its first branch; chain `.when(...)` and `.otherwise(...)`."""
    from .expr import when as native_when

    return CompatExpression(case=native_when(_expression_operand(condition)).then(_expression_operand(value)))


def StarExpression(*, exclude: Iterable[object] | None = None) -> CompatExpression:
    """`*`, optionally excluding columns by name or column expression."""
    from .expr import star

    names = [entry if isinstance(entry, str) else cast("CompatExpression", entry).get_name() for entry in exclude or []]
    return CompatExpression(star(exclude=names))


def SQLExpression(expression: str) -> CompatExpression:
    """A raw SQL fragment, spliced unchanged."""
    from .expr import sql_expr

    return CompatExpression(sql_expr(expression))


def DefaultExpression() -> CompatExpression:
    """The DEFAULT marker, for inserts."""
    from .expr import sql_expr

    return CompatExpression(sql_expr("DEFAULT"))


def CoalesceOperator(*args: object) -> CompatExpression:
    """The first argument that is not NULL."""
    from .expr import coalesce

    return CompatExpression(coalesce(*(_expression_operand(a) for a in args)))


#: The old class name, for isinstance checks in migrating code.
Expression = CompatExpression


class CompatRelation:
    """A lazy relation the old client's way: SQL text composed over one connection.

    The old verbs take SQL fragments as text, so composition is textual and
    never touches the native expression layer. Fetching follows the old
    model: `fetchone`/`fetchmany` run the relation once and drain the held
    result; a bare `fetchall` with nothing held runs fresh every call;
    `execute()` resets the held result.
    """

    def __init__(
        self,
        connection: CompatConnection,
        sql: str,
        alias: str | None = None,
        table: str | None = None,
        kind: str | None = None,
        name: str | None = None,
    ) -> None:
        #: Held strongly, so a relation keeps its cursor alive (old GH-315).
        self._connection = connection
        self._sql = sql
        self._relation_alias = alias
        #: The table this relation reads whole, when it does: insert() is a
        #: table-relation verb, as it was on the old client.
        self._table = table
        self._kind = kind
        #: A table or view relation joins under its own name, as it did on
        #: the old client, so conditions may qualify columns with it.
        self._name = name
        self._number = next(_relation_numbers)
        self._result: LiveResult | None = None
        self._closed = False
        #: Bound eagerly, as the old relations were: a bad fragment or a
        #: closed connection errors at the verb, not at the first fetch.
        self._schema = self._bind()

    def _bind(self) -> builtins.list[tuple[str, str]] | None:
        try:
            schema, _ = self._connection._engine().bind(self._sql)
        except InvalidInputError as error:
            # PIVOT and friends expand into multiple engine statements and
            # refuse to bind; they still execute, so the schema stays unknown
            # until a result exists.
            if "expands into multiple engine statements" in str(error):
                return None
            raise
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None
        return schema

    def __getattr__(self, name: str) -> NoReturn:
        # The old client reported a closed connection before an unknown verb;
        # reached only for names this face does not carry.
        if not name.startswith("_") and self._connection._raw is None:
            raise ConnectionException(_CLOSED_MESSAGE)
        message = f"'CompatRelation' object has no attribute {name!r}"
        raise AttributeError(message)

    def _source(self) -> str:
        aliased = f" AS {quote(self._relation_alias)}" if self._relation_alias else ""
        return f"({self._sql}){aliased}"

    def _named_source(self) -> str:
        return f"({self._sql}) AS {quote(self.alias)}"

    @property
    def alias(self) -> str:
        """The name this relation joins under, generated when none was set."""
        return self._relation_alias or self._name or f"unnamed_relation_{self._number}"

    def set_alias(self, alias: str) -> CompatRelation:
        """The same relation under a name later fragments can reference."""
        return CompatRelation(self._connection, self._sql, alias, table=self._table, kind=self._kind, name=self._name)

    def filter(self, condition: object) -> CompatRelation:
        """The rows where the condition holds, given as SQL text or an expression."""
        return CompatRelation(self._connection, f"SELECT * FROM {self._source()} WHERE {_operand(condition)}")

    def project(self, *columns: object, groups: str = "") -> CompatRelation:
        """The listed columns or expressions; `groups` was accepted and ignored there too."""
        rendered = ", ".join(_operand(column) for column in columns)
        return CompatRelation(self._connection, f"SELECT {rendered} FROM {self._source()}")

    def aggregate(self, aggregates: object, groups: str = "") -> CompatRelation:
        """Aggregate over the whole relation, or per group; expressions welcome."""
        if not isinstance(aggregates, str):
            aggregates = ", ".join(_operand(a) for a in cast("Iterable[object]", aggregates))
        grouped = f" GROUP BY {groups}" if groups else ""
        return CompatRelation(self._connection, f"SELECT {aggregates} FROM {self._source()}{grouped}")

    def query(self, virtual_table_name: str, sql: str) -> CompatRelation | None:
        """Run SQL that refers to this relation by the given name.

        The relation is registered as a temporary view under the name, as
        the old client registered it; a statement without rows runs on the
        spot and returns None, the old way.
        """
        with self._run(f"CREATE OR REPLACE TEMPORARY VIEW {quote(virtual_table_name)} AS {self._sql}") as result:
            result.drain()
        return self._connection.sql(sql)

    def _run(self, sql: str | None = None) -> LiveResult:
        try:
            return self._connection._execute(sql if sql is not None else self._sql)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    def _statement(self, sql: str, parameters: Sequence[Any] | None = None) -> None:
        try:
            self._connection.run(sql, parameters)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    def __repr__(self) -> str:
        """The first rows as a table, the way the old relations printed."""
        from .frame import sql as frame_sql

        try:
            return frame_sql(self._sql).preview(self._connection)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    def select(self, *args: object, groups: str = "") -> CompatRelation:
        """The old alias of `project`."""
        return self.project(*args)

    def order(self, order_expr: str) -> CompatRelation:
        """The rows sorted by the SQL fragment."""
        return CompatRelation(self._connection, f"SELECT * FROM {self._source()} ORDER BY {order_expr}")

    def sort(self, *args: object) -> CompatRelation:
        """The old alias family of `order`, taking keys as text or expressions."""
        from .expr import Expr, suspended_sinks

        def key(value: object) -> str:
            if isinstance(value, CompatExpression):
                value = value._value()
            if isinstance(value, Expr):
                with suspended_sinks():
                    return value.as_order()
            return str(value)

        return self.order(", ".join(key(argument) for argument in args))

    def limit(self, n: int, offset: int = 0) -> CompatRelation:
        """The first `n` rows, optionally past an offset."""
        tail = f" OFFSET {int(offset)}" if offset else ""
        return CompatRelation(self._connection, f"SELECT * FROM {self._source()} LIMIT {int(n)}{tail}")

    def distinct(self) -> CompatRelation:
        """The distinct rows."""
        return CompatRelation(self._connection, f"SELECT DISTINCT * FROM {self._source()}")

    def unique(self, unique_aggr: str) -> CompatRelation:
        """The distinct values of the listed columns."""
        return CompatRelation(self._connection, f"SELECT DISTINCT {unique_aggr} FROM {self._source()}")

    def _combine_guard(self, other: CompatRelation) -> None:
        # The old client refused to combine with a relation whose connection
        # is gone, in the closed-connection words.
        if other._connection._raw is None:
            raise ConnectionException(_CLOSED_MESSAGE)

    def write_csv(
        self,
        file_name: str,
        *,
        sep: str | None = None,
        na_rep: str | None = None,
        header: bool | None = None,
        quotechar: str | None = None,
        escapechar: str | None = None,
        date_format: str | None = None,
        timestamp_format: str | None = None,
        quoting: str | int | None = None,
        encoding: str | None = None,
        compression: str | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: builtins.list[str] | None = None,
        write_partition_columns: bool | None = None,
    ) -> None:
        """Write the relation as CSV: the old pandas-flavored names over the native COPY sink."""
        from .expr import star
        from .frame import sql as frame_sql

        options: dict[str, Any] = {
            name: value
            for name, value in (
                ("sep", sep),
                ("null", na_rep),
                ("header", header),
                ("quote", quotechar),
                ("escape", escapechar),
                ("dateformat", date_format),
                ("timestampformat", timestamp_format),
                ("encoding", encoding),
                ("compression", compression),
                ("overwrite", overwrite),
                ("per_thread_output", per_thread_output),
                ("use_tmp_file", use_tmp_file),
                ("partition_by", partition_by),
                ("write_partition_columns", write_partition_columns),
            )
            if value is not None
        }
        if quoting is not None:
            # The old client accepted csv.QUOTE_ALL (and its name) alone.
            if str(quoting).lower() not in ("1", "all", "force", "quote_all"):
                message = f"Unsupported value for 'quoting': {quoting!r}"
                raise InvalidInputException(message)
            options["force_quote"] = star()
        try:
            frame_sql(self._sql).to_csv(self._connection, file_name, **options)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    to_csv = write_csv

    def write_parquet(
        self,
        file_name: str,
        *,
        compression: str | None = None,
        field_ids: object = None,
        row_group_size_bytes: int | str | None = None,
        row_group_size: int | None = None,
        overwrite: bool | None = None,
        per_thread_output: bool | None = None,
        use_tmp_file: bool | None = None,
        partition_by: builtins.list[str] | None = None,
        write_partition_columns: bool | None = None,
        append: bool | None = None,
        filename_pattern: str | None = None,
        file_size_bytes: str | int | None = None,
    ) -> None:
        """Write the relation as Parquet, over the native COPY sink."""
        from .frame import sql as frame_sql

        options: dict[str, Any] = {
            name: value
            for name, value in (
                ("compression", compression),
                ("field_ids", field_ids),
                ("row_group_size_bytes", row_group_size_bytes),
                ("row_group_size", row_group_size),
                ("overwrite", overwrite),
                ("per_thread_output", per_thread_output),
                ("use_tmp_file", use_tmp_file),
                ("partition_by", partition_by),
                ("write_partition_columns", write_partition_columns),
                ("append", append),
                ("filename_pattern", filename_pattern),
                ("file_size_bytes", file_size_bytes),
            )
            if value is not None
        }
        try:
            frame_sql(self._sql).to_parquet(self._connection, file_name, **options)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    to_parquet = write_parquet

    def update(self, set: Mapping[str, object], *, condition: object = None) -> None:
        """UPDATE the table this relation reads; a table-relation verb, as it was."""
        from .expr import render_literal

        if self._table is None:
            message = "'DuckDBPyRelation.update' can only be used on a table relation"
            raise InvalidInputException(message)

        def rendered(value: object) -> str:
            from .expr import Expr

            if isinstance(value, (CompatExpression, Expr)):
                return _operand(value)
            return render_literal(value)

        assignments = ", ".join(f"{quote(name)} = {rendered(value)}" for name, value in set.items())
        where = f" WHERE {_operand(condition)}" if condition is not None else ""
        self._statement(f"UPDATE {qualified(self._table)} SET {assignments}{where}")

    def select_types(self, types: Iterable[object]) -> CompatRelation:
        """The columns whose type matches any of the given type texts."""
        wanted = {str(entry).strip().upper() for entry in types}
        keep = [name for name, kind in ((d[0], d[1]) for d in self.description) if str(kind).upper() in wanted]
        return self.project(", ".join(quote(name) for name in keep))

    select_dtypes = select_types

    def __contains__(self, name: str) -> bool:
        """Whether the relation has a column of this name."""
        return name in self.columns

    def fetch_df_chunk(self, vectors_per_chunk: int = 1, *, date_as_object: bool = False) -> pandas.DataFrame:
        """Up to this many engine chunks of the held result as a DataFrame; empty at the end."""
        import numpy as np

        from ._numpy import _columns_from_views, _frame_from_columns

        held = self._held()
        result = held.result
        names = [name for name, _ in result.schema]
        views = []
        try:
            for _ in range(int(vectors_per_chunk)):
                view = result.fetch_chunk_view()
                if view is None:
                    break
                views.append(view)
            meta = result.schema_types
        except InterfaceError:
            raise self._stale() from None
        return _frame_from_columns(_columns_from_views(np, names, views, meta), date_as_object=date_as_object)

    def join(self, other_rel: CompatRelation, condition: object, how: str = "inner") -> CompatRelation:
        """Join on a condition, or by the named columns when the condition is only names."""
        self._combine_guard(other_rel)
        condition = _operand(condition)
        kind = _JOIN_KINDS.get(how.strip().lower())
        if kind is None:
            message = f"Unsupported join type {how}"
            raise InvalidInputException(message)
        if self.alias.casefold() == other_rel.alias.casefold():
            message = (
                "Both relations have the same alias, please change the alias of one or both "
                "relations using 'rel = rel.set_alias(<new alias>)'"
            )
            raise InvalidInputException(message)
        clause = f"ON ({condition})"
        if _IDENTIFIER_LIST.fullmatch(condition):
            # Name-shaped conditions join USING, but only when they really
            # are column references: a literal like "true" binds on its own
            # and belongs in ON, the way the old parser decided.
            try:
                self._connection._engine().bind(f"SELECT {condition}")
            except ProgrammingError:
                clause = f"USING ({condition})"
            except Error:
                pass
        return CompatRelation(
            self._connection, f"SELECT * FROM {self._named_source()} {kind} {other_rel._named_source()} {clause}"
        )

    def cross(self, other_rel: CompatRelation) -> CompatRelation:
        """The cross product with another relation."""
        self._combine_guard(other_rel)
        return CompatRelation(
            self._connection, f"SELECT * FROM {self._named_source()} CROSS JOIN {other_rel._named_source()}"
        )

    def union(self, union_rel: CompatRelation) -> CompatRelation:
        """UNION ALL, as the old client's union was."""
        self._combine_guard(union_rel)
        return CompatRelation(self._connection, f"({self._sql}) UNION ALL ({union_rel._sql})")

    def except_(self, other_rel: CompatRelation) -> CompatRelation:
        """EXCEPT ALL, as the old client's except_ was."""
        self._combine_guard(other_rel)
        return CompatRelation(self._connection, f"({self._sql}) EXCEPT ALL ({other_rel._sql})")

    def intersect(self, other_rel: CompatRelation) -> CompatRelation:
        """INTERSECT ALL, as the old client's intersect was."""
        self._combine_guard(other_rel)
        return CompatRelation(self._connection, f"({self._sql}) INTERSECT ALL ({other_rel._sql})")

    # -- the old aggregate shorthands, all one template

    def _aggregate_operand(self, piece: str, function: str, parameter: str) -> str:
        # The old client parsed each operand and quoted it as an identifier
        # when the parse failed, which is how reserved words and spaced
        # names worked. The binder stands in for the parser here: a
        # parse-level failure routes to the quoted form, a binding failure
        # is left for the built relation to report.
        if piece == "*":
            return piece
        arguments = f"{piece},{parameter}" if parameter else piece
        try:
            self._connection._engine().bind(f"SELECT {function}({arguments}) FROM ({self._sql})")
        except (ParserError, InvalidInputError):
            return quote(piece)
        except Error:
            pass
        return piece

    def _aggregate_call(
        self, function: str, expression: str, groups: str, window_spec: str, projected_columns: str, parameter: str = ""
    ) -> CompatRelation:
        if groups and window_spec:
            message = "Either groups or window must be set (can't be both at the same time)"
            raise InvalidInputException(message)
        # The old client silently discarded a trailing "order by ..." in the
        # groups; rows keep their scan order, and the adopted tests rely on
        # exactly that.
        marker = groups.lower().find(" order by ")
        if marker != -1:
            groups = groups[:marker]
        inputs = [piece.strip() for piece in expression.split(",") if piece.strip()] if expression else []
        if not inputs and function.casefold() == "count":
            inputs = ["*"]
        tail = f" {window_spec}" if window_spec else ""
        calls = []
        for piece in inputs:
            operand = self._aggregate_operand(piece, function, parameter)
            arguments = f"{operand},{parameter}" if parameter else operand
            calls.append(f"{function}({arguments}){tail}")
        if not inputs and parameter:
            calls.append(f"{function}({parameter}){tail}")
        prefix = f"{projected_columns}, " if projected_columns else ""
        rendered = prefix + ", ".join(calls)
        if window_spec:
            return CompatRelation(self._connection, f"SELECT {rendered} FROM {self._source()}")
        return CompatRelation(self._connection, f"SELECT {rendered} FROM {self._source()} GROUP BY {groups or 'ALL'}")

    def apply(
        self,
        function_name: str,
        function_aggr: str,
        group_expr: str = "",
        function_parameter: str = "",
        projected_columns: str = "",
    ) -> CompatRelation:
        """Any function by name over the listed columns, the old passthrough."""
        return self._aggregate_call(function_name, function_aggr, group_expr, "", projected_columns, function_parameter)

    def any_value(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The engine's any_value aggregate over each listed column."""
        return self._aggregate_call("any_value", expression, groups, window_spec, projected_columns)

    def arg_max(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The value of the first column at the second's maximum."""
        return self._aggregate_call("arg_max", arg_column, groups, window_spec, projected_columns, value_column)

    def arg_min(
        self, arg_column: str, value_column: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The value of the first column at the second's minimum."""
        return self._aggregate_call("arg_min", arg_column, groups, window_spec, projected_columns, value_column)

    def avg(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The average over each listed column."""
        return self._aggregate_call("avg", expression, groups, window_spec, projected_columns)

    def bit_and(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The bitwise AND over each listed column."""
        return self._aggregate_call("bit_and", expression, groups, window_spec, projected_columns)

    def bit_or(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The bitwise OR over each listed column."""
        return self._aggregate_call("bit_or", expression, groups, window_spec, projected_columns)

    def bit_xor(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The bitwise XOR over each listed column."""
        return self._aggregate_call("bit_xor", expression, groups, window_spec, projected_columns)

    def bitstring_agg(
        self,
        expression: str,
        min: int | None = None,
        max: int | None = None,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> CompatRelation:
        """The bitstring aggregate over each listed column, with the old min/max checks."""
        if (min is None) != (max is None):
            message = "Both min and max values must be set"
            raise InvalidInputException(message)
        if min is not None and (not isinstance(min, int) or not isinstance(max, int)):
            message = "min and max must be of type int"
            raise InvalidTypeException(message)
        parameter = "" if min is None or max is None else f"{int(min)},{int(max)}"
        return self._aggregate_call("bitstring_agg", expression, groups, window_spec, projected_columns, parameter)

    def bool_and(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The boolean AND over each listed column."""
        return self._aggregate_call("bool_and", expression, groups, window_spec, projected_columns)

    def bool_or(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The boolean OR over each listed column."""
        return self._aggregate_call("bool_or", expression, groups, window_spec, projected_columns)

    def count(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The count over each listed column; a star count with none."""
        return self._aggregate_call("count", expression, groups, window_spec, projected_columns)

    def favg(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The Kahan-summed average over each listed column."""
        return self._aggregate_call("favg", expression, groups, window_spec, projected_columns)

    def first(self, expression: str, groups: str = "", projected_columns: str = "") -> CompatRelation:
        """The first value over each listed column."""
        return self._aggregate_call('"first"', expression, groups, "", projected_columns)

    def fsum(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The Kahan sum over each listed column."""
        return self._aggregate_call("fsum", expression, groups, window_spec, projected_columns)

    def geomean(self, expression: str, groups: str = "", projected_columns: str = "") -> CompatRelation:
        """The geometric mean over each listed column."""
        return self._aggregate_call("geomean", expression, groups, "", projected_columns)

    def histogram(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The histogram over each listed column."""
        return self._aggregate_call("histogram", expression, groups, window_spec, projected_columns)

    def last(self, expression: str, groups: str = "", projected_columns: str = "") -> CompatRelation:
        """The last value over each listed column."""
        return self._aggregate_call('"last"', expression, groups, "", projected_columns)

    def list(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The values gathered into a list, over each listed column."""
        return self._aggregate_call("list", expression, groups, window_spec, projected_columns)

    def max(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The maximum over each listed column."""
        return self._aggregate_call("max", expression, groups, window_spec, projected_columns)

    def mean(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The average, under its old alias."""
        return self._aggregate_call("avg", expression, groups, window_spec, projected_columns)

    def median(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The median over each listed column."""
        return self._aggregate_call("median", expression, groups, window_spec, projected_columns)

    def min(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The minimum over each listed column."""
        return self._aggregate_call("min", expression, groups, window_spec, projected_columns)

    def mode(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The mode over each listed column."""
        return self._aggregate_call("mode", expression, groups, window_spec, projected_columns)

    def product(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The product over each listed column."""
        return self._aggregate_call("product", expression, groups, window_spec, projected_columns)

    def quantile(
        self,
        expression: str,
        q: object = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> CompatRelation:
        """The discrete quantile, under its old shorthand name."""
        return self._aggregate_call(
            "quantile", expression, groups, window_spec, projected_columns, _quantile_parameter(q)
        )

    def quantile_cont(
        self,
        expression: str,
        q: object = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> CompatRelation:
        """The interpolated quantile."""
        return self._aggregate_call(
            "quantile_cont", expression, groups, window_spec, projected_columns, _quantile_parameter(q)
        )

    def quantile_disc(
        self,
        expression: str,
        q: object = 0.5,
        groups: str = "",
        window_spec: str = "",
        projected_columns: str = "",
    ) -> CompatRelation:
        """The discrete quantile."""
        return self._aggregate_call(
            "quantile_disc", expression, groups, window_spec, projected_columns, _quantile_parameter(q)
        )

    def std(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The sample standard deviation, under its old shorthand name."""
        return self._aggregate_call("stddev_samp", expression, groups, window_spec, projected_columns)

    stddev = std
    stddev_samp = std

    def stddev_pop(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The population standard deviation over each listed column."""
        return self._aggregate_call("stddev_pop", expression, groups, window_spec, projected_columns)

    def string_agg(
        self, expression: str, sep: str = ",", groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The strings joined, the separator carried as a quoted literal."""
        quoted = "'" + sep.replace("'", "''") + "'"
        return self._aggregate_call("string_agg", expression, groups, window_spec, projected_columns, quoted)

    def sum(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The sum over each listed column."""
        return self._aggregate_call("sum", expression, groups, window_spec, projected_columns)

    def value_counts(self, expression: str, groups: str = "") -> CompatRelation:
        """The values with their counts, projecting the column beside its count."""
        return self._aggregate_call("count", expression, groups, "", expression)

    def var(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The sample variance, under its old shorthand name."""
        return self._aggregate_call("var_samp", expression, groups, window_spec, projected_columns)

    var_samp = var
    variance = var

    def var_pop(
        self, expression: str, groups: str = "", window_spec: str = "", projected_columns: str = ""
    ) -> CompatRelation:
        """The population variance over each listed column."""
        return self._aggregate_call("var_pop", expression, groups, window_spec, projected_columns)

    # -- the old window shorthands

    def _window_call(self, call: str, window_spec: str, projected_columns: str) -> CompatRelation:
        prefix = f"{projected_columns}, " if projected_columns else ""
        return CompatRelation(self._connection, f"SELECT {prefix}{call} {window_spec} FROM {self._source()}")

    def row_number(self, window_spec: str, projected_columns: str = "") -> CompatRelation:
        """The row number over the window."""
        return self._window_call("row_number()", window_spec, projected_columns)

    def rank(self, window_spec: str, projected_columns: str = "") -> CompatRelation:
        """The rank over the window."""
        return self._window_call("rank()", window_spec, projected_columns)

    def dense_rank(self, window_spec: str, projected_columns: str = "") -> CompatRelation:
        """The dense rank over the window."""
        return self._window_call("dense_rank()", window_spec, projected_columns)

    rank_dense = dense_rank

    def percent_rank(self, window_spec: str, projected_columns: str = "") -> CompatRelation:
        """The percent rank over the window."""
        return self._window_call("percent_rank()", window_spec, projected_columns)

    def cume_dist(self, window_spec: str, projected_columns: str = "") -> CompatRelation:
        """The cumulative distribution over the window."""
        return self._window_call("cume_dist()", window_spec, projected_columns)

    def n_tile(self, window_spec: str, num_buckets: int, projected_columns: str = "") -> CompatRelation:
        """The bucket number over the window."""
        return self._window_call(f"ntile({int(num_buckets)})", window_spec, projected_columns)

    def lag(
        self,
        expression: str,
        window_spec: str,
        offset: int = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> CompatRelation:
        """The lagging value over the window, with the old offset and default arguments."""
        suffix = " ignore nulls" if ignore_nulls else ""
        return self._window_call(
            f"lag({expression}, {int(offset)}, {default_value}{suffix})", window_spec, projected_columns
        )

    def lead(
        self,
        expression: str,
        window_spec: str,
        offset: int = 1,
        default_value: str = "NULL",
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> CompatRelation:
        """The leading value over the window, with the old offset and default arguments."""
        suffix = " ignore nulls" if ignore_nulls else ""
        return self._window_call(
            f"lead({expression}, {int(offset)}, {default_value}{suffix})", window_spec, projected_columns
        )

    def first_value(self, expression: str, window_spec: str = "", projected_columns: str = "") -> CompatRelation:
        """The first value over the window."""
        return self._window_call(f"first_value({expression})", window_spec, projected_columns)

    def last_value(self, expression: str, window_spec: str = "", projected_columns: str = "") -> CompatRelation:
        """The last value over the window."""
        return self._window_call(f"last_value({expression})", window_spec, projected_columns)

    def nth_value(
        self,
        expression: str,
        window_spec: str,
        offset: int,
        ignore_nulls: bool = False,
        projected_columns: str = "",
    ) -> CompatRelation:
        """The nth value over the window."""
        suffix = " ignore nulls" if ignore_nulls else ""
        return self._window_call(f"nth_value({expression}, {int(offset)}{suffix})", window_spec, projected_columns)

    # -- materialization and introspection

    def create(self, table_name: str) -> None:
        """A new table holding this relation's rows, through the native verb."""
        from .frame import sql as frame_sql

        try:
            frame_sql(self._sql).create(self._connection, table_name)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    to_table = create

    def create_view(self, view_name: str, replace: bool = True) -> CompatRelation:
        """A view over this relation's SQL, effective immediately."""
        prefix = "CREATE OR REPLACE VIEW" if replace else "CREATE VIEW"
        self._statement(f"{prefix} {qualified(view_name)} AS {self._sql}")
        return self

    to_view = create_view

    def insert_into(self, table_name: str) -> None:
        """Append this relation's rows to a table by position, through the native verb."""
        from .frame import sql as frame_sql

        try:
            frame_sql(self._sql).insert_into(self._connection, table_name)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    def insert(self, values: object) -> None:
        """Append one row of values; a table-relation verb, as it was."""
        if self._table is None:
            message = "'DuckDBPyRelation.insert' can only be used on a table relation"
            raise InvalidInputException(message)
        from .frame import values as frame_values

        # The connection speaks before the row is inspected, as it did on
        # the old client: the closed words beat an arity complaint.
        if self._connection._raw is None:
            raise ConnectionException(_CLOSED_MESSAGE)
        row = tuple(cast("Iterable[Any]", values))
        try:
            frame_values([row], self.columns).insert_into(self._connection, self._table)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    def describe(self) -> CompatRelation:
        """The old summary: count/mean/stddev/min/max/median per column, one row each."""
        aggregates = ("count", "mean", "stddev", "min", "max", "median")
        numeric_only = {"mean", "stddev", "median"}
        inner = []
        outer = ["unnest(['count', 'mean', 'stddev', 'min', 'max', 'median']) AS aggr"]
        for index, entry in enumerate(self.description):
            name, type_text = entry[0], entry[1]
            base = str(type_text).split("(")[0].strip().upper()
            numeric = base in _NUMERIC_TYPES
            cast_to = "DOUBLE" if numeric else "VARCHAR"
            cells = []
            for position, aggregate in enumerate(aggregates):
                if not numeric and aggregate in numeric_only:
                    cells.append("NULL")
                    continue
                cell = f"c{index}_{position}"
                inner.append(f"CAST({aggregate}({quote(name)}) AS {cast_to}) AS {cell}")
                cells.append(cell)
            outer.append(f"unnest([{', '.join(cells)}]) AS {quote(name)}")
        # A relation with no columns still describes: one aggr column of names.
        source = f"(SELECT {', '.join(inner)} FROM {self._source()})" if inner else "(SELECT 1)"
        return CompatRelation(self._connection, f"SELECT {', '.join(outer)} FROM {source}")

    def explain(self, type: str = "standard") -> str:
        """The engine's plan for this relation, as text; "analyze" runs it."""
        keyword = "EXPLAIN ANALYZE" if "analyze" in str(type).lower() else "EXPLAIN"
        with self._run(f"{keyword} {self._sql}") as result:
            rows = result.fetch_all()
        return "\n".join(str(value) for _, value in rows)

    def show(self, **_: object) -> None:
        """Print the first rows as a table."""
        from .frame import sql as frame_sql

        frame_sql(self._sql).show(self._connection)

    def sql_query(self) -> str:
        """The SQL text this relation stands for."""
        return self._sql

    @property
    def type(self) -> str:
        """The old relation kind: TABLE, VIEW, MATERIALIZED or QUERY."""
        if self._table:
            return "TABLE_RELATION"
        return self._kind or "QUERY_RELATION"

    @property
    def columns(self) -> builtins.list[str]:
        """The column names, from binding the text."""
        return [entry[0] for entry in self.description]

    @property
    def types(self) -> builtins.list[str]:
        """The column types, as text; the old client returned type objects."""
        return [entry[1] for entry in self.description]

    dtypes = types

    @property
    def shape(self) -> tuple[int, int]:
        """(row count, column count); counting runs the relation."""
        return (len(self), len(self.columns))

    def __len__(self) -> int:
        """The row count, through the native count terminal."""
        from .frame import sql as frame_sql

        try:
            return frame_sql(self._sql).count(self._connection)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None

    def __getitem__(self, name: str) -> CompatRelation:
        """A single-column projection, as the old subscript was."""
        return self.project(name)

    def execute(self) -> CompatRelation:
        """Run the relation and hold its rows, resetting whatever was held."""
        self._close_result()
        self._closed = False
        self._result = self._run()
        return self

    def close(self) -> None:
        """Release the held result; fetches afterwards say `result closed`."""
        self._close_result()
        self._closed = True

    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, running the relation first if nothing is held."""
        held = self._held()
        try:
            rows = held.fetch_rows(1)
        except InterfaceError:
            raise self._stale() from None
        return rows[0] if rows else None

    def fetchmany(self, size: int = 1) -> builtins.list[tuple[Any, ...]]:
        """Up to `size` rows, running the relation first if nothing is held."""
        if size <= 0:
            return []
        held = self._held()
        try:
            return held.fetch_rows(size)
        except InterfaceError:
            raise self._stale() from None

    def fetchall(self) -> builtins.list[tuple[Any, ...]]:
        """Every row: the rest of a held result, or a fresh run each call."""
        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is not None:
            try:
                rows = self._result.fetch_all()
            except InterfaceError:
                raise self._stale() from None
            self._close_result()
            return rows
        with self._run() as result:
            return result.fetch_all()

    def fetchnumpy(self) -> dict[str, Any]:
        """The relation as numpy arrays: the rest of a held result, or a fresh run."""
        from ._numpy import fetch_numpy

        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is not None:
            result = self._result
            self._result = None
            try:
                return fetch_numpy(result.result)
            except InterfaceError:
                raise self._stale() from None
            finally:
                # A conversion that failed midway must not orphan the open
                # result: one live result would block the next statement.
                result.close()
        with self._run() as result:
            return fetch_numpy(result.result)

    def fetchdf(self, date_as_object: bool = False) -> pandas.DataFrame:
        """The relation as a pandas DataFrame: the rest of a held result, or a fresh run."""
        from ._numpy import to_dataframe

        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is not None:
            result = self._result
            self._result = None
            try:
                return to_dataframe(result.result, date_as_object=date_as_object)
            except InterfaceError:
                raise self._stale() from None
            finally:
                result.close()
        with self._run() as result:
            return to_dataframe(result.result, date_as_object=date_as_object)

    #: The old client's aliases for `fetchdf`.
    df = fetchdf
    to_df = fetchdf

    @property
    def description(self) -> builtins.list[tuple[Any, ...]]:
        """Column metadata, from the held result or from binding the text."""
        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is not None:
            try:
                schema = self._result.result.schema
            except InterfaceError:
                raise self._stale() from None
        elif self._schema is not None:
            schema = self._schema
        else:
            with self._run() as result:
                result.fetch_rows(1)
                schema = result.result.schema
            self._schema = schema
        return [(column, type_text, None, None, None, None, None) for column, type_text in schema]

    def _close_result(self) -> None:
        if self._result is not None:
            self._result.close()
            self._result = None

    def _stale(self) -> InvalidInputError:
        # The held result was force-closed under us, by the connection's
        # close cascade; adopt the closed state so every fetch reports it
        # with the one exception this class promises.
        self._result = None
        self._closed = True
        message = "result closed"
        return InvalidInputException(message)

    def _held(self) -> LiveResult:
        if self._closed:
            message = "result closed"
            raise InvalidInputException(message)
        if self._result is None:
            self.execute()
        return cast("LiveResult", self._result)


class CompatConnection(Connection):
    """A connection speaking the old client's execute-and-fetch vocabulary."""

    def __init__(self, database: _duckdb.Database, catalog: _Catalog | None = None) -> None:
        super().__init__(database, catalog)
        self._held: LiveResult | None = None
        #: Whether anything was fetched from the held result yet: the old
        #: client silently discarded an untouched result when the next
        #: statement arrived, and kept a touched one, fully materialized.
        self._held_touched = False
        self._compat_description: list[tuple[Any, ...]] | None = None
        #: Cursors made from this connection; the old client closed them with it.
        self._cursors: weakref.WeakSet[CompatConnection] = weakref.WeakSet()
        #: The shared-instance holder, kept so the cache entry lives while we do.
        self._instance: _Instance | None = None

    def execute(self, sql: str, parameters: Sequence[Any] | Mapping[Any, Any] | None = None) -> CompatConnection:
        """Run a statement, holding its rows for the fetch family. Returns self.

        A statement that produces no rows is applied on the spot: the old
        client's INSERT took effect at execute, and this face keeps that.
        """
        self._release_held()
        if isinstance(parameters, Mapping):
            # The old client took any parameter-dict key, so {1: v} fills $1;
            # the native seam refuses non-strings, and the leniency lives here.
            parameters = {_name(key): value for key, value in parameters.items()}
        result = self._execute(sql, parameters)
        if result.result.result_type == "rows":
            self._held = result
            self._held_touched = False
            self._compat_description = [
                (name, type_text, None, None, None, None, None) for name, type_text in result.result.schema
            ]
        else:
            with result:
                result.drain()
        return self

    def executemany(self, sql: str, seq_of_parameters: Iterable[Sequence[Any] | Mapping[Any, Any]]) -> CompatConnection:
        """Run one statement once per parameter set. Returns self."""
        sets = builtins.list(seq_of_parameters)
        if not sets:
            message = "executemany requires a non-empty list of parameter sets to be provided"
            raise InvalidInputException(message)
        self._release_held()
        for parameters in sets:
            self.execute(sql, parameters)
        return self

    def fetchone(self) -> tuple[Any, ...] | None:
        """The next row, or None when the held result is exhausted."""
        rows = self._require_held().fetch_rows(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        """Up to `size` rows; the old client's default is one."""
        if size <= 0:
            return []
        return self._require_held().fetch_rows(size)

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Every remaining row of the held result."""
        return self._require_held().fetch_all()

    def fetchnumpy(self) -> dict[str, Any]:
        """The held result as numpy arrays, consuming it."""
        from ._numpy import fetch_numpy

        result = self._require_held()
        try:
            return fetch_numpy(result.result)
        finally:
            self._release_held()

    def fetchdf(self, date_as_object: bool = False) -> pandas.DataFrame | None:
        """The held result as a pandas DataFrame, or None when nothing is held.

        None rather than an error on the empty case, because the old
        client's `df()` answered that way after the result was consumed.
        """
        if self._held is None:
            return None
        from ._numpy import to_dataframe

        try:
            return to_dataframe(self._held.result, date_as_object=date_as_object)
        finally:
            self._release_held()

    def fetch_df_chunk(self, vectors_per_chunk: int = 1, *, date_as_object: bool = False) -> pandas.DataFrame:
        """Up to this many engine chunks of the held result as a DataFrame; empty at the end."""
        import numpy as np

        from ._numpy import _columns_from_views, _frame_from_columns

        result = self._require_held().result
        names = [name for name, _ in result.schema]
        views = []
        for _ in range(int(vectors_per_chunk)):
            view = result.fetch_chunk_view()
            if view is None:
                break
            views.append(view)
        return _frame_from_columns(
            _columns_from_views(np, names, views, result.schema_types), date_as_object=date_as_object
        )

    #: The old client's aliases for `fetchdf`.
    df = fetchdf
    fetch_df = fetchdf

    @property
    def description(self) -> list[tuple[Any, ...]] | None:
        """Column metadata of the held result, or None when there is none."""
        return self._compat_description

    @property
    def rowcount(self) -> int:
        """Always -1, as the old client reported it. Real counts live in `run()` and dbapi."""
        return -1

    def sql(self, query: str) -> CompatRelation | None:
        """A lazy relation for a query; anything else runs on the spot, the old way.

        The statement classifies itself: it is executed (which prepares but
        runs nothing), asked what it is, and the classification result is
        closed before this returns, so no result is ever left open on the
        connection. A parser-certified query comes back lazy; a statement
        producing rows any other way (RETURNING above all) runs exactly
        once, its rows carried in a materialized relation; the rest is
        drained on the spot and returns None.
        """
        cleaned = query.strip().rstrip(";")
        if self._held is not None and not self._held_touched:
            # The old client silently discarded a held result nothing had
            # fetched from when the next statement arrived; a touched one it
            # kept, materialized. This client streams, so the touched case
            # raises the engine's own live-result refusal instead.
            self._release_held()
        try:
            probe = self._execute(cleaned)
        except InterfaceError:
            raise ConnectionException(_CLOSED_MESSAGE) from None
        materialized: tuple[Any, Any] | None = None
        with probe:
            try:
                statement = probe.result.statement_type
            except InvalidInputError:
                # Only the multi-expanding statements (the PIVOT family)
                # cannot answer before stepping, and those are queries.
                statement = "select"
            if statement not in _LAZY_STATEMENTS:
                if probe.result.result_type == "rows":
                    materialized = (probe.result.schema, probe.fetch_all())
                else:
                    probe.drain()
        if materialized is not None:
            return _materialized_relation(self, materialized[0], materialized[1])
        if statement in _LAZY_STATEMENTS:
            return CompatRelation(self, cleaned)
        return None

    def query(self, query: str) -> CompatRelation | None:
        """The old alias of `sql`."""
        return self.sql(query)

    def table(self, name: str) -> CompatRelation:
        """A relation reading a table by name."""
        return CompatRelation(self, f"SELECT * FROM {qualified(name)}", table=name, name=name)

    def view(self, name: str) -> CompatRelation:
        """A relation reading a view by name."""
        return CompatRelation(self, f"SELECT * FROM {qualified(name)}", kind="VIEW_RELATION", name=name)

    def table_function(self, name: str, params: object = None) -> CompatRelation:
        """A relation over a table function, through the native source."""
        from .expr import suspended_sinks
        from .frame import table_function as frame_table_function

        if params is None:
            params = []
        if not isinstance(params, builtins.list):
            message = "'params' has to be a list of parameters"
            raise InvalidInputException(message)
        with suspended_sinks():
            sql = frame_table_function(name, *params).render()
        return CompatRelation(self, sql)

    def begin(self) -> CompatConnection:
        """BEGIN TRANSACTION, as a method because the old connection had one."""
        self.run("BEGIN TRANSACTION")
        return self

    def commit(self) -> CompatConnection:
        """COMMIT the open transaction."""
        self.run("COMMIT")
        return self

    def rollback(self) -> CompatConnection:
        """ROLLBACK the open transaction."""
        self.run("ROLLBACK")
        return self

    def cursor(self) -> CompatConnection:
        """An independent duplicate with its own transaction and temp schema.

        The old client's `cursor()` behaved this way, and this face keeps it:
        the PEP 249 cursor that shares its parent's transaction lives in
        `duckdb.dbapi`, not here. Closing this connection closes the cursor,
        as the old client did.
        """
        child = cast("CompatConnection", self.duplicate())
        # The child shares the instance holder: any live cursor keeps the
        # path's shared-database cache entry alive, as the old client did.
        child._instance = self._instance
        self._cursors.add(child)
        return child

    def close(self) -> None:
        """Close this connection and every cursor made from it.

        Every step runs whatever any one of them raises: close must always
        mean closed, never a connection left usable because one cursor's
        close failed first.
        """
        failures: list[BaseException] = []
        for child in list(self._cursors):
            try:
                child.close()
            except Exception as error:  # every cursor must be tried
                failures.append(error)
        try:
            self._release_held()
        except Exception as error:
            failures.append(error)
        self._instance = None
        try:
            super().close()
        except Exception as error:
            failures.append(error)
        if failures:
            raise failures[0]

    def _release_held(self) -> None:
        if self._held is not None:
            self._held.close()
            self._held = None
        self._compat_description = None

    def _require_held(self) -> LiveResult:
        # The held check speaks first, so fetching after close() reports the
        # old client's "No open result set" rather than a closed connection.
        if self._held is None:
            message = "No open result set"
            raise InvalidInputException(message)
        self._held_touched = True
        return self._held


class _Instance:
    """One shared database and its catalog, held so the cache can reference it weakly.

    The engine's Database object cannot take weak references, so this plain
    holder stands in: every connection to the path keeps the holder alive,
    and when the last one closes, the holder dies and the cache entry with
    it, releasing the file.
    """

    __slots__ = ("__weakref__", "catalog", "database")

    def __init__(self, database: _duckdb.Database, catalog: _Catalog) -> None:
        self.database = database
        self.catalog = catalog


#: One shared database instance per path, as the old client's connect() gave
#: out. Weak, so the instance dies with its last connection and a later
#: connect() opens the file afresh.
_instances: dict[str, tuple[weakref.ref[_Instance], frozenset[tuple[str, str]]]] = {}
_instances_lock = threading.Lock()


def connect(
    database: str = ":memory:", read_only: bool = False, config: Mapping[str, object] | None = None
) -> CompatConnection:
    """Open a connection the old client's way: positional path, read_only flag, config dict.

    Connections to the same path share one database instance, so two
    writers see each other and `:memory:name` is shared; asking for the
    same path with a different configuration is refused in the old
    client's words. Plain in-memory databases are never shared.
    """
    if database == ":default:":
        return default_connection()
    options: dict[str, object] = dict(config or {})
    if read_only:
        options["access_mode"] = "read_only"
    options.setdefault("duckdb_api", f"python/{__formatted_python_version__}")
    rendered = [(key, str(value)) for key, value in options.items()]
    if database in ("", ":memory:"):
        return CompatConnection(_duckdb.Database(database, rendered))
    key = database if database.startswith(":memory:") else str(Path(database).resolve())
    fingerprint = frozenset(rendered)
    with _instances_lock:
        for stale in [path for path, (reference, _) in _instances.items() if reference() is None]:
            del _instances[stale]
        entry = _instances.get(key)
        shared = entry[0]() if entry is not None else None
        if shared is not None and entry is not None:
            if fingerprint != entry[1]:
                message = (
                    "Can't open a connection to same database file with a "
                    "different configuration than existing connections"
                )
                raise ConnectionException(message)
        else:
            if entry is not None:
                del _instances[key]
            shared = _Instance(_duckdb.Database(database, rendered), _Catalog())
            _instances[key] = (weakref.ref(shared), fingerprint)
        connection = CompatConnection(shared.database, shared.catalog)
        connection._instance = shared
        return connection


_default: CompatConnection | None = None
_default_lock = threading.Lock()


def default_connection() -> CompatConnection:
    """The module-level connection behind `execute()` and friends, made lazily."""
    global _default
    with _default_lock:
        if _default is None or _default._database is None:
            _default = connect(":memory:")
        return _default


def set_default_connection(connection: CompatConnection) -> None:
    """Route the module-level surface at this connection, as the old client allowed."""
    if not isinstance(cast("object", connection), CompatConnection):
        # The old client's nanobind boundary rejected anything else with
        # this wording, and the adopted tests match on it.
        message = f"set_default_connection(): incompatible function arguments. Invoked with: {connection!r}"
        raise TypeError(message)
    global _default
    with _default_lock:
        _default = connection


def execute(sql: str, parameters: Sequence[Any] | Mapping[Any, Any] | None = None) -> CompatConnection:
    """`execute` on the default connection, as the old module surface had."""
    return default_connection().execute(sql, parameters)


def sql(query: str) -> CompatRelation | None:
    """A relation over the default connection; a statement without rows runs on the spot."""
    return default_connection().sql(query)


def query(query: str) -> CompatRelation | None:
    """The old alias of module-level `sql`."""
    return default_connection().sql(query)


def from_query(query: str) -> CompatRelation | None:
    """The old alias of module-level `sql`."""
    return default_connection().sql(query)


def table(name: str) -> CompatRelation:
    """A relation reading a table on the default connection."""
    return default_connection().table(name)


def table_function(name: str, params: object = None) -> CompatRelation:
    """A relation over a table function on the default connection."""
    return default_connection().table_function(name, params)


def close() -> None:
    """Close the default connection; the next module-level call makes a fresh one."""
    global _default
    with _default_lock:
        if _default is not None:
            _default.close()
            _default = None


def description() -> list[tuple[Any, ...]] | None:
    """The default connection's `description`."""
    return default_connection().description
