"""The one place where the relation layer and the expression builder ask the engine something.

`bind` asks for the schema of one statement without running it, `submit` runs one statement with its parameter values
and returns a result in the format a terminal needs. The registry functions put a Python object or a retained result
under a name that statements on the same connection can read, for as long as a handle to it lives. The literal and
parser functions say how a Python value binds and reads as SQL, and how SQL expression text parses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import _duckdb

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from duckdb.sqltypes import DuckDBPyType

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "Result",
    "Signature",
    "SourceHandle",
    "arrow_schema",
    "bind",
    "catalog_version",
    "create_view",
    "literal_type",
    "name_scope",
    "parse_expressions",
    "parse_type",
    "prepare_script",
    "register_collection",
    "register_filelike",
    "register_map",
    "register_source",
    "submit",
    "table_kind",
    "temporal_literal",
    "value_literal",
]

DEFAULT_BATCH_SIZE = 131072

Values = list[Any] | dict[str, Any] | None

Result = _duckdb.RelationResult
SourceHandle = _duckdb.SourceHandle


class Signature(NamedTuple):
    """What the binder says about one statement."""

    names: list[str]
    types: list[DuckDBPyType]
    parameters: list[tuple[str, DuckDBPyType]]
    return_type: str
    deferred: bool
    captured: Sequence[tuple[str, SourceHandle]] = ()


def bind(con: _duckdb.DuckDBPyConnection, sql: str, capture: bool = False) -> Signature:  # noqa: FBT001
    """The output names and types of one statement, its parameters and what it returns, without running it.

    With `capture`, the Python variables the statement reads are registered and listed under the names it used.
    """
    if capture:
        names, types, parameters, return_type, deferred, captured = con._bind(sql, capture=True)
        return Signature(names, types, parameters, return_type, deferred, captured)
    names, types, parameters, return_type, deferred = con._bind(sql)
    return Signature(names, types, parameters, return_type, deferred)


def submit(
    con: _duckdb.DuckDBPyConnection,
    sql: str,
    values: Values,
    format: str = "rows",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Result:
    """Run one statement with its values, to the format a terminal needs."""
    return con._submit(sql, values, format, batch_size)  # type: ignore[arg-type]


def prepare_script(
    con: _duckdb.DuckDBPyConnection, query: object, values: Values
) -> tuple[Result | None, str | None, str | None]:
    """Run every statement of a script but the last; the rows an earlier one kept, the last one's text and type."""
    if isinstance(query, _duckdb.Statement):
        return None, query.query, query.type.name
    if not isinstance(query, str):
        msg = "Invalid Input Error: Please provide either a DuckDBPyStatement or a string representing the query"
        raise _duckdb.InvalidInputException(msg)
    kept, last_text, last_type = con._prepare_script(query, values)
    return kept, last_text, last_type.upper() if last_type is not None else None


def arrow_schema(con: _duckdb.DuckDBPyConnection, names: list[str], types: Sequence[DuckDBPyType | str]) -> Any:  # noqa: ANN401
    """The Arrow schema of columns with these names and types, without running a statement."""
    return _duckdb._arrow_schema(names, list(types), connection=con)


def parse_type(con: _duckdb.DuckDBPyConnection, text: str) -> DuckDBPyType:
    """A type name parsed on this connection, so its own types resolve."""
    return con.sqltype(text)


def catalog_version(con: _duckdb.DuckDBPyConnection) -> int:
    """How many statements that may change what a name binds to have run on this connection."""
    return con._catalog_version()


def table_kind(con: _duckdb.DuckDBPyConnection, name: str | tuple[str, ...]) -> str:
    """Whether a name reads a "table", a "view" or "none" of the two, without running a statement.

    A string is one identifier however many dots it holds, and a tuple is the parts of a qualified name, so the two
    readings of a dotted name are asked separately.
    """
    return con._table_kind(name)


def name_scope(con: _duckdb.DuckDBPyConnection, name: str) -> str:
    """Whether one identifier names a "database", a "schema" of the search path, or "none" of the two."""
    return con._name_scope(name)


def register_source(con: _duckdb.DuckDBPyConnection, obj: object, name: str | None = None) -> SourceHandle:
    """Put a Python object under a name statements on this connection can read."""
    return con._register_source(obj, name)


def register_collection(con: _duckdb.DuckDBPyConnection, result: Result) -> SourceHandle:
    """Keep the rows of a retained result under a name statements on this connection can read."""
    return con._register_collection(result)


def register_map(
    con: _duckdb.DuckDBPyConnection,
    function: Callable[..., Any],
    schema: object,
    input_sql: str,
) -> SourceHandle:
    """Register the map of a Python function over the rows of a query as a source."""
    return con._register_map(function, schema, input_sql)  # type: ignore[arg-type]


def register_filelike(con: _duckdb.DuckDBPyConnection, obj: object) -> SourceHandle:
    """Put a file-like object on the connection's object store; the handle's name is its path there."""
    return con._register_filelike(obj)


def create_view(con: _duckdb.DuckDBPyConnection, name: str, sql: str, *, replace: bool) -> None:
    """Create the persistent view `name` over the SELECT in `sql`, the view owning what the registry names it reads."""
    con._create_view(name, sql, replace=replace)


def literal_type(value: object) -> str:
    """The SQL type `value` binds as when passed as a parameter; raises the conversion error where it cannot bind."""
    return _duckdb._literal_type(value)


def temporal_literal(value: object) -> str:
    """`value` written as a SQL literal that means what it means bound as a parameter."""
    return _duckdb._temporal_literal(value)


def value_literal(value: object) -> str:
    """A value written as the SQL literal of its object converted to its type, as a parameter is."""
    return _duckdb._value_literal(value)


def parse_expressions(text: str) -> list[str]:
    """The expressions of a comma separated list, each rendered back by the engine's parser with its alias kept."""
    return _duckdb._parse_expressions(text)
