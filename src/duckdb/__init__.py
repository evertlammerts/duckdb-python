"""The DuckDB Python package."""

from importlib.metadata import version as _package_version

from . import dbapi, exceptions
from ._duckdb import library_version
from .connection import Connection, connect
from .expr import (
    Expr,
    coalesce,
    col,
    count_all,
    dense_rank,
    first_value,
    fn,
    lag,
    last_value,
    lead,
    lit,
    ntile,
    param,
    rank,
    row_number,
    sql_expr,
    star,
    when,
)
from .frame import Bound, Column, Frame, NeedsConnection, Step, sql, table, values

__all__ = [
    "Bound",
    "Column",
    "Connection",
    "Expr",
    "Frame",
    "NeedsConnection",
    "Step",
    "__version__",
    "coalesce",
    "col",
    "connect",
    "count_all",
    "dbapi",
    "dense_rank",
    "duckdb_version",
    "exceptions",
    "first_value",
    "fn",
    "lag",
    "last_value",
    "lead",
    "library_version",
    "lit",
    "ntile",
    "param",
    "rank",
    "row_number",
    "sql",
    "sql_expr",
    "star",
    "table",
    "values",
    "when",
]

#: Version of this package.
__version__: str = _package_version("duckdb")


def duckdb_version() -> str:
    """Version of the DuckDB engine this package is linked against.

    Distinct from :data:`__version__`: the package and the engine are versioned
    separately and released on their own cadences.
    """
    return library_version()
