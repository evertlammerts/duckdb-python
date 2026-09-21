"""The DuckDB Python package: the engine, its errors, and two ways to talk to it.

`duckdb.frame` builds queries as plans and runs them on a connection; `duckdb.dbapi` is PEP 249. They share the
engine and `duckdb.exceptions`, and nothing else.
"""

from importlib.metadata import version as _package_version

from . import dbapi, exceptions, frame
from ._duckdb import library_version

__all__ = [
    "__version__",
    "dbapi",
    "duckdb_version",
    "exceptions",
    "frame",
    "library_version",
]

#: Version of this package.
__version__: str = _package_version("duckdb")


def duckdb_version() -> str:
    """The DuckDB version this package is linked against, versioned separately from `__version__`."""
    return library_version()
