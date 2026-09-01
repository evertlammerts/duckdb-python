"""Ours, not adopted: the switch that runs the old tests against the migration face.

With DUCKDB_COMPAT_FACADE set, `duckdb.connect` is replaced by
`duckdb.compat.connect` before the adopted tests run, previewing what a
drop-in `connect()` would score. The verbatim files under suite/ are never
edited; the patch lives here, one directory up.
"""

import os

if os.environ.get("DUCKDB_COMPAT_FACADE"):
    import duckdb
    from duckdb import compat

    for _name in (
        "connect",
        "execute",
        "sql",
        "query",
        "from_query",
        "default_connection",
        "description",
        "apilevel",
        "paramstyle",
        "threadsafety",
        "BINARY",
        "DATETIME",
        "NUMBER",
        "ROWID",
        "STRING",
        "CatalogException",
        "ConnectionException",
        "InterruptException",
        "InvalidInputException",
        "__formatted_python_version__",
    ):
        setattr(duckdb, _name, getattr(compat, _name))
