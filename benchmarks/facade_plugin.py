"""Pytest plugin for the cross-client A/B lane. See benchmarks/README.md.

Loaded with `-p facade_plugin` against the OLD client's benchmark files, it
routes their `duckdb.connect` at the migration face, so they measure neo
through the surface they were written for. Never loaded by this suite itself.
"""

import duckdb
from duckdb import compat

duckdb.connect = compat.connect  # type: ignore[assignment]
