"""Load with `-p facade_plugin` to point `duckdb.connect` at the compatibility API. See benchmarks/README.md."""

import duckdb
from duckdb import compat

duckdb.connect = compat.connect  # type: ignore[assignment]
