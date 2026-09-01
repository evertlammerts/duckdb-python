# duckdb (neo)

The DuckDB Python package, rebuilt on the DuckDB C++ API.

A plan is a value: queries are built without a connection and run by passing
one. The package has zero runtime dependencies, and `duckdb.dbapi` is a strict
PEP 249 face over the same engine bindings.

## Layout

- `src/duckdb/` is the Python package. `_aggregates.py`, `_error_codes.py` and
  `_func_namespaces.py` are generated: edit `scripts/func_namespaces.toml` or
  bump `engine.pin`, then regenerate with the scripts beside the table.
- `src/_duckdb/` is the nanobind extension module over the C++ API.
- `tests/` is the pytest suite. `pytest -m corpus` additionally runs the
  engine's own sqllogictest corpus and needs a duckdb checkout
  (`DUCKDB_SOURCE`).

## Building

The engine is neither vendored nor downloaded: DuckDB does not yet publish a
libduckdb carrying the V2 C API, so the build links against one built from the
SHA in `engine.pin`. `scripts/build_engine_bundle.py` turns a checkout into a
self-contained bundle, and `SKBUILD_CMAKE_DEFINE` points the build at it:

    python scripts/build_engine_bundle.py <duckdb-checkout> engine-bundle
    export SKBUILD_CMAKE_DEFINE="DUCKDB_CPP_DIR=$PWD/engine-bundle;DUCKDB_ROOT=$PWD/engine-bundle"
    uv sync --only-group build --no-install-project
    uv sync --no-build-isolation --group build --group dev
    uv run --no-sync pytest

Building against a raw checkout works too; see the `DUCKDB_CPP_DIR` and
`DUCKDB_INCLUDE_DIR` notes in `CMakeLists.txt`.

Design notes, and the record of deliberate divergences from the previous
client, live in the maintainers' notes outside this repository.
