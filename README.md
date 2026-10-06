# duckdb (neo)

The DuckDB Python package, rebuilt on the DuckDB C++ API.

A plan is a value: queries are built without a connection and run by passing
one. The package has zero runtime dependencies, and `duckdb.dbapi` is a strict
PEP 249 face over the same engine bindings.

## Layout

- `src/duckdb/` is the Python package: `exceptions`, `dbapi`, the `frame`
  package holding the plan and connection modules, and `_expressions/`, the
  expression builder and its SQL rendering the frame package is built on;
  `_sources/` classifies the Python objects both faces register as tables.
  `_error_codes.py` and, under `_expressions/`, `aggregates.py`,
  `func_namespaces.py` and `keywords.py` are generated from the engine's
  catalog: after editing `scripts/func_namespaces.toml` or moving to a new
  engine, regenerate with the scripts beside the table.
- `src/_duckdb/` is the nanobind extension module over the C++ API.
- `tests/` is the pytest suite. `pytest -m corpus` additionally runs the
  engine's own sqllogictest corpus and needs a duckdb checkout at the loaded
  engine's commit (`DUCKDB_SOURCE`).

## Building

The build links against DuckDB's prebuilt shared library. `scripts/engine.py
fetch` resolves the newest published 2.0 engine (the latest alpha until 2.0.0
is released, the latest 2.0 release after that) and unpacks its library and C
headers into `engine/`; `--install tpch` adds the extension the TPC-H tests
use:

    uv run scripts/engine.py fetch --install tpch
    uv sync --only-group build --no-install-project
    uv sync --no-build-isolation --group build --group dev
    uv run --no-sync pytest

Setting `DUCKDB_ROOT` builds against another engine instead: a directory
holding libduckdb and its C headers side by side, which
`scripts/build_engine_bundle.py <checkout> <dir>` produces from a checkout.
A plain `uv sync` rebuilds after an engine change: the build's cache keys
cover `engine/`, `third_party/duckdb_cpp/` and `DUCKDB_ROOT`. The one blind
spot is a directory outside the repo whose contents change in place under an
unchanged `DUCKDB_ROOT`; rebuild that with `--reinstall-package duckdb`.
A `-DDUCKDB_ROOT` CMake define lasts exactly one configure, and a build
tool's automatic CMake re-run drops it; the environment variable (or
`SKBUILD_CMAKE_DEFINE`, which is passed on every configure) is the override
that persists.

The C++ API the extension is built on is DuckDB source, shipped in
`third_party/duckdb_cpp` and pinned by its `REF` file. It compiles against the
fetched engine's own C headers, so an engine that changed a C signature fails
the build instead of corrupting memory at run time; the pin is the floor of
the supported engine range. `uv run scripts/engine.py update-cpp-api` moves
the pin to the newest published engine. Because `fetch` floats with the
newest engine, a fresh alpha can turn the build red until the pin moves; for
a build that always compiles, fetch the pin's own engine:

    uv run scripts/engine.py fetch --engine "$(cat third_party/duckdb_cpp/REF)" --install tpch

An unreleased engine whose C headers moved past the pin also needs its
`tools/cpp`, passed as `-DDUCKDB_CPP_DIR` through `SKBUILD_CMAKE_DEFINE`.

Design notes, and the record of deliberate divergences from the previous
client, live in the maintainers' notes outside this repository.
