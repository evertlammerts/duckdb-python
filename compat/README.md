# Compat coverage: the old client's own tests

A verbatim copy of test files from `duckdb/duckdb-python` `main`
(`2ba111556c0a`, copied 2026-09-01), run against this client to measure
drop-in compatibility. This is a measurement, not a gate: the repo's own
suite in `tests/` gates changes; this one only reports.

Rules:

- Files under `suite/` are never edited to pass. A failure here is data:
  surface this client does not have, a deliberate divergence for the
  behavior-change log (`BEHAVIOR.md`, beside this file), or a real bug. A
  test asserting a recorded divergence is xfailed from `conftest.py` with a
  reason naming its log entry.
- The slice grows deliberately, file by file. Copy new files verbatim from
  the same source and record the source SHA here.
- Nothing here implies the old surface will be reproduced as-is. What the
  old suite exercises and what this client promises are separate decisions;
  the report just keeps the distance honest.

Run it:

    uv run --no-sync python scripts/compat_report.py
    uv run --no-sync python scripts/compat_report.py --facade

The report clusters outcomes (passed, missing surface, behavior, skipped),
prints the top failure signatures, and always exits 0. `--facade` patches
`duckdb.connect` and the old module-level names to `duckdb.compat` through
`compat/conftest.py`, previewing what a drop-in 2.0 would score; the
verbatim files are never touched. A path argument restricts the run to the
suite files under it and is not passed to pytest; every other argument is
passed through, so `scripts/compat_report.py suite/test_result.py -v` runs
that one file verbosely.

Current slice: 36 files from `tests/fast/`, `tests/fast/api/` and
`tests/fast/relational_api/`: the connection, statement, transaction,
result and interrupt files, the DB-API set (`test_dbapi*.py`,
`test_cursor.py`, `test_dbapi_fetch.py`), and the relational-API set
(groupings, joins, pivot, aggregations, windows, close, description,
functions, query, table_function) with the data file it reads under
`suite/data/`. The old suite's `conftest.py` rides along verbatim. The `compat` dependency
group carries what these files import at module level (numpy, pandas); a
third-party module missing from the venv is reported as `environment`, so
it never counts against this client's surface. Deliberately excluded so
far: files importing pyarrow, polars, torch or tensorflow (wave-2 surface),
files needing the old build tree or its data files, and
`test_query_progress.py` (needs the pytest-reraise plugin).
