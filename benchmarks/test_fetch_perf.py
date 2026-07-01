"""Standalone CodSpeed benchmark module — NOT integrated (not in pyproject, not in CI, not committed).

Purpose: A/B the binding-layer perf between the two builds (pybind11 `main` vs nanobind cutover), in particular
the narrow-column `fetchall` regression. Run the SAME file under each build's interpreter and compare:

  M=/Users/evert/projects/duckdb-python/main/.venv-release/bin/python
  C=/Users/evert/projects/duckdb-python/wt-cutover/.venv-release/bin/python
  cd /Users/evert/projects/duckdb-python/wt-cutover
  $M -m pytest benchmarks/test_fetch_perf.py --codspeed --codspeed-mode=walltime -o addopts=
  $C -m pytest benchmarks/test_fetch_perf.py --codspeed --codspeed-mode=walltime -o addopts=

NOTE: macOS arm64 has no Valgrind, so only `--codspeed-mode=walltime` works locally (wall-clock stats). The
deterministic instruction-count mode (`--codspeed-mode=simulation`) needs Linux + the CodSpeed instrument
(CI, or `codspeed run` in a Linux container). In CI/cloud, CodSpeed compares each run against a git baseline;
locally we get the same benchmark workflow but A/B by running the file under the two interpreters by hand.
"""

import duckdb
import pytest


@pytest.fixture
def con():
    c = duckdb.connect()
    yield c
    c.close()


def _bench_fetchall(benchmark, con, query):
    con.execute(query).fetchall()  # warm the engine before measuring
    benchmark(lambda: con.execute(query).fetchall())


def test_fetchall_int(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a FROM range(200000) t(i)")


def test_fetchall_smallint(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT (i % 100)::INTEGER AS a FROM range(200000) t(i)")


def test_fetchall_double(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT (i * 1.5)::DOUBLE AS a FROM range(200000) t(i)")


def test_fetchall_2int(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range(200000) t(i)")


def test_fetchall_str(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT ('str_value_' || i) AS s FROM range(100000) t(i)")


def test_fetchall_mixed(benchmark, con):
    query = (
        "SELECT i::BIGINT AS bi, ('str_' || i) AS s, [i, i + 1, i + 2] AS lst, "
        "{'a': i, 'b': i + 1} AS st FROM range(50000) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchone_iter(benchmark, con):
    query = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(100000) t(i)"

    def run():
        rel = con.execute(query)
        while rel.fetchone() is not None:
            pass

    benchmark(run)


# --------------------------------------------------------------------------- #
# ADDED: small-N instruction-count-gate variants (the narrow-numeric fixed-cost path, noise-free at range(2048)
# under simulation mode in CI), expensive scalar OUT-row types (timestamptz pytz-per-row, blob, null-heavy), a
# heterogeneous per-cell-dispatch row (hugeint+uuid+decimal128+varchar, distinct from homogeneous columns), and
# the batched fetchmany loop.
# --------------------------------------------------------------------------- #


def test_fetchall_int_gate(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a FROM range(2048) t(i)")


def test_fetchall_2int_gate(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT i::BIGINT AS a, (i + 1)::BIGINT AS b FROM range(2048) t(i)")


def test_fetchall_null_heavy(benchmark, con):
    _bench_fetchall(
        benchmark, con, "SELECT CASE WHEN i % 2 = 0 THEN NULL ELSE i::BIGINT END FROM range(200000) t(i)"
    )


def test_fetchall_timestamptz(benchmark, con):
    _bench_fetchall(
        benchmark, con, "SELECT (TIMESTAMPTZ '2020-01-01' + (i * INTERVAL 1 SECOND)) FROM range(100000) t(i)"
    )


def test_fetchall_decimal128(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT ((i * 1.5)::DECIMAL(28, 6)) FROM range(200000) t(i)")


def test_fetchall_blob(benchmark, con):
    _bench_fetchall(benchmark, con, "SELECT ('blob_value_' || i)::BLOB FROM range(100000) t(i)")


def test_fetchall_mixed_wide(benchmark, con):
    # heterogeneous row -> per-cell type dispatch in the Fetchone column loop (distinct branch/cache profile
    # from the homogeneous single-type columns above)
    query = (
        "SELECT (i::HUGEINT * 1000000000000) AS h, gen_random_uuid() AS u, "
        "((i * 1.5)::DECIMAL(28, 6)) AS d, ('string_' || i) AS s FROM range(100000) t(i)"
    )
    _bench_fetchall(benchmark, con, query)


def test_fetchmany_batched(benchmark, con):
    query = "SELECT i::BIGINT AS a, (i * 1.5)::DOUBLE AS b FROM range(100000) t(i)"

    def run():
        rel = con.execute(query)
        while True:
            rows = rel.fetchmany(10_000)
            if not rows:
                break

    benchmark(run)


def test_expr_many(benchmark):
    def run():
        out = []
        for i in range(2000):
            col = duckdb.ColumnExpression(f"col_{i}")
            const = duckdb.ConstantExpression(i)
            out.append(((col + const) * duckdb.ConstantExpression(2)).alias(f"a{i}"))
        return len(out)

    benchmark(run)
