"""A process that ends with an open result over a pandas object column must exit, not hang.

Each case runs a script in a fresh interpreter and checks that it exits within the deadline, with the script's own
exit status and no fatal error.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pandas")

ROWS = 1_000_000
DEADLINE = 60
BODY_DONE = "BODY_DONE"

FRAME = f"""
import duckdb
import pandas as pd

df = pd.DataFrame({{"s": pd.Series([f"v{{i}}" for i in range({ROWS})], dtype=object), "i": range({ROWS})}})
"""

DECIMAL_FRAME = f"""
import decimal
import duckdb
import pandas as pd

df = pd.DataFrame({{"d": pd.Series([decimal.Decimal(i) for i in range({ROWS})], dtype=object), "i": range({ROWS})}})
"""


def _run(body: str, *, head: str = FRAME) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(head) + textwrap.dedent(body) + f"\nprint({BODY_DONE!r}, flush=True)\n"
    try:
        return subprocess.run(
            [sys.executable, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=DEADLINE,
            check=False,
            env={"SKBUILD_EDITABLE_VERBOSE": "0", **__import__("os").environ},
        )
    except subprocess.TimeoutExpired as hang:
        stdout = hang.stdout.decode() if isinstance(hang.stdout, bytes) else hang.stdout or ""
        stderr = hang.stderr.decode() if isinstance(hang.stderr, bytes) else hang.stderr or ""
        pytest.fail(f"the process hung at exit\nstdout:\n{stdout[-2000:]}\nstderr:\n{stderr[-2000:]}")


def _assert_exit(proc: subprocess.CompletedProcess[str], *, status: int = 0, body_done: bool = True) -> None:
    report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    assert proc.returncode == status, report
    assert (BODY_DONE in proc.stdout.splitlines()) == body_done, report
    assert "Fatal Python error" not in proc.stderr, report


OPEN_RESULT_CASES = [
    pytest.param(
        """
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        """,
        id="dbapi_fetchone",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        print(len(con.execute("SELECT * FROM df").fetchmany(10)))
        """,
        id="dbapi_fetchmany",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        cur = con.cursor()
        print(cur.execute("SELECT * FROM df").fetchone())
        """,
        id="cursor",
    ),
    pytest.param(
        """
        print(duckdb.execute("SELECT * FROM df").fetchone())
        """,
        id="default_connection",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        con.execute("SELECT * FROM df")
        print(len(con.fetch_df_chunk()))
        """,
        id="df_chunk",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        capsule = con.sql("SELECT * FROM df").__arrow_c_stream__()
        print(type(capsule).__name__)
        """,
        id="relation_arrow_stream",
    ),
    pytest.param(
        """
        con = duckdb.connect()
        reader = con.sql("SELECT * FROM df").to_arrow_reader(1000)
        print(reader.read_next_batch().num_rows)
        """,
        id="arrow_reader",
        marks=pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="needs pyarrow"),
    ),
    pytest.param(
        """
        con = duckdb.connect(config={"threads": 4})
        print(con.execute("SELECT * FROM df WHERE i % 2 = 0").fetchone())
        """,
        id="filter_four_threads",
    ),
    pytest.param(
        """
        import sys
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        print("BODY_DONE", flush=True)
        sys.exit(0)
        """,
        id="sys_exit",
    ),
]


class TestOpenResultAtExit:
    @pytest.mark.parametrize("body", OPEN_RESULT_CASES)
    def test_exits_with_an_open_result_over_an_object_column(self, body: str) -> None:
        _assert_exit(_run(body))

    def test_exits_with_an_open_result_over_decimal_cells(self) -> None:
        body = """
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        """
        _assert_exit(_run(body, head=DECIMAL_FRAME))

    def test_an_exception_after_the_fetch_exits_with_status_one(self) -> None:
        body = """
        con = duckdb.connect()
        print(con.execute("SELECT * FROM df").fetchone())
        raise RuntimeError("boom")
        """
        proc = _run(body)
        _assert_exit(proc, status=1, body_done=False)
        assert "RuntimeError: boom" in proc.stderr

    def test_a_result_open_on_a_busy_thread_does_not_block_the_exit(self) -> None:
        body = """
        import threading
        con = duckdb.connect()
        started = threading.Event()

        def work():
            cursor = con.cursor()
            cursor.execute("SELECT * FROM df")
            started.set()
            try:
                while True:
                    cursor.fetchmany(1)
            except duckdb.Error:
                pass

        threading.Thread(target=work, daemon=True).start()
        started.wait()
        """
        _assert_exit(_run(body))


class TestAfterTheHook:
    def test_a_later_exit_handler_still_runs_a_statement(self) -> None:
        # Registered before the import, so it runs after duckdb's exit hook
        handler = """
        import atexit

        state = {}

        def after():
            print("AFTER", state["con"].execute("SELECT count(*) FROM df").fetchone(), flush=True)

        atexit.register(after)
        """
        head = textwrap.dedent(handler) + FRAME
        body = """
        con = duckdb.connect()
        state["con"] = con
        print(con.execute("SELECT * FROM df").fetchone())
        """
        proc = _run(body, head=head)
        _assert_exit(proc)
        assert f"AFTER ({ROWS},)" in proc.stdout, proc.stdout


class TestUnchangedExits:
    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(
                """
                con = duckdb.connect()
                print(con.execute("SELECT * FROM df").fetchone())
                con.close()
                """,
                id="closed_before_exit",
            ),
            pytest.param(
                """
                con = duckdb.connect()
                print(len(con.execute("SELECT * FROM df").fetchall()))
                """,
                id="fetched_to_the_end",
            ),
            pytest.param(
                """
                def read():
                    con = duckdb.connect()
                    return con.execute("SELECT * FROM df").fetchone()

                print(read())
                """,
                id="connection_in_function",
            ),
        ],
    )
    def test_exits_as_before(self, body: str) -> None:
        _assert_exit(_run(body))
