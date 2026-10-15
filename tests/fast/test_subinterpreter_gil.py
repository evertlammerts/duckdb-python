"""duckdb keeps working in the main interpreter after the process has created a subinterpreter.

Creating any subinterpreter, even one that never imports duckdb, makes PyGILState_Check report the GIL as held on
every thread for the rest of the process. That switch cannot be undone, so each case runs in a fresh process.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

BODY_DONE = "BODY_DONE"
STARTED = "STARTED"
PRECONDITION = "PyGILState_Check with the GIL released: "
INTERPRETERS = "_interpreters" if sys.version_info >= (3, 13) else "_xxsubinterpreters"


def _needs(module: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(importlib.util.find_spec(module) is None, reason=f"{module} is not installed")


# A core dump of a process with hundreds of engine threads takes minutes to write.
NO_CORE_DUMP = """
import resource

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
"""

# A CDLL call drops the GIL, so the answer shows whether the subinterpreter switched the check off.
SUBINTERPRETER = f"""
import ctypes
import importlib

interpreters = importlib.import_module({INTERPRETERS!r})
interpreters.destroy(interpreters.create())
answer = ctypes.CDLL(None, handle=getattr(sys, "dllhandle", None)).PyGILState_Check()
print({PRECONDITION!r} + str(answer))
if answer != 1:
    sys.exit(0)
"""

# Another Python thread holds the GIL while duckdb asks whether this one does; before 3.12 that tells the check
# apart from one asking only whether some thread state is current. It starts after the imports, which it would
# slow down by a GIL handover on every file access.
SPINNER = """
import threading

stop = threading.Event()


def spin():
    while not stop.is_set():
        pass


spinner = threading.Thread(target=spin, daemon=True)
spinner.start()
"""

SPINNER_DONE = """
stop.set()
spinner.join()
"""

PANDAS_IMPORTS = """
import pandas as pd
"""

PANDAS_SCAN = f"""
df = pd.DataFrame({{"a": [1, 2, 3]}})
print({STARTED!r})
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
print(con.sql("SELECT sum(a) FROM df").fetchall())
"""

ARROW_IMPORTS = """
import pyarrow

from duckdb.sqltypes import BIGINT
"""

# The UDF makes ending the query wait for a task that needs the GIL.
ARROW_READER = f"""
con = duckdb.connect()
con.execute("SET enable_progress_bar = false")
con.execute("SET max_streaming_buffer_size='1MB'")
con.create_function("identity", lambda i: i, [BIGINT], BIGINT)
reader = con.sql("SELECT identity(i) AS i FROM range(2000000) t(i)").to_arrow_reader(1024)
print({STARTED!r})
print(sum(len(reader.read_next_batch()) for _ in range(50)))
del reader
print(con.execute("SELECT 42").fetchall())
"""


def _text(stream: str | bytes | None) -> str:
    return stream.decode() if isinstance(stream, bytes) else stream or ""


def _run(imports: str, body: str) -> subprocess.CompletedProcess[str]:
    script = "".join(
        textwrap.dedent(part)
        for part in (
            NO_CORE_DUMP if sys.platform != "win32" else "",
            "import sys\n",
            SUBINTERPRETER,
            "import duckdb\n",
            imports,
            SPINNER,
            body,
            SPINNER_DONE,
            f"print({BODY_DONE!r})\n",
        )
    )
    try:
        return subprocess.run(
            [sys.executable, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as hang:
        started = _text(hang.stdout)
        where = "after" if STARTED in started else "before"
        msg = f"the child hung {where} reaching the statement under test"
        msg += f"\nstdout:\n{started[-2000:]}\nstderr:\n{_text(hang.stderr)[-2000:]}"
        raise AssertionError(msg) from hang


@_needs(INTERPRETERS)
@pytest.mark.parametrize(
    ("imports", "body", "expected"),
    [
        pytest.param(PANDAS_IMPORTS, PANDAS_SCAN, ["[(6,)]"], id="pandas_scan", marks=_needs("pandas")),
        pytest.param(ARROW_IMPORTS, ARROW_READER, ["51200", "[(42,)]"], id="arrow_reader", marks=_needs("pyarrow")),
    ],
)
def test_works_after_a_subinterpreter(imports: str, body: str, expected: list[str]) -> None:
    proc = _run(imports, body)
    report = f"exit status {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    answers = [line.removeprefix(PRECONDITION) for line in proc.stdout.splitlines() if line.startswith(PRECONDITION)]
    assert answers, report
    if answers != ["1"]:
        pytest.skip("creating a subinterpreter no longer switches PyGILState_Check off, so nothing here is tested")
    assert proc.returncode == 0, report
    assert proc.stdout.splitlines()[-len(expected) - 1 :] == [*expected, BODY_DONE], report
