"""The capsule protocol without pyarrow: polars consumes a bound plan directly, and the reader names the gap.

Apart from the main Arrow egress tests, since those require pyarrow and this behavior matters most without it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.requires("polars")


def test_polars_consumes_the_capsule_and_the_reader_names_the_gap() -> None:
    probe = textwrap.dedent(
        """
        import sys

        sys.modules["pyarrow"] = None
        import polars as pl

        import duckdb

        con = duckdb.frame.connect()
        bound = duckdb.frame.sql("SELECT i AS n FROM range(1000) t(i)").on(con)
        frame = pl.DataFrame(bound)
        assert frame.height == 1000 and frame["n"].to_list()[:3] == [0, 1, 2]
        try:
            duckdb.frame.sql("SELECT 1").to_reader(con)
            raise SystemExit("to_reader worked without pyarrow")
        except ImportError as error:
            assert "pyarrow" in str(error)
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
