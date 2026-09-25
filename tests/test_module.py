"""The module imports, exports what it claims, and pulls in nothing else."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import duckdb

from ._support import installed_as_wheel

# Data libraries a user may have installed; importing this package must never load them on its own.
OPTIONAL_DEPENDENCIES = ("numpy", "pandas", "pyarrow", "polars")


@pytest.mark.skipif(not installed_as_wheel(), reason="editable install redirects the extension")
def test_extension_loads_from_inside_the_package() -> None:
    # A stray duckdb/ directory on sys.path shadows the installed package and confuses everything after.
    extension = Path(duckdb._duckdb.__file__)
    assert extension.parent == Path(duckdb.__file__).parent


def test_all_names_are_importable() -> None:
    for name in duckdb.__all__:
        assert hasattr(duckdb, name), f"__all__ advertises {name!r}, which does not exist"


def test_importing_duckdb_pulls_in_no_optional_dependency() -> None:
    # This interpreter has already imported the test dependencies, so the check must run elsewhere.
    probe = (
        "import sys; import duckdb; "
        f"leaked = [m for m in {OPTIONAL_DEPENDENCIES!r} if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    leaked = result.stdout.strip()
    assert not leaked, f"importing duckdb dragged in: {leaked}"


def test_the_package_works_with_every_optional_dependency_absent() -> None:
    # Blocking the modules in a fresh interpreter makes every import of them fail as on a bare install.
    probe = (
        "import sys\n"
        f"for name in {OPTIONAL_DEPENDENCIES!r}:\n"
        "    sys.modules[name] = None\n"
        "import duckdb, duckdb.dbapi, duckdb._sources, duckdb._sources.arrow, duckdb._sources.pyarrow, "
        "duckdb._sources.pandas, duckdb._sources.polars, duckdb._sources.numpy, duckdb._expressions.arrow, "
        "duckdb._expressions.polars\n"
        "from duckdb.frame import col, connect\n"
        "con = connect()\n"
        "with con._execute('SELECT 42') as result:\n"
        "    assert result.fetch_all() == [(42,)]\n"
        "class Exporter:\n"
        "    def __arrow_c_stream__(self, requested_schema=None):\n"
        "        return None\n"
        "con.register('plain', Exporter())\n"
        "assert (col('a') > 1).fragment() == '(\"a\" > 1)'\n"
        "class Frame:\n"
        "    pass\n"
        "Frame.__module__, Frame.__name__ = 'pandas.core.frame', 'DataFrame'\n"
        "try:\n"
        "    con.register('df', Frame())\n"
        "except TypeError as error:\n"
        "    assert 'needs pandas' in str(error), error\n"
        "else:\n"
        "    raise AssertionError('a pandas frame registered without pandas')\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ok", result.stderr


def test_a_pandas_frame_registers_and_queries_with_pyarrow_absent() -> None:
    """Without pyarrow every pandas frame is native, so the package reads one straight off its numpy columns."""
    probe = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"
        "import duckdb\n"
        "import pandas as pd\n"
        "from duckdb.frame import connect\n"
        "con = connect()\n"
        "frame = pd.DataFrame({'i': range(5), 'f': [1.5, 2.5, 3.5, 4.5, 5.5], 's': [str(i) for i in range(5)]})\n"
        "con.register('df', frame)\n"
        "with con._execute('SELECT count(*), sum(i), sum(f), max(s) FROM df') as result:\n"
        "    assert result.fetch_all() == [(5, 10, 17.5, '4')]\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "ok", result.stderr
