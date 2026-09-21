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
