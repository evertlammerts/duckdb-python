"""What the installed wheel must contain and how it must be tagged; an editable install is skipped instead."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import duckdb

from ._support import gil_enabled, installed_as_wheel

ENGINE_LIBRARY_NAMES = ("libduckdb.dylib", "libduckdb.so", "duckdb.dll")


@pytest.mark.skipif(not installed_as_wheel(), reason="editable install redirects the extension")
def test_engine_ships_beside_the_extension() -> None:
    # DuckDB is linked, never dlopened, and found through a run path into the extension's own directory.
    package_dir = Path(duckdb.__file__).parent
    found = [name for name in ENGINE_LIBRARY_NAMES if (package_dir / name).exists()]
    assert found, f"no engine library in {package_dir}"


@pytest.mark.skipif(not gil_enabled(), reason="free-threaded builds are not abi3")
@pytest.mark.skipif(sys.version_info < (3, 12), reason="nanobind has no stable ABI below 3.12")
def test_extension_is_built_against_the_stable_abi() -> None:
    # A stable-ABI build silently does nothing without CMake's SABIModule, so only the file name is honest.
    assert ".abi3." in Path(duckdb._duckdb.__file__).name


@pytest.mark.skipif(sys.version_info >= (3, 12), reason="3.12 and up build against the stable ABI")
def test_extension_is_version_specific_below_3_12() -> None:
    # There is no stable ABI below 3.12, so the name must carry the interpreter version.
    name = Path(duckdb._duckdb.__file__).name
    assert ".abi3." not in name
    assert f"{sys.version_info.major}{sys.version_info.minor}" in name


@pytest.mark.freethreaded
def test_gil_stays_disabled_after_import() -> None:
    # Without the module's free-threading declaration CPython silently switches the lock back on at import.
    assert not gil_enabled()
