"""Invariants about what the built artifact contains and how it is tagged.

These describe the installed wheel. An editable install redirects the extension
and the engine elsewhere, so they are skipped there rather than asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import duckdb

from ._support import gil_enabled, installed_as_wheel

ENGINE_LIBRARY_NAMES = ("libduckdb.dylib", "libduckdb.so", "duckdb.dll")


@pytest.mark.skipif(not installed_as_wheel(), reason="editable install redirects the extension")
def test_engine_ships_beside_the_extension() -> None:
    # The engine is linked, never dlopened, and is found through an rpath
    # pointing at the extension's own directory.
    package_dir = Path(duckdb.__file__).parent
    found = [name for name in ENGINE_LIBRARY_NAMES if (package_dir / name).exists()]
    assert found, f"no engine library in {package_dir}"


@pytest.mark.skipif(not gil_enabled(), reason="free-threaded builds are not abi3")
def test_extension_is_built_against_the_stable_abi() -> None:
    # STABLE_ABI silently does nothing if CMake was not asked for
    # Development.SABIModule. The wheel is still tagged abi3 either way, so the
    # filename is the only honest signal.
    assert ".abi3." in Path(duckdb._duckdb.__file__).name


@pytest.mark.freethreaded
def test_gil_stays_disabled_after_import() -> None:
    # The module declares free-threading support; without that declaration
    # CPython silently re-enables the GIL on import.
    assert not gil_enabled()
