"""The package version and the engine version are separate things."""

from __future__ import annotations

import re
from importlib.metadata import version as package_metadata_version

import duckdb

PEP440 = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def test_package_version_is_pep440() -> None:
    assert PEP440.match(duckdb.__version__), duckdb.__version__


def test_package_version_matches_installed_metadata() -> None:
    assert duckdb.__version__ == package_metadata_version("duckdb")


def test_engine_version_is_reported() -> None:
    engine = duckdb.duckdb_version()
    assert engine, "engine reported an empty version"
    # A depth-1 checkout with no tags makes DuckDB's `git describe` fail, and
    # the engine then reports something that is not a version at all.
    assert re.match(r"^v?\d+\.\d+\.\d+", engine), engine


def test_package_and_engine_versions_are_not_conflated() -> None:
    # They are released on separate cadences. Reporting one as the other is a
    # regression that reads as harmless until someone version-gates on it.
    assert duckdb.__version__ != duckdb.duckdb_version()
