"""The DuckDB Python package."""

from importlib.metadata import version as _package_version

from ._duckdb import library_version

__all__ = ["__version__", "duckdb_version", "library_version"]

#: Version of this package.
__version__: str = _package_version("duckdb")


def duckdb_version() -> str:
    """Version of the DuckDB engine this package is linked against.

    Distinct from :data:`__version__`: the package and the engine are versioned
    separately and released on their own cadences.
    """
    return library_version()
