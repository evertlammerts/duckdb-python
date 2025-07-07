"""DuckDB Python packaging, versioning, and build tooling.

Requires Python >= 3.5 and does not work on mobile platforms due to the use of the `subprocess` module.
"""
import functools
from typing import Callable

from duckdb_packaging.setuptools_scm_version import handle_version_overrides


def process_force_version(func: Callable):
    """Decorator that handles force version overrides with OVERRIDE_GIT_DESCRIBE."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        handle_version_overrides()
        return func(*args, **kwargs)

    return wrapper
