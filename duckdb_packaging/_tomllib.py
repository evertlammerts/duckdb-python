"""Copied from scikit-build-core. Makes sure we can read toml on Python versions <= 3.10."""
from __future__ import annotations

import sys

if sys.version_info < (3, 11):
    from tomli import load, loads
else:
    from tomllib import load, loads

__all__ = ["load", "loads"]


def __dir__() -> list[str]:
    return __all__
