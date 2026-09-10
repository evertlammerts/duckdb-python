"""Checks on the running interpreter and on the duckdb install, shared by the test modules."""

from __future__ import annotations

import sys


def gil_enabled() -> bool:
    """Whether this interpreter holds the global interpreter lock, which every version before 3.13 always does."""
    probe = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else bool(probe())


def installed_as_wheel() -> bool:
    """Whether duckdb is installed as a wheel; an editable install puts its files in the build directory instead."""
    import sysconfig
    from pathlib import Path

    import duckdb

    package_dir = Path(duckdb.__file__).resolve().parent
    roots = {sysconfig.get_path(name) for name in ("purelib", "platlib")}
    return any(root and package_dir.is_relative_to(Path(root).resolve()) for root in roots)
