"""Interpreter capability checks shared by the test modules."""

from __future__ import annotations

import sys


def gil_enabled() -> bool:
    """Whether this interpreter is running with the GIL.

    `sys._is_gil_enabled` arrived in 3.13. On 3.12, the oldest interpreter this
    package supports, there is no free-threaded build, so the GIL is always on.
    """
    probe = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else bool(probe())


def installed_as_wheel() -> bool:
    """Whether duckdb is installed as a wheel rather than an editable checkout.

    An editable install redirects the extension and the engine into the build
    directory, so the layout invariants a wheel must satisfy do not hold and
    asserting them would only produce noise.
    """
    import sysconfig
    from pathlib import Path

    import duckdb

    package_dir = Path(duckdb.__file__).resolve().parent
    roots = {sysconfig.get_path(name) for name in ("purelib", "platlib")}
    return any(root and package_dir.is_relative_to(Path(root).resolve()) for root in roots)
