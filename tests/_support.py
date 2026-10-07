"""Checks on the running interpreter and on the duckdb install, and tzinfo probes, shared by the test modules."""

from __future__ import annotations

import datetime
import sys
from typing import Any


class NoOffset(datetime.tzinfo):
    """A time zone that gives no offset, which makes a datetime carrying it naive, as Python defines it."""

    def utcoffset(self, moment: datetime.datetime | None) -> None:
        return None

    def dst(self, moment: datetime.datetime | None) -> None:
        return None

    def tzname(self, moment: datetime.datetime | None) -> str:
        return "none"


class Failing(datetime.tzinfo):
    """A time zone whose offset computation itself raises the given error.

    A fresh instance per call: a stored exception would keep its traceback, whose frames hold the connection under
    test alive past interpreter exit.
    """

    def __init__(self, error: type[Exception], message: str) -> None:
        self.error = error
        self.message = message

    def utcoffset(self, moment: datetime.datetime | None) -> datetime.timedelta:
        raise self.error(self.message)

    def dst(self, moment: datetime.datetime | None) -> None:
        return None

    def tzname(self, moment: datetime.datetime | None) -> str:
        return "failing"


class Emptying(datetime.tzinfo):
    """UTC, emptying a list or dict each time it gives an offset: a callback that changes a value while it converts."""

    def __init__(self, container: list[Any] | dict[Any, Any]) -> None:
        self.container = container

    def utcoffset(self, moment: datetime.datetime | None) -> datetime.timedelta:
        self.container.clear()
        return datetime.timedelta(0)

    def dst(self, moment: datetime.datetime | None) -> None:
        return None

    def tzname(self, moment: datetime.datetime | None) -> str:
        return "emptying"


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
